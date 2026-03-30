"""
HTTP-обёртка над llama-server с интерфейсом совместимым с llama_cpp.Llama.

Позволяет запускать LLM inference на хосте Windows (V100 TCC-режим, недоступен в WSL2/Docker)
через OpenAI-compatible completions API, пока RAG pipeline работает внутри Docker.

llama-server запускается на хосте:
    llama-server.exe -hf unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M \\
        -c 16384 --parallel 2 --flash-attn on \\
        --cache-type-k q8_0 --cache-type-v q8_0 \\
        -ngl 99 --main-gpu 0 --host 0.0.0.0 --port 8080 \\
        --jinja --reasoning-budget 0

Docker-контейнер обращается по: http://host.docker.internal:8080/v1/completions
"""

import logging
from typing import Any, Dict, List, Optional

import requests

from core.observability import observe_llm_call

logger = logging.getLogger(__name__)


class LlamaServerClient:
    """Тонкая обёртка над /v1/completions с сигнатурой, совместимой с llama_cpp.Llama.

    Поддерживает все параметры, которые использует agent_service.py и query_planner_service.py:
    temperature, top_p, top_k, repeat_penalty, stop, seed, logit_bias, grammar (GBNF).
    """

    def __init__(self, base_url: str, model: str = "local", timeout: int = 120):
        """
        Args:
            base_url: URL llama-server, например "http://host.docker.internal:8080"
            model:    имя модели для поля model в запросе (для логов, не влияет на inference)
            timeout:  таймаут HTTP-запроса в секундах
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._session = requests.Session()
        logger.info(f"LlamaServerClient: base_url={self.base_url}, model={self.model}")

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        grammar: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Вызывает /v1/completions и возвращает ответ в формате llama_cpp.

        Returns:
            {"choices": [{"text": "...", "finish_reason": "stop"}], ...}
        """
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
        }
        if stop:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed
        if grammar:
            payload["grammar"] = grammar
        if logit_bias:
            # llama_cpp использует dict {token_id: bias},
            # llama-server ожидает список пар [[token_id, bias], ...]
            payload["logit_bias"] = [
                [int(tid), float(bias)] for tid, bias in logit_bias.items()
            ]

        with observe_llm_call(
            name="llm_completion", model=self.model, input=prompt[:500],
        ) as span:
            resp = self._session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if span:
                usage = data.get("usage", {})
                span.update(
                    model=data.get("model", self.model),
                    output=data.get("choices", [{}])[0].get("text", "")[:500],
                    usage={
                        "input": usage.get("prompt_tokens", 0),
                        "output": usage.get("completion_tokens", 0),
                        "total": usage.get("total_tokens", 0),
                    },
                )
            return data

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        presence_penalty: float = 1.5,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Вызывает `/v1/chat/completions` с поддержкой native function calling.

        `__call__()` сохраняется для legacy `/v1/completions` сценариев
        (`qa_service`, `query_planner_service`). Этот метод используется
        агентом с `messages` и `tools` schema.
        """
        payload: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            # Qwen3 thinking mode конфликтует с tool_calls в history
            # (ошибка "Assistant response prefill is incompatible with enable_thinking").
            # Для agent pipeline thinking не нужен — отключаем на обоих уровнях:
            # top-level для llama-server и chat_template_kwargs для jinja template.
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.model:
            payload["model"] = self.model
        if tools:
            payload["tools"] = tools
        if stop:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed

        with observe_llm_call(
            name="llm_chat_completion",
            model=self.model,
            input=messages,
            metadata={"tools_count": len(tools) if tools else 0},
        ) as span:
            resp = self._session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )

            # Retry при 400: context overflow → обрезаем messages,
            # prefill/thinking конфликт → убираем enable_thinking.
            if resp.status_code == 400:
                error_text = resp.text[:500]
                logger.warning("LLM chat_completion 400 (will retry): %s", error_text)

                # Обрезаем messages до последних 4 (system + user + 2 recent)
                if len(payload["messages"]) > 4:
                    payload["messages"] = payload["messages"][:2] + payload["messages"][-2:]
                    logger.info("Retrying with trimmed messages: %d → %d", len(messages), 4)

                # Убираем enable_thinking на retry
                payload.pop("enable_thinking", None)

                resp = self._session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )

            if resp.status_code >= 400:
                logger.error(
                    "LLM chat_completion %d: %s",
                    resp.status_code,
                    resp.text[:500],
                )
            resp.raise_for_status()
            data = resp.json()
            if span:
                usage = data.get("usage", {})
                span.update(
                    model=data.get("model", self.model),
                    output=data.get("choices", [{}])[0].get("message", {}),
                    usage={
                        "input": usage.get("prompt_tokens", 0),
                        "output": usage.get("completion_tokens", 0),
                        "total": usage.get("total_tokens", 0),
                    },
                )
            return data

    def tokenize(self, text: bytes, add_bos: bool = True) -> List[int]:
        """Токенизирует текст через /tokenize endpoint.

        Метод оставлен для совместимости со старыми вызовами и отладкой токенизации.
        При ошибке возвращает пустой список.
        """
        try:
            resp = self._session.post(
                f"{self.base_url}/tokenize",
                json={
                    "content": text.decode("utf-8", errors="replace"),
                    "add_special": add_bos,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            tokens = data.get("tokens", [])
            # llama-server возвращает плоский список int-ов
            if tokens and isinstance(tokens[0], int):
                return tokens
            # fallback: список объектов {"id": N}
            return [t["id"] for t in tokens if isinstance(t, dict)]
        except Exception as e:
            logger.debug(f"LlamaServerClient.tokenize failed (non-critical): {e}")
            return []
