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

        resp = self._session.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

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
        }
        if self.model:
            payload["model"] = self.model
        if tools:
            payload["tools"] = tools
        if stop:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed

        resp = self._session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

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
