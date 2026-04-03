"""
HTTP wrapper over llama-server with a llama_cpp.Llama-compatible interface.

Runs LLM inference on Windows host (V100 TCC mode, inaccessible from WSL2/Docker)
via OpenAI-compatible completions API. Docker container connects via
http://host.docker.internal:8080/v1/completions.
"""

import logging
from typing import Any

import requests

from core.observability import observe_llm_call

logger = logging.getLogger(__name__)


class LlamaServerClient:
    """Thin wrapper over llama-server /v1/completions and /v1/chat/completions.

    Supports both legacy completions API (QAService, QueryPlannerService)
    and native function-calling chat API (AgentService).
    """

    def __init__(self, base_url: str, model: str = "local", timeout: int = 120):
        """
        Args:
            base_url: llama-server URL, e.g. "http://host.docker.internal:8080"
            model:    model name for request payload (logging only, no effect on inference)
            timeout:  HTTP request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._session = requests.Session()
        logger.info("LlamaServerClient: base_url=%s, model=%s", self.base_url, self.model)

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: list[str] | None = None,
        seed: int | None = None,
        logit_bias: dict[int, float] | None = None,
        grammar: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Call /v1/completions and return response in llama_cpp format.

        Returns:
            {"choices": [{"text": "...", "finish_reason": "stop"}], ...}
        """
        payload: dict[str, Any] = {
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
                    usage_details={
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                )
            return data

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        presence_penalty: float = 1.5,
        stop: list[str] | None = None,
        seed: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Call /v1/chat/completions with native function calling support.

        Used by AgentService with messages + tools schema.
        See also __call__() for /v1/completions (QAService, QueryPlannerService).
        """
        payload: dict[str, Any] = {
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
        if response_format:
            payload["response_format"] = response_format
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

                # Обрезаем messages, сохраняя compose_context pair если есть
                if len(payload["messages"]) > 4:
                    head = payload["messages"][:2]
                    tail = payload["messages"][2:]
                    # Ищем последний compose_context tool message + его assistant
                    compose_pair: list = []
                    for i in range(len(tail) - 1, -1, -1):
                        if tail[i].get("role") == "tool" and tail[i].get("name") == "compose_context":
                            compose_pair = [tail[i]]
                            # Берём предшествующий assistant message
                            if i > 0 and tail[i - 1].get("role") == "assistant":
                                compose_pair = [tail[i - 1], tail[i]]
                            break
                    if compose_pair:
                        payload["messages"] = head + compose_pair
                    else:
                        payload["messages"] = head + tail[-2:]
                    logger.info("Retrying with trimmed messages: %d → %d", len(messages), len(payload["messages"]))

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
                output_msg = data.get("choices", [{}])[0].get("message", {})
                span.update(
                    model=data.get("model", self.model),
                    output=output_msg,
                    usage_details={
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                    metadata={
                        "message_count": len(messages),
                        "tool_calls": [
                            tc.get("function", {}).get("name", "?")
                            for tc in (output_msg.get("tool_calls") or [])
                        ] or None,
                        "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
                    },
                )
            # SPEC-RAG-20d: аккумулируем tokens в RequestContext для root trace
            try:
                from services.agent.state import _request_ctx
                _ctx = _request_ctx.get(None)
                if _ctx:
                    _ctx.total_prompt_tokens += usage.get("prompt_tokens", 0)
                    _ctx.total_completion_tokens += usage.get("completion_tokens", 0)
            except Exception:  # broad: observability graceful degradation
                pass
            return data

    def tokenize(self, text: bytes, add_bos: bool = True) -> list[int]:
        """Tokenize text via /tokenize endpoint. Returns empty list on error."""
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
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            logger.debug("LlamaServerClient.tokenize failed (non-critical): %s", e)
            return []
