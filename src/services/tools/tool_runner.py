from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.observability import observe_span
from schemas.agent import AgentAction, ToolMeta, ToolRequest, ToolResponse

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    pass


def _run_with_timeout(
    func: Callable[..., dict[str, Any]], timeout_sec: float, *args, **kwargs
) -> tuple[bool, dict[str, Any], str | None]:
    """Выполняет функцию с таймаутом в отдельном потоке. Возвращает (ok, data, error)."""
    import concurrent.futures as _fut
    import contextvars

    start = time.perf_counter()
    error_msg: str | None = None
    ok = True
    data: dict[str, Any] = {}

    # copy_context() пробрасывает ContextVars (включая Langfuse trace context)
    # в дочерний поток — без этого child spans создают отдельные traces
    ctx = contextvars.copy_context()
    with _fut.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(ctx.run, func, *args, **kwargs)
        try:
            data = future.result(timeout=max(0.001, float(timeout_sec))) or {}
        except _fut.TimeoutError:
            ok = False
            error_msg = f"timeout>{int(timeout_sec*1000)}ms"
        except Exception as e:  # broad: tool execution safety
            ok = False
            error_msg = str(e)

    _took_ms = int((time.perf_counter() - start) * 1000)
    return ok, data, error_msg


@dataclass
class _ToolEntry:
    name: str
    func: Callable[..., dict[str, Any]]
    timeout_sec: float


class ToolRunner:
    """Реестр инструментов + единый запуск с таймаутом и JSON-трейсом.

    Журналирование: на каждый вызов пишет JSON-строку в stdout и системный логгер.
    Формат: {"req","step","tool","ok","took_ms","error"?}
    """

    def __init__(self, default_timeout_sec: float = 5.0):
        self._default_timeout = max(0.1, float(default_timeout_sec))
        self._registry: dict[str, _ToolEntry] = {}

    def register(
        self,
        name: str,
        func: Callable[..., dict[str, Any]],
        timeout_sec: float | None = None,
    ) -> None:
        self._registry[name] = _ToolEntry(
            name=name, func=func, timeout_sec=timeout_sec or self._default_timeout
        )

    def run(
        self, request_id: str, step: int, req: ToolRequest,
        deadline: float | None = None,
    ) -> AgentAction:
        """Запускает tool с timeout. Если deadline задан — min(tool_timeout, remaining)."""
        entry = self._registry.get(req.tool)
        if entry is None:
            meta = ToolMeta(took_ms=0, error=f"tool_not_found:{req.tool}")
            resp = ToolResponse(ok=False, data={}, meta=meta)
            action = AgentAction(step=step, tool=req.tool, input=req.input, output=resp)
            self._log_trace(request_id, action)
            return action

        # FIX-08: remaining budget из request deadline
        effective_timeout = entry.timeout_sec
        if deadline is not None:
            remaining = max(0.5, deadline - time.monotonic())
            effective_timeout = min(effective_timeout, remaining)

        _system_tools = {"verify", "fetch_docs"}
        _tool_prefix = "tool:" if req.tool not in _system_tools else "tool[system]:"
        with observe_span(
            f"{_tool_prefix}{req.tool}", as_type="tool",
            input=req.input, metadata={"step": step},
        ) as span:
            started = time.perf_counter()
            ok, data, error = _run_with_timeout(entry.func, effective_timeout, **req.input)
            took_ms = int((time.perf_counter() - started) * 1000)

            # P0 fix: если tool вернул {"error": ...}, это НЕ успех
            if ok and isinstance(data, dict) and data.get("error"):
                ok = False
                error = error or str(data["error"])

            if span:
                # Включаем summary данных для observability (не полные тексты)
                output_summary: dict = {
                    "ok": ok and error is None,
                    "took_ms": took_ms,
                    "error": error,
                }
                if ok and isinstance(data, dict):
                    # Compact summary: counts и ключевые метрики
                    if "hits" in data:
                        output_summary["hits_count"] = len(data.get("hits", []))
                    if "citation_coverage" in data:
                        output_summary["coverage"] = data["citation_coverage"]
                    if "citations" in data:
                        output_summary["citations_count"] = len(data.get("citations", []))
                    if "indices" in data:
                        output_summary["reranked_count"] = len(data.get("indices", []))
                    if "answer" in data:
                        output_summary["answer_len"] = len(str(data.get("answer", "")))
                    if "prompt" in data:
                        output_summary["prompt_len"] = len(str(data.get("prompt", "")))
                span.update(output=output_summary)
                # SPEC-RAG-20d: mark failed tools для Langfuse error filtering
                if not ok or error:
                    span.update(level="ERROR", status_message=str(error)[:200] if error else "tool_failed")

        meta = ToolMeta(took_ms=took_ms, error=error)
        resp = ToolResponse(ok=ok and error is None, data=data if ok else {}, meta=meta)
        action = AgentAction(step=step, tool=req.tool, input=req.input, output=resp)
        self._log_trace(request_id, action)
        return action

    def _log_trace(self, request_id: str, action: AgentAction) -> None:
        payload = {
            "req": request_id,
            "step": action.step,
            "tool": action.tool,
            "ok": action.output.ok,
            "took_ms": action.output.meta.took_ms,
        }
        if action.output.meta.error:
            payload["error"] = action.output.meta.error
        line = json.dumps(payload, ensure_ascii=False)
        # stdout
        print(line, flush=True)
        # logger
        logger.info(line)
