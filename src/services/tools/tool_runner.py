from __future__ import annotations

import json
import logging
import signal
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from schemas.agent import ToolRequest, ToolResponse, ToolMeta, AgentAction

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    pass


def _run_with_timeout(
    func: Callable[..., Dict[str, Any]], timeout_sec: float, *args, **kwargs
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """Выполняет функцию с таймаутом в отдельном потоке. Возвращает (ok, data, error)."""
    import concurrent.futures as _fut

    start = time.perf_counter()
    error_msg: Optional[str] = None
    ok = True
    data: Dict[str, Any] = {}

    with _fut.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(func, *args, **kwargs)
        try:
            data = future.result(timeout=max(0.001, float(timeout_sec))) or {}
        except _fut.TimeoutError:
            ok = False
            error_msg = f"timeout>{int(timeout_sec*1000)}ms"
        except Exception as e:  # noqa: B902
            ok = False
            error_msg = str(e)

    took_ms = int((time.perf_counter() - start) * 1000)
    return ok, data, error_msg


@dataclass
class _ToolEntry:
    name: str
    func: Callable[..., Dict[str, Any]]
    timeout_sec: float


class ToolRunner:
    """Реестр инструментов + единый запуск с таймаутом и JSON-трейсом.

    Журналирование: на каждый вызов пишет JSON-строку в stdout и системный логгер.
    Формат: {"req","step","tool","ok","took_ms","error"?}
    """

    def __init__(self, default_timeout_sec: float = 5.0):
        self._default_timeout = max(0.1, float(default_timeout_sec))
        self._registry: Dict[str, _ToolEntry] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Dict[str, Any]],
        timeout_sec: Optional[float] = None,
    ) -> None:
        self._registry[name] = _ToolEntry(
            name=name, func=func, timeout_sec=timeout_sec or self._default_timeout
        )

    def run(self, request_id: str, step: int, req: ToolRequest) -> AgentAction:
        entry = self._registry.get(req.tool)
        if entry is None:
            meta = ToolMeta(took_ms=0, error=f"tool_not_found:{req.tool}")
            resp = ToolResponse(ok=False, data={}, meta=meta)
            action = AgentAction(step=step, tool=req.tool, input=req.input, output=resp)
            self._log_trace(request_id, action)
            return action

        started = time.perf_counter()
        ok, data, error = _run_with_timeout(entry.func, entry.timeout_sec, **req.input)
        took_ms = int((time.perf_counter() - started) * 1000)
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
