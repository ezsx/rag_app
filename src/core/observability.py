"""
Langfuse observability — lazy imports, graceful degradation.

Все Langfuse imports сосредоточены ТОЛЬКО в этом модуле.
Runtime модули импортируют:
    from core.observability import observe_trace, observe_span, observe_llm_call, get_langfuse

Если langfuse не установлен или сервер недоступен — все функции
возвращают nullcontext/None, zero impact на runtime.

Гарантия: ни один вызов observability (init, update, exit) не бросает
исключение в production path. Все ошибки SDK логируются и глотаются.
"""

import json
import logging
import sys
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

_client = None
_enabled = None  # None = ещё не проверяли


def _try_init():
    """Lazy init: один раз пробуем импортировать и подключиться."""
    global _client, _enabled
    if _enabled is not None:
        return _enabled
    try:
        from langfuse import get_client
        _client = get_client()
        _enabled = True
        logger.info("Langfuse observability enabled")
    except ImportError:
        _enabled = False
        logger.info("langfuse package not installed — observability disabled")
    except Exception as e:  # broad: observability graceful degradation
        _enabled = False
        logger.warning("Langfuse init failed — observability disabled: %s", e)
    return _enabled


def get_langfuse():
    """Возвращает Langfuse client или None."""
    if not _try_init():
        return None
    return _client


class _SafeSpan:
    """Обёртка над Langfuse span/generation — все методы best-effort.

    span.update() и другие вызовы SDK никогда не бросают наружу.
    Callsites могут безопасно вызывать span.update(...) без try/except.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner):
        self._inner = inner

    def update(self, **kwargs):
        try:
            self._inner.update(**kwargs)
        except Exception as e:  # broad: observability graceful degradation
            logger.debug("Langfuse span.update() failed (ignored): %s", e)

    def set_attribute(self, key, value):
        """Устанавливает OTel атрибут на underlying span."""
        try:
            self._inner.set_attribute(key, value)
        except Exception as e:  # broad: observability graceful degradation
            logger.debug("Langfuse span.set_attribute(%s) failed (ignored): %s", key, e)

    def __getattr__(self, name):
        """Проксируем остальные атрибуты с защитой."""
        attr = getattr(self._inner, name)
        if callable(attr):
            def _safe_call(*args, **kw):
                try:
                    return attr(*args, **kw)
                except Exception as e:  # broad: observability graceful degradation
                    logger.debug("Langfuse span.%s() failed (ignored): %s", name, e)
                    return None
            return _safe_call
        return attr


def _safe_exit(cm, exc_info=None):
    """Вызывает cm.__exit__ без пробрасывания ошибок SDK."""
    try:
        if exc_info:
            cm.__exit__(*exc_info)
        else:
            cm.__exit__(None, None, None)
    except Exception as e:  # broad: observability graceful degradation
        logger.debug("Langfuse cm.__exit__() failed (ignored): %s", e)


def _set_trace_attributes(span, *, name=None, session_id=None, tags=None,
                          input_data=None, output_data=None, metadata=None):
    """Устанавливает trace-level OTel атрибуты на span.

    Langfuse SDK v4 читает эти атрибуты для формирования trace в UI:
    - langfuse.trace.name → имя trace
    - session.id → группировка по сессии
    - langfuse.trace.tags → теги для фильтрации
    - langfuse.trace.input/output → вход/выход trace
    - langfuse.trace.metadata → произвольные метаданные
    """
    try:
        from langfuse import LangfuseOtelSpanAttributes as Attr
        if name:
            span.set_attribute(Attr.TRACE_NAME, name)
        if session_id:
            span.set_attribute(Attr.TRACE_SESSION_ID, session_id)
        if tags:
            span.set_attribute(Attr.TRACE_TAGS, json.dumps(tags))
        if input_data:
            # SPEC-RAG-20d: защита от double encoding — если уже string, парсим обратно
            _inp = input_data
            if isinstance(_inp, str):
                try:
                    _inp = json.loads(_inp)
                except (json.JSONDecodeError, TypeError):
                    pass
            span.set_attribute(Attr.TRACE_INPUT, json.dumps(_inp, ensure_ascii=False, default=str))
        if output_data:
            _out = output_data
            if isinstance(_out, str):
                try:
                    _out = json.loads(_out)
                except (json.JSONDecodeError, TypeError):
                    pass
            span.set_attribute(Attr.TRACE_OUTPUT, json.dumps(_out, ensure_ascii=False, default=str))
        if metadata:
            span.set_attribute(Attr.TRACE_METADATA, json.dumps(metadata, ensure_ascii=False, default=str))
    except Exception as e:  # broad: observability graceful degradation
        logger.debug("Failed to set trace attributes: %s", e)


@contextmanager
def observe_trace(
    name: str,
    *,
    session_id: str | None = None,
    tags: list[str] | None = None,
    input_data: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Root trace context manager. Все child observe_span/observe_llm_call вложатся внутрь.

    Создаёт OTel span с as_root=True — Langfuse группирует все дочерние
    spans под одним trace в UI.
    """
    client = get_langfuse()
    if client is None:
        yield None
        return
    try:
        cm = client.start_as_current_observation(
            as_type="span", name=name,
        )
        raw_span = cm.__enter__()
        # Помечаем как root trace
        try:
            from langfuse import LangfuseOtelSpanAttributes as Attr
            raw_span.set_attribute(Attr.AS_ROOT, True)
        except Exception:  # broad: observability graceful degradation
            pass
        # Trace-level атрибуты
        _set_trace_attributes(
            raw_span, name=name, session_id=session_id,
            tags=tags, input_data=input_data, metadata=metadata,
        )
    except Exception as e:  # broad: observability graceful degradation
        logger.warning("Langfuse observe_trace(%s) init failed: %s", name, e)
        yield None
        return
    try:
        yield _SafeSpan(raw_span)
    except BaseException:
        _safe_exit(cm, sys.exc_info())
        raise
    else:
        _safe_exit(cm)


@contextmanager
def observe_span(name, as_type="span", **kwargs):
    """Context manager для child span/tool. Вкладывается в текущий active trace/span."""
    client = get_langfuse()
    if client is None:
        yield None
        return
    try:
        cm = client.start_as_current_observation(
            as_type=as_type, name=name, **kwargs
        )
        raw_span = cm.__enter__()
    except Exception as e:  # broad: observability graceful degradation
        logger.warning("Langfuse observe_span(%s) init failed: %s", name, e)
        yield None
        return
    try:
        yield _SafeSpan(raw_span)
    except BaseException:
        _safe_exit(cm, sys.exc_info())
        raise
    else:
        _safe_exit(cm)


@contextmanager
def observe_llm_call(name="llm_call", model="", **kwargs):
    """Context manager для LLM generation span.

    Создаёт Langfuse generation span с типом 'generation' —
    автоматически трейсит input/output/usage/model.
    """
    client = get_langfuse()
    if client is None:
        yield None
        return
    try:
        cm = client.start_as_current_observation(
            as_type="generation", name=name, model=model, **kwargs
        )
        raw_gen = cm.__enter__()
    except Exception as e:  # broad: observability graceful degradation
        logger.warning("Langfuse observe_llm_call(%s) init failed: %s", name, e)
        yield None
        return
    try:
        yield _SafeSpan(raw_gen)
    except BaseException:
        _safe_exit(cm, sys.exc_info())
        raise
    else:
        _safe_exit(cm)
