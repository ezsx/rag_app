"""
Langfuse observability — lazy imports, graceful degradation.

Все Langfuse imports сосредоточены ТОЛЬКО в этом модуле.
Runtime модули импортируют:
    from core.observability import observe_span, observe_llm_call, get_langfuse

Если langfuse не установлен или сервер недоступен — все функции
возвращают nullcontext/None, zero impact на runtime.

Гарантия: ни один вызов observability (init, update, exit) не бросает
исключение в production path. Все ошибки SDK логируются и глотаются.
"""

import logging
import sys
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_client = None
_enabled = None  # None = ещё не проверяли


def _try_init():
    """Lazy init: один раз пробуем импортировать и подключиться."""
    global _client, _enabled
    if _enabled is not None:
        return _enabled
    try:
        from langfuse import get_client  # noqa: F811
        _client = get_client()
        _enabled = True
        logger.info("Langfuse observability enabled")
    except ImportError:
        _enabled = False
        logger.info("langfuse package not installed — observability disabled")
    except Exception as e:
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
        except Exception as e:
            logger.debug("Langfuse span.update() failed (ignored): %s", e)

    def __getattr__(self, name):
        """Проксируем остальные атрибуты с защитой."""
        attr = getattr(self._inner, name)
        if callable(attr):
            def _safe_call(*args, **kw):
                try:
                    return attr(*args, **kw)
                except Exception as e:
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
    except Exception as e:
        logger.debug("Langfuse cm.__exit__() failed (ignored): %s", e)


@contextmanager
def observe_span(name, **kwargs):
    """Context manager для span. Graceful: yield None если Langfuse недоступен.

    Гарантии:
    - Ошибка при создании span → yield None, продолжаем без tracing
    - Ошибка при update/exit → логируем, не бросаем
    - Исключение изнутри тела (yield) → пробрасываем, span закрываем best-effort
    """
    client = get_langfuse()
    if client is None:
        yield None
        return
    try:
        cm = client.start_as_current_observation(
            as_type="span", name=name, **kwargs
        )
        raw_span = cm.__enter__()
    except Exception as e:
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

    Те же гарантии graceful degradation что и observe_span.
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
    except Exception as e:
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
