from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from adapters.chroma import Retriever

logger = logging.getLogger(__name__)


def fetch_docs(
    retriever: Retriever,
    ids: Optional[List[str]] = None,
    window: Optional[List[int]] = None,
    doc_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Батч-выгрузка документов по ids из активной коллекции.

    MVP: Chroma коллекция не гарантирует прямого get по произвольным id, поэтому
    используем имеющиеся тексты в hits. Если текст отсутствует, возвращаем заглушку
    с пустым текстом и метаданными по умолчанию.
    """
    final_ids: List[str] = ids or doc_ids or []
    if not final_ids:
        logger.debug("fetch_docs called without ids")
        return {"docs": []}
    try:
        docs = retriever.get_by_ids(final_ids)
        logger.debug("fetch_docs succeeded | ids=%s | count=%s", final_ids, len(docs))
        return {"docs": docs}
    except Exception as exc:
        logger.error("fetch_docs failed for ids=%s: %s", final_ids, exc)
        return {"docs": [{"id": _id, "text": "", "metadata": {}} for _id in final_ids]}
