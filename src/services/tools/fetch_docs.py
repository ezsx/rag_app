from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from adapters.chroma import Retriever

logger = logging.getLogger(__name__)


def fetch_docs(
    retriever: Retriever, ids: List[str], window: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Батч-выгрузка документов по ids из активной коллекции.

    MVP: Chroma коллекция не гарантирует прямого get по произвольным id, поэтому
    используем имеющиеся тексты в hits. Если текст отсутствует, возвращаем заглушку
    с пустым текстом и метаданными по умолчанию.
    """
    if not ids:
        return {"docs": []}
    try:
        docs = retriever.get_by_ids(ids)
        return {"docs": docs}
    except Exception:
        # Фолбэк: пустые тексты
        return {"docs": [{"id": _id, "text": "", "metadata": {}} for _id in ids]}
