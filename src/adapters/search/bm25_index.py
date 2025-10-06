from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict

import tantivy  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class BM25Doc:
    doc_id: str
    text: str
    channel_id: int
    channel_username: Optional[str]
    date_days: int
    date_iso: str
    views: Optional[int]
    reply_to: Optional[int]
    msg_id: int


@dataclass
class BM25Query:
    must_terms: List[str]
    should_terms: List[str]
    filters: Dict[str, object]


@dataclass
class BM25Hit:
    doc_id: str
    text: str
    metadata: Dict
    bm25_score: float


@dataclass
class IndexHandle:
    schema: tantivy.Schema
    index: tantivy.Index
    writer: Optional[tantivy.IndexWriter]
    reader: Optional[tantivy.IndexReader]
    searcher: Optional[tantivy.IndexSearcher]
    paths: Dict[str, str]
    fields: Optional[Dict[str, object]] = None  # Field refs (compat with older tantivy)


class BM25IndexManager:
    def __init__(self, index_root: str, reload_min_interval_sec: int = 5):
        self.index_root = index_root
        os.makedirs(self.index_root, exist_ok=True)
        self._handles: Dict[str, IndexHandle] = {}
        self._last_reload_ts: Dict[str, float] = {}
        self._reload_min_interval = reload_min_interval_sec

    def _build_reader_and_searcher(
        self, index: tantivy.Index
    ) -> (Optional[tantivy.IndexReader], Optional[tantivy.IndexSearcher]):
        """Создает reader/searcher с учётом различий версий python-tantivy."""

        # Современные версии предоставляют builder API
        if hasattr(index, "reader"):
            try:
                reader = index.reader()
                searcher = reader.searcher()
                return reader, searcher
            except Exception as exc:
                logger.warning(f"Unable to construct reader via index.reader(): {exc}")

        if hasattr(index, "reader_builder"):
            try:
                builder = index.reader_builder()
                # Попробуем закрепить ручную политику перезагрузки если доступна
                if hasattr(builder, "reload_policy"):
                    try:
                        builder.reload_policy("manual")
                    except Exception as exc:
                        logger.debug(
                            "reader_builder.reload_policy manual failed: %s", exc
                        )
                reader = builder.build()
                searcher = reader.searcher()
                return reader, searcher
            except Exception as exc:
                logger.warning(
                    f"Unable to construct reader via reader_builder(): {exc}"
                )

        # Фолбэк: напрямую создаём searcher без reader
        try:
            searcher = index.searcher()
            return None, searcher
        except Exception as exc:
            logger.error(f"Unable to construct searcher: {exc}")
            return None, None

    def _resolve_fields(self, schema: tantivy.Schema) -> Dict[str, object]:
        """Совместимо получает ссылки на поля по именам в разных версиях python-tantivy."""

        def gf(name: str):
            # Попробуем несколько вариантов API
            if hasattr(schema, "get_field"):
                return getattr(schema, "get_field")(name)
            if hasattr(schema, "get_field_by_name"):
                return getattr(schema, "get_field_by_name")(name)
            # Некоторые версии предоставляют индексацию по имени
            if hasattr(schema, name):
                return getattr(schema, name)
            raise AttributeError(
                "Schema does not provide a field accessor compatible with this code"
            )

        return {
            "doc_id": gf("doc_id"),
            "text": gf("text"),
            "channel_id": gf("channel_id"),
            "channel_username": gf("channel_username"),
            "date_days": gf("date_days"),
            "date_iso": gf("date_iso"),
            "views": gf("views"),
            "reply_to": gf("reply_to"),
            "msg_id": gf("msg_id"),
        }

    def _schema(self) -> (tantivy.Schema, Dict[str, object]):
        builder = tantivy.SchemaBuilder()
        f_doc_id = builder.add_text_field("doc_id", stored=True)
        f_text = builder.add_text_field(
            "text", stored=True
        )  # анализатор по умолчанию (ru недоступен везде)

        # Совместимость с разными версиями python-tantivy
        def _add_u64(name: str, stored: bool = True, fast: bool = True):
            if hasattr(builder, "add_u64_field"):
                return builder.add_u64_field(name, stored=stored, fast=fast)
            if hasattr(builder, "add_unsigned_field"):
                return builder.add_unsigned_field(name, stored=stored, fast=fast)
            return builder.add_u64_field(name, stored=stored, fast=fast)

        def _add_i64(name: str, stored: bool = True, fast: bool = True):
            if hasattr(builder, "add_i64_field"):
                return builder.add_i64_field(name, stored=stored, fast=fast)
            if hasattr(builder, "add_integer_field"):
                return builder.add_integer_field(name, stored=stored, fast=fast)
            return builder.add_i64_field(name, stored=stored, fast=fast)

        # Telegram channel_id может быть отрицательным (например, -100...),
        # поэтому используем знаковое целое для совместимости
        f_channel_id = _add_i64("channel_id", stored=True, fast=True)
        f_channel_username = builder.add_text_field("channel_username", stored=True)
        f_date_days = _add_i64("date_days", stored=True, fast=True)
        f_date_iso = builder.add_text_field("date_iso", stored=True)
        f_views = _add_u64("views", stored=True, fast=True)
        f_reply_to = _add_u64("reply_to", stored=True, fast=True)
        f_msg_id = _add_u64("msg_id", stored=True)
        schema = builder.build()
        fields = {
            "doc_id": f_doc_id,
            "text": f_text,
            "channel_id": f_channel_id,
            "channel_username": f_channel_username,
            "date_days": f_date_days,
            "date_iso": f_date_iso,
            "views": f_views,
            "reply_to": f_reply_to,
            "msg_id": f_msg_id,
        }
        return schema, fields

    def get_or_create(self, collection: str, for_write: bool = False) -> IndexHandle:
        if collection in self._handles:
            handle = self._handles[collection]
            # reader/searcher лениво
            return handle

        path = os.path.join(self.index_root, collection)
        os.makedirs(path, exist_ok=True)

        schema, fields = self._schema()
        if os.listdir(path):
            # Открытие индекса: разные версии python-tantivy имеют разные сигнатуры
            try:
                index = tantivy.Index.open(schema, path)  # старый API
            except TypeError:
                index = tantivy.Index.open(path)  # новый API
            logger.info(f"BM25 index opened: {path}")
        else:
            index = tantivy.Index(schema, path)
            logger.info(f"BM25 index created: {path}")

        writer = index.writer() if for_write else None
        reader = None
        searcher = None
        if not for_write:
            reader, searcher = self._build_reader_and_searcher(index)

        handle = IndexHandle(
            schema=schema,
            index=index,
            writer=writer,
            reader=reader,
            searcher=searcher,
            paths={"root": path},
            # Поля больше не резолвим: работаем по строковым именам полей для совместимости
            fields=None,
        )
        self._handles[collection] = handle
        self._last_reload_ts[collection] = 0.0
        return handle

    def reload(self, collection: str) -> None:
        now = time.time()
        if now - self._last_reload_ts.get(collection, 0.0) < self._reload_min_interval:
            return
        handle = self.get_or_create(collection)
        if handle.reader is not None:
            try:
                handle.reader.reload()
                handle.searcher = handle.reader.searcher()
                self._last_reload_ts[collection] = now
                logger.info(f"BM25 reload searcher (reader-based): {collection}")
                return
            except Exception as exc:
                logger.warning(f"Reader reload failed, rebuilding searcher: {exc}")
                handle.reader = None

        reader, searcher = self._build_reader_and_searcher(handle.index)
        handle.reader = reader
        handle.searcher = searcher
        self._last_reload_ts[collection] = now
        logger.info(f"BM25 reload searcher: {collection}")

    def add_documents(
        self, collection: str, docs: List[BM25Doc], commit_every: int = 1000
    ) -> None:
        handle = self.get_or_create(collection, for_write=True)
        writer = handle.writer
        assert writer is not None

        added = 0

        # Совместимость c разными версиями python-tantivy: add_u64 / add_unsigned, add_i64 / add_integer
        def _add_u64(doc: tantivy.Document, field, value: int) -> None:
            # Во многих версиях можно передавать строковое имя поля вместо Field
            if isinstance(field, str):
                name = field
            else:
                # На всякий случай, если нам передали Field
                try:
                    name = field
                except Exception:
                    name = field
            if hasattr(doc, "add_u64"):
                doc.add_u64(name, int(value))
            elif hasattr(doc, "add_unsigned"):
                doc.add_unsigned(name, int(value))
            else:
                doc.add_u64(name, int(value))

        def _add_i64(doc: tantivy.Document, field, value: int) -> None:
            if isinstance(field, str):
                name = field
            else:
                name = field
            if hasattr(doc, "add_i64"):
                doc.add_i64(name, int(value))
            elif hasattr(doc, "add_integer"):
                doc.add_integer(name, int(value))
            else:
                doc.add_i64(name, int(value))

        for d in docs:
            doc = tantivy.Document()
            # Пишем по именам полей для лучшей совместимости
            doc.add_text("doc_id", d.doc_id)
            doc.add_text("text", d.text)
            _add_i64(doc, "channel_id", int(d.channel_id))
            doc.add_text("channel_username", d.channel_username or "")
            _add_i64(doc, "date_days", int(d.date_days))
            doc.add_text("date_iso", d.date_iso)
            if d.views is not None:
                _add_u64(doc, "views", int(d.views))
            if d.reply_to is not None:
                _add_u64(doc, "reply_to", int(d.reply_to))
            _add_u64(doc, "msg_id", int(d.msg_id))
            writer.add_document(doc)
            added += 1
            if added % commit_every == 0:
                writer.commit()
                logger.info(f"BM25 committed {added} docs to {collection}")

        writer.commit()
        logger.info(f"BM25 committed total {added} docs to {collection}")

    def search(self, collection: str, query: BM25Query, top_k: int) -> List[BM25Hit]:
        self.reload(collection)
        handle = self.get_or_create(collection)
        assert handle.searcher is not None

        # Сбор строкового запроса и парсинг по именам полей
        fs = query.filters or {}
        parts: List[str] = []
        for term in query.must_terms or []:
            if term:
                parts.append(f"+({term})")
        for term in query.should_terms or []:
            if term:
                parts.append(f"{term}")
        if cu := fs.get("channel_usernames"):
            for u in cu:
                parts.append(f"channel_username:{u}")
        if cids := fs.get("channel_ids"):
            for cid in cids:
                parts.append(f"channel_id:{int(cid)}")
        if (rt := fs.get("reply_to")) is not None:
            parts.append(f"reply_to:{int(rt)}")
        if (mv := fs.get("min_views")) is not None:
            parts.append(f"views:[{int(mv)} TO *]")
        df = fs.get("date_from_days")
        dt = fs.get("date_to_days")
        if df is not None or dt is not None:
            lo = "*" if df is None else str(int(df))
            hi = "*" if dt is None else str(int(dt))
            parts.append(f"date_days:[{lo} TO {hi}]")

        if not parts:
            return []
        query_str = " AND ".join(parts)
        parsed = handle.index.parse_query(query_str, ["text"])  # type: ignore[attr-defined]
        top = handle.searcher.search(parsed, top_k)
        hits: List[BM25Hit] = []
        for score, addr in top.hits:
            doc = handle.searcher.doc(addr)
            data = doc.to_dict()
            hits.append(
                BM25Hit(
                    doc_id=data.get("doc_id", [""])[0],
                    text=data.get("text", [""])[0],
                    metadata={
                        "channel_id": int(data.get("channel_id", [0])[0]),
                        "channel_username": data.get("channel_username", [""])[0],
                        "date_days": int(data.get("date_days", [0])[0]),
                        "date_iso": data.get("date_iso", [""])[0],
                        "views": (
                            int(data.get("views", [0])[0])
                            if data.get("views")
                            else None
                        ),
                        "reply_to": (
                            int(data.get("reply_to", [0])[0])
                            if data.get("reply_to")
                            else None
                        ),
                        "msg_id": (
                            int(data.get("msg_id", [0])[0]) if data.get("msg_id") else 0
                        ),
                    },
                    bm25_score=float(score),
                )
            )
        return hits
