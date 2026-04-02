"""
Telegram → Qdrant ingestor (Phase 1)
------------------------------------
CLI-скрипт для загрузки сообщений Telegram-каналов в Qdrant.

Dense: TEI HTTP → Qwen3-Embedding-0.6B @ host.docker.internal:8082
Sparse: fastembed SparseTextEmbedding (Qdrant/bm25, language=russian, CPU)
Store: Qdrant (Docker CPU)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from dateutil import parser as date_parser
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon import errors as tg_errors

# payload_enrichment — в том же каталоге scripts/
sys.path.insert(0, os.path.dirname(__file__))
from payload_enrichment import build_enriched_payload
from telethon.errors import FloodWaitError, SessionPasswordNeededError
from telethon.tl.types import Message
from tqdm.asyncio import tqdm_asyncio

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from adapters.qdrant.store import PointDocument, QdrantStore
from adapters.tei.embedding_client import TEIEmbeddingClient
from core.settings import get_settings

try:
    from fastembed import SparseTextEmbedding
except ImportError as exc:
    raise ImportError(
        "fastembed не установлен. Добавьте в requirements: fastembed>=0.3.0"
    ) from exc


EMBED_RETRY_ATTEMPTS = 5
EMBED_RETRY_BASE_DELAY_S = 2.0
EMBED_RETRY_MAX_DELAY_S = 20.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest Telegram channel(s) into Qdrant")
    p.add_argument(
        "--channel", required=False, default=None, help="@username or chat id"
    )
    p.add_argument(
        "--channels",
        required=False,
        default=None,
        help="Comma-separated list of channels, e.g. @a,@b,@c",
    )
    p.add_argument("--since", required=True, help="ISO date (inclusive)")
    p.add_argument("--until", required=True, help="ISO date (exclusive)")
    p.add_argument(
        "--collection",
        required=False,
        default=None,
        help="Qdrant collection name (default: QDRANT_COLLECTION env or settings)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Сколько сообщений обрабатывать за один TEI embed запрос (default: 64)",
    )
    p.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="ingest N newest messages then stop (debug)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Split long messages into chunks of N chars (0 = no split)",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _compute_embed_retry_delay(attempt: int) -> float:
    """Возвращает задержку перед повторной попыткой embed с простым exponential backoff."""
    return min(
        EMBED_RETRY_BASE_DELAY_S * (2 ** max(0, attempt - 1)),
        EMBED_RETRY_MAX_DELAY_S,
    )


async def _embed_documents_with_retry(
    embedding_client: TEIEmbeddingClient,
    texts: list[str],
    *,
    channel_hint: str | None,
    batch_no: int,
) -> list[list[float]]:
    """Запрашивает dense embeddings с retry для transient ошибок сети/таймаутов.

    Если TEI временно недоступен, делаем несколько повторных попыток с backoff.
    После исчерпания попыток исключение пробрасывается наверх, чтобы ingest
    завершился ошибкой, а не оставил тихую дыру в коллекции.
    """
    last_exc: Exception | None = None

    for attempt in range(1, EMBED_RETRY_ATTEMPTS + 1):
        try:
            return await embedding_client.embed_documents(texts)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_exc = exc
            if attempt >= EMBED_RETRY_ATTEMPTS:
                logger.error(
                    "TEI embed не восстановился после %d попыток: channel=%s batch=%d texts=%d",
                    EMBED_RETRY_ATTEMPTS,
                    channel_hint,
                    batch_no,
                    len(texts),
                )
                raise

            delay_s = _compute_embed_retry_delay(attempt)
            logger.warning(
                "TEI embed временно недоступен, retry через %.1fs: channel=%s batch=%d attempt=%d/%d error=%s",
                delay_s,
                channel_hint,
                batch_no,
                attempt,
                EMBED_RETRY_ATTEMPTS,
                exc.__class__.__name__,
            )
            await asyncio.sleep(delay_s)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("TEI embedding retry loop завершился без результата")


load_dotenv()

logger = logging.getLogger("ingest_telegram")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))


CHUNK_CHAR_THRESHOLD = 1500
CHUNK_TARGET_SIZE = 1200


async def create_telegram_client() -> TelegramClient:
    api_id_str = os.getenv("TG_API_ID")
    if not api_id_str:
        logger.error("TG_API_ID not set; add it to .env or environment")
        sys.exit(1)
    api_id = int(api_id_str)
    api_hash = os.getenv("TG_API_HASH")
    # Persist session file to avoid re-auth every run. Using TG_SESSION allows
    # passing an absolute path (e.g. a mounted Docker volume). If the path
    # points to a directory, we append the default filename so Telethon gets a
    # valid file path.
    session_path_env = os.getenv("TG_SESSION", "telegram.session")
    # Expand tilde and make absolute for safety
    session_path = os.path.expanduser(session_path_env)
    if os.path.isdir(session_path):
        session_path = os.path.join(session_path, "telegram.session")

    # Ensure parent dir exists so Telethon can write the session file even in a
    # freshly mounted volume.
    parent_dir = os.path.dirname(session_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    phone = os.getenv("TG_PHONE")
    bot_token = os.getenv("BOT_TOKEN")

    if not (api_id and api_hash):
        logger.error("TG_API_ID and TG_API_HASH must be set in environment")
        sys.exit(1)

    client = TelegramClient(session_path, api_id, api_hash, device_model="RAG-Scraper")
    await client.connect()

    if not await client.is_user_authorized():
        if bot_token:
            await client.start(bot_token=bot_token)
        elif phone:
            await client.send_code_request(phone)
            code = input("Enter the code you just received: ")
            try:
                await client.sign_in(phone, code)
            except SessionPasswordNeededError:
                pwd = input("Two‑factor enabled. Enter your password: ")
                await client.sign_in(password=pwd)
        else:
            logger.error("Provide BOT_TOKEN or TG_PHONE for authorization.")
            sys.exit(1)
    return client


def split_into_batches(seq: list[Any], batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


async def gather_messages(
    client: TelegramClient,
    channel: str,
    start: datetime,
    end: datetime,
    limit: int | None = None,
) -> list[Message]:
    """Fetch messages with *start* ≤ date < *end*.

    Telegram's API returns history in reverse-chronological order by default
    (newest → oldest). We leverage that to avoid walking the whole history of
    busy channels: request messages older than *end* and stop once we cross
    *start*."""

    collected: list[Message] = []

    # Request messages older-than `end` (exclusive). We keep the default
    # order (newest→oldest) so once we reach dates older than `start` we can
    # break early.
    async for msg in client.iter_messages(channel, offset_date=end):
        # Message dates from Telethon are timezone-aware UTC; normalise for
        # reliable comparisons.
        msg_dt = _to_utc_naive(msg.date)

        if msg_dt < start:
            break  # we went past the desired range
        if msg_dt >= end:
            continue  # should not normally happen, but be safe

        if not (msg.message and msg.message.strip()):
            continue  # skip non-text and service messages

        collected.append(msg)

        if limit and len(collected) >= limit:
            break

    # We collected newest→oldest; reverse to chronological order for nicer
    # batching/ingestion.
    collected.reverse()
    return collected


def _split_text(text: str, chunk_size: int) -> list[str]:
    if chunk_size and chunk_size > 0 and len(text) > chunk_size:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    return [text]


def _recursive_split(text: str, target: int, separators: list[str]) -> list[str]:
    """Рекурсивно делит длинный текст по иерархии сепараторов."""
    if len(text) <= target:
        return [text]

    if not separators:
        return [text[i : i + target] for i in range(0, len(text), target)]

    sep = separators[0]
    rest_seps = separators[1:]
    parts = text.split(sep)
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= target:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(part) > target:
            chunks.extend(_recursive_split(part, target, rest_seps))
            current = ""
        else:
            current = part

    if current:
        chunks.append(current)

    return chunks


def _smart_chunk(
    text: str,
    threshold: int = CHUNK_CHAR_THRESHOLD,
    target: int = CHUNK_TARGET_SIZE,
) -> list[str]:
    """Two-tier chunking: короткие посты целиком, длинные — recursive split."""
    if len(text) <= threshold:
        return [text]

    chunks = _recursive_split(text, target, ["\n\n", "\n", ". ", " "])
    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]


async def _gather_with_retries(
    client: TelegramClient,
    channel: str,
    start: datetime,
    end: datetime,
    limit: int | None = None,
    max_retries: int = 5,
):
    attempt = 0
    while True:
        try:
            return await gather_messages(client, channel, start, end, limit)
        except FloodWaitError as e:
            wait_s = int(getattr(e, "seconds", 0) or 0) or 60
            logger.warning(
                "FloodWait channel=%s: ждём %ss (attempt %d/%d)",
                channel,
                wait_s,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(wait_s + min(30, attempt * 5))
        except (tg_errors.RPCError, OSError) as e:
            if attempt >= max_retries - 1:
                logger.error("Исчерпаны ретраи сбора сообщений для %s: %s", channel, e)
                raise
            backoff = min(60, (2**attempt) * 2)
            logger.warning(
                "Сетевая ошибка для %s: %s. Повтор через %ss (attempt %d/%d)",
                channel,
                e,
                backoff,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(backoff)
        attempt += 1


def _build_point_docs_flat(
    source_messages: list[Message],
    texts: list[str],
    dense_vectors: list[list[float]],
    sparse_results: list[Any],
    channel_name: str,
    chunk_size: int,
    colbert_vectors: list[list[list[float]]] | None = None,
) -> list[PointDocument]:
    """
    Flat-вариант построения PointDocument, где `source_messages[i] ↔ texts[i]`.

    `channel_name` должен быть уже нормализован для display/url payload.
    Для стабильного upsert `point_id` строится только от `channel_id`,
    чтобы rename канала или разный CLI hint не меняли UUID точки.
    """
    docs: list[PointDocument] = []
    msg_chunk_counter: dict[int, int] = {}
    msg_chunk_totals: dict[int, int] = {}

    for message in source_messages:
        msg_chunk_totals[int(message.id)] = msg_chunk_totals.get(int(message.id), 0) + 1

    for i, (message, text) in enumerate(zip(source_messages, texts)):
        chunk_idx = msg_chunk_counter.get(message.id, 0)
        msg_chunk_counter[message.id] = chunk_idx + 1

        stable_channel_id = int(message.chat_id)
        if chunk_size > 0 or msg_chunk_totals.get(int(message.id), 0) > 1:
            point_id = f"{stable_channel_id}:{int(message.id)}:{chunk_idx}"
        else:
            point_id = f"{stable_channel_id}:{int(message.id)}"

        author: str | None = None
        sender = getattr(message, "sender", None)
        if sender is not None:
            first = getattr(sender, "first_name", None) or ""
            last = getattr(sender, "last_name", None) or ""
            author = (first + " " + last).strip() or None

        date_iso = _to_utc_naive(message.date).isoformat()
        # SPEC-RAG-12: enriched payload с entities, urls, temporal fields
        payload = build_enriched_payload(
            text=text,
            message=message,
            channel_name=channel_name,
            date_iso=date_iso,
            point_id=point_id,
            author=author,
        )

        sparse = sparse_results[i]
        cvec = colbert_vectors[i] if colbert_vectors and i < len(colbert_vectors) else None
        docs.append(
            PointDocument(
                point_id=point_id,
                dense_vector=dense_vectors[i],
                sparse_indices=sparse.indices.astype("uint32").tolist(),
                sparse_values=sparse.values.astype("float32").tolist(),
                colbert_vectors=cvec,
                payload=payload,
            )
        )

    return docs


async def ingest_batches(
    messages: list[Message],
    batch_size: int,
    embedding_client: TEIEmbeddingClient,
    sparse_encoder: SparseTextEmbedding,
    qdrant_store: QdrantStore,
    channel_hint: str | None = None,
    chunk_size: int = 0,
    chunk_char_threshold: int = CHUNK_CHAR_THRESHOLD,
    chunk_target_size: int = CHUNK_TARGET_SIZE,
    progress_cb: Any | None = None,
    log_every: int = 200,
) -> dict[str, int]:
    """
    Основной цикл инжеста: Telegram messages → TEI dense → fastembed sparse → Qdrant.
    """
    import time as _time

    total = len(messages)
    processed_in_channel = 0
    total_written_qdrant = 0
    start_ts = _time.time()
    last_log_ts = start_ts

    for batch_no, batch in enumerate(
        tqdm_asyncio(
            split_into_batches(messages, max(1, int(batch_size))),
            total=(total + max(1, int(batch_size)) - 1) // max(1, int(batch_size)),
            desc="Ingesting",
            unit="batch",
        ),
        start=1,
    ):
        texts: list[str] = []
        source_messages: list[Message] = []

        for message in batch:
            text_full = (message.message or "").strip()
            if not text_full:
                continue
            if chunk_size > 0:
                parts = _split_text(text_full, chunk_size)
            else:
                parts = _smart_chunk(
                    text_full,
                    threshold=chunk_char_threshold,
                    target=chunk_target_size,
                )
            for part in parts:
                texts.append(part)
                source_messages.append(message)

        if not texts:
            processed_in_channel += len(batch)
            continue

        if channel_hint and isinstance(channel_hint, str) and channel_hint.startswith("@"):
            channel_name = channel_hint.lstrip("@")
        elif channel_hint:
            channel_name = str(channel_hint)
        else:
            channel_name = str(source_messages[0].chat_id)

        try:
            dense_vectors = await _embed_documents_with_retry(
                embedding_client,
                texts,
                channel_hint=channel_hint,
                batch_no=batch_no,
            )
            sparse_results = list(sparse_encoder.embed(texts))
            # ColBERT encoding через gpu_server (если доступен)
            colbert_vectors = None
            colbert_url = os.getenv("COLBERT_URL") or os.getenv("EMBEDDING_TEI_URL", "")
            if colbert_url:
                try:
                    import httpx as _hx
                    async with _hx.AsyncClient(timeout=60) as _cc:
                        _cr = await _cc.post(
                            f"{colbert_url}/colbert-encode",
                            json={"texts": texts, "is_query": False},
                        )
                        _cr.raise_for_status()
                        colbert_vectors = _cr.json()
                except Exception as _ce:
                    logger.warning("ColBERT encode failed (will skip): %s", _ce)
            point_docs = _build_point_docs_flat(
                source_messages=source_messages,
                texts=texts,
                dense_vectors=dense_vectors,
                sparse_results=sparse_results,
                colbert_vectors=colbert_vectors,
                channel_name=channel_name,
                chunk_size=chunk_size,
            )
            written = await qdrant_store.upsert(point_docs)
            total_written_qdrant += written
        except Exception as exc:
            logger.exception(
                "Ошибка при обработке батча channel=%s, batch=%d. Прерываем ingest, чтобы не оставлять дыру в коллекции: %s",
                channel_hint,
                batch_no,
                exc,
            )
            raise

        processed_in_channel += len(batch)

        totals_from_cb = None
        if progress_cb is not None:
            try:
                totals_from_cb = progress_cb(
                    {
                        "processed_in_channel": processed_in_channel,
                        "written_qdrant": total_written_qdrant,
                        "batch_size": len(batch),
                        "total_in_channel": total,
                    }
                )
            except Exception:
                totals_from_cb = None

        should_log = (processed_in_channel % max(1, log_every) == 0) or (
            processed_in_channel >= total
        )
        if should_log:
            now = _time.time()
            elapsed = now - start_ts
            step_elapsed = now - last_log_ts
            last_log_ts = now
            speed = (processed_in_channel / elapsed) if elapsed > 0 else 0.0
            if isinstance(totals_from_cb, dict):
                logger.info(
                    "progress channel=%s processed_in_channel=%d/%d processed_total=%s written_qdrant=%d elapsed_s=%.1f speed_msg_s=%.1f",
                    (channel_hint or "?"),
                    processed_in_channel,
                    total,
                    totals_from_cb.get("processed_total", "-"),
                    total_written_qdrant,
                    step_elapsed,
                    speed,
                )
            else:
                logger.info(
                    "progress channel=%s processed_in_channel=%d/%d written_qdrant=%d elapsed_s=%.1f speed_msg_s=%.1f",
                    (channel_hint or "?"),
                    processed_in_channel,
                    total,
                    total_written_qdrant,
                    step_elapsed,
                    speed,
                )

    return {
        "processed_in_channel": processed_in_channel,
        "written_qdrant": total_written_qdrant,
        "total_in_channel": total,
    }


def _to_utc_naive(dt: datetime) -> datetime:
    """Return timezone-naive UTC datetime for reliable comparison with
    Telegram timestamps (which Telethon returns as naive UTC)."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(UTC).replace(tzinfo=None)


def load_state(
    path: str = os.path.join(
        os.path.dirname(__file__), "sessions", "telegram_ingest.state.json"
    )
) -> dict[str, Any]:
    try:
        import json

        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            logger.info(
                "[checkpoint] state файл отсутствует: %s (пока не используется)", path
            )
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[checkpoint] не удалось загрузить state: %s", e)
        return {}


def save_state(
    state: dict[str, Any],
    path: str = os.path.join(
        os.path.dirname(__file__), "sessions", "telegram_ingest.state.json"
    ),
) -> None:
    try:
        import json

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info("[checkpoint] state сохранён: %s (пока не используется)", path)
    except Exception as e:
        logger.warning("[checkpoint] не удалось сохранить state: %s", e)


async def main() -> None:
    args = _parse_args()
    logger.setLevel(args.log_level.upper())

    channels: list[str] = []
    if getattr(args, "channel", None):
        channels.append(args.channel)
    if getattr(args, "channels", None):
        for part in str(args.channels).split(","):
            part = part.strip()
            if part:
                channels.append(part)
    seen = set()
    channels = [c for c in channels if not (c in seen or seen.add(c))]
    if not channels:
        logger.error("Не указан ни один канал. Используйте --channel или --channels")
        sys.exit(2)

    settings = get_settings()
    collection_name = (
        args.collection or os.getenv("QDRANT_COLLECTION") or settings.qdrant_collection
    )
    batch_size = max(1, int(args.batch_size))
    chunk_size = int(getattr(args, "chunk_size", 0) or 0)
    chunk_char_threshold = int(settings.chunk_char_threshold)
    chunk_target_size = int(settings.chunk_target_size)

    embedding_tei_url = os.getenv("EMBEDDING_TEI_URL") or settings.embedding_tei_url
    logger.info("TEI embedding: %s", embedding_tei_url)
    embedding_client = TEIEmbeddingClient(
        base_url=embedding_tei_url,
        query_instruction=settings.embedding_query_instruction,
    )
    if not await embedding_client.healthcheck():
        logger.warning(
            "TEI embedding service недоступен: %s. Продолжаем (может восстановиться).",
            embedding_tei_url,
        )

    logger.info("Инициализация sparse encoder: Qdrant/bm25 (language=russian)...")
    sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25", language="russian")
    logger.info("Sparse encoder готов")

    qdrant_url = os.getenv("QDRANT_URL") or settings.qdrant_url
    logger.info("Qdrant: %s / collection=%s", qdrant_url, collection_name)
    qdrant_store = QdrantStore(url=qdrant_url, collection=collection_name)
    await qdrant_store.ensure_collection()
    logger.info("Коллекция '%s' готова", collection_name)

    start_iso = _to_utc_naive(date_parser.isoparse(args.since))
    end_iso = _to_utc_naive(date_parser.isoparse(args.until))

    logger.info("Подключение к Telegram…")
    client = await create_telegram_client()
    total_processed = 0
    total_written_qdrant = 0

    def _progress_cb(batch_stats: dict[str, Any]) -> dict[str, Any]:
        nonlocal total_processed, total_written_qdrant
        total_processed += int(batch_stats.get("batch_size", 0))
        total_written_qdrant = int(batch_stats.get("written_qdrant", 0))
        return {
            "processed_total": total_processed,
            "written_qdrant_total": total_written_qdrant,
        }

    try:
        for ch in channels:
            logger.info(
                "▶ start channel=%s dates=%s→%s", ch, start_iso.date(), end_iso.date()
            )

            msgs = await _gather_with_retries(
                client, ch, start_iso, end_iso, args.max_messages
            )
            logger.info("Получено %d сообщений для %s", len(msgs), ch)
            if not msgs:
                logger.warning("Пусто для %s — пропускаем", ch)
                continue

            stats = await ingest_batches(
                messages=msgs,
                batch_size=batch_size,
                embedding_client=embedding_client,
                sparse_encoder=sparse_encoder,
                qdrant_store=qdrant_store,
                channel_hint=ch if isinstance(ch, str) else None,
                chunk_size=chunk_size,
                chunk_char_threshold=chunk_char_threshold,
                chunk_target_size=chunk_target_size,
                progress_cb=_progress_cb,
                log_every=200,
            )

            logger.info(
                "✔ finish channel=%s read=%d written_qdrant=%d",
                ch,
                stats.get("processed_in_channel", 0),
                stats.get("written_qdrant", 0),
            )

        info = await qdrant_store.collection_info()
        logger.info(
            "Все каналы обработаны. В коллекции '%s': %d точек",
            collection_name,
            info.get("points_count", 0),
        )

    except Exception as exc:
        logger.error("Ошибка во время инжеста: %s", exc)
        raise

    finally:
        await client.disconnect()
        await embedding_client.aclose()
        await qdrant_store.aclose()
        logger.info("Соединения закрыты")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
