"""
Telegram → ChromaDB ingestor
----------------------------
CLI‑скрипт, который загружает сообщения публичного/приватного канала Telegram
в указанном диапазоне дат, рассчитывает встраивания (BGE‑base‑v1.5 или другая
указанная модель) и пачками пишет их в коллекцию ChromaDB.

Запуск (пример):
$ python -m scripts.ingest_telegram \
      --channel @some_channel \
      --since 2024-06-01 \
      --until 2024-07-01 \
      --collection tg_some_channel

Обязательно положите в .env:
TG_API_ID=123456
TG_API_HASH=abcdef0123456789abcdef0123456789
# если используете user‑account логин → TG_PHONE=+79995555555
# либо BOT_TOKEN=123456:ABC‑DEF… (тогда авторизация по боту)

CHROMA_HOST=localhost
CHROMA_PORT=8000

Скрипт безопасно продолжит работу после обрыва сети или FloodWait,
пишет прогресс и ожидаемое время завершения.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
import logging
import argparse
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from dateutil import parser as date_parser, tz

from telethon import TelegramClient, errors as tg_errors, events
from telethon.errors import FloodWaitError, SessionPasswordNeededError
from telethon.tl.types import Message

import numpy as np
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import torch
from sentence_transformers import SentenceTransformer

# ────────────────────────────────────────────────────────────────
# Config & CLI parsing
# ────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest Telegram channel(s) into ChromaDB")
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
    p.add_argument("--collection", required=True, help="Chroma collection name")
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (auto-detected based on device)",
    )
    p.add_argument(
        "--embed-model-key", default=None, help="Embedding model key from config"
    )
    p.add_argument(
        "--embed-model", default=None, help="Direct embedding model name (fallback)"
    )
    p.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
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
    p.add_argument(
        "--gpu-batch-multiplier", type=int, default=4, help="GPU batch size multiplier"
    )
    return p.parse_args()


load_dotenv()

# ────────────────────────────────────────────────────────────────
# Logger
# ────────────────────────────────────────────────────────────────
logger = logging.getLogger("ingest_telegram")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# ────────────────────────────────────────────────────────────────
# Telegram utils
# ────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────
# Chroma utils
# ────────────────────────────────────────────────────────────────


def detect_optimal_device() -> str:
    """Автоматически определяет лучшее устройство для вычислений"""
    if torch.cuda.is_available():
        logger.info(f"🚀 CUDA доступна: {torch.cuda.get_device_name(0)}")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("🚀 MPS (Apple Silicon) доступен")
        return "mps"
    else:
        logger.info("💻 Используем CPU")
        return "cpu"


def get_optimal_batch_size(
    device: str, base_batch_size: int = 64, gpu_multiplier: int = 4
) -> int:
    """Определяет оптимальный размер батча в зависимости от устройства"""
    if device in ["cuda", "mps"]:
        optimal_size = base_batch_size * gpu_multiplier
        logger.info(f"📊 GPU detected, increasing batch size to {optimal_size}")
        return optimal_size
    else:
        logger.info(f"📊 CPU detected, using batch size {base_batch_size}")
        return base_batch_size


def resolve_embedding_model(
    embed_model_key: Optional[str], embed_model: Optional[str]
) -> str:
    """Определяет модель embedding из конфигурации или fallback"""
    # Попытка использовать конфигурацию из .env как в API
    if not embed_model_key:
        embed_model_key = os.getenv("EMBEDDING_MODEL_KEY", "multilingual-e5-large")

    # Импортируем конфигурацию моделей
    try:
        import sys
        from pathlib import Path

        # Добавляем src в путь
        src_path = Path(__file__).parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from utils.model_downloader import RECOMMENDED_MODELS

        if embed_model_key in RECOMMENDED_MODELS["embedding"]:
            model_name = RECOMMENDED_MODELS["embedding"][embed_model_key]["name"]
            logger.info(f"🎯 Используем модель из конфигурации: {model_name}")
            return model_name
    except Exception as e:
        logger.warning(f"Не удалось загрузить конфигурацию моделей: {e}")

    # Fallback
    if embed_model:
        logger.info(f"🎯 Используем указанную модель: {embed_model}")
        return embed_model

    # По умолчанию для русского языка
    default_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    logger.info(f"🎯 Используем модель по умолчанию: {default_model}")
    return default_model


class FastEmbeddingFunction:
    """Оптимизированная функция embedding с поддержкой GPU и батчинга"""

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        logger.info(f"⚡ Загружаем embedding модель: {model_name} на {device}")
        self.model = SentenceTransformer(model_name, device=device)
        if device in ["cuda", "mps"]:
            try:
                self.model.half()
            except AttributeError:
                pass
            logger.info("🔥 Включена GPU оптимизация")

    # sig must be (self, input)
    def __call__(self, input):
        if not input:
            return np.empty((0, 768), dtype=np.float32)
        with torch.no_grad():
            emb = self.model.encode(
                input,
                convert_to_tensor=False,  # np.ndarray
                device=self.device,
                show_progress_bar=False,
                batch_size=32 if self.device == "cpu" else 128,
            )
            return emb


def create_chroma_collection(name: str, embed_model: str, device: str):
    """Connect to Chroma HTTP server and obtain (or create) a collection.

    The Chroma Python API changed multiple times; we probe the available
    symbols/parameters at runtime so that the script works with 0.4.x and
    newer versions alike."""

    # ---------------------------
    # Build embedding function
    # ---------------------------
    from inspect import signature

    hf_kwargs: Dict[str, Any] = {"model_name": embed_model}

    sig = signature(SentenceTransformerEmbeddingFunction)

    if "device" in sig.parameters:
        hf_kwargs["device"] = device
    elif "device_type" in sig.parameters:
        hf_kwargs["device_type"] = device

    # Optional normalization flag changed name across versions.
    if "normalize" in sig.parameters:
        hf_kwargs["normalize"] = True
    elif "normalize_embeddings" in sig.parameters:
        hf_kwargs["normalize_embeddings"] = True

    # Добавляем ключ HuggingFace из окружения, если доступен,
    # так как новые версии chromadb требуют обязательного `api_key`.
    api_key = os.getenv("CHROMA_HUGGINGFACE_API_KEY") or os.getenv(
        "HUGGINGFACE_API_KEY"
    )
    if api_key:
        if "api_key" in sig.parameters:
            hf_kwargs["api_key"] = api_key
        elif "huggingface_api_key" in sig.parameters:
            hf_kwargs["huggingface_api_key"] = api_key

    # Используем нашу оптимизированную функцию embedding
    try:
        embed_fn = FastEmbeddingFunction(embed_model, device)
        logger.info(f"✅ Используем оптимизированную embedding функцию")
    except Exception as e:
        logger.warning(f"Fallback на стандартную embedding функцию: {e}")
        embed_fn = SentenceTransformerEmbeddingFunction(**hf_kwargs)

    # ---------------------------
    # Connect to Chroma
    # ---------------------------
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", 8000))

    if hasattr(chromadb, "HttpClient"):
        client = chromadb.HttpClient(host=host, port=port)
    else:
        # Fallback to old API
        client = chromadb.Client(host=host, port=port)

    # ---------------------------
    # Get or create collection (with embedding function attached)
    # ---------------------------
    try:
        collection = client.get_collection(name, embedding_function=embed_fn)
        logger.info("Using existing collection %s (count=%s)", name, collection.count())
    except Exception as e:
        # get_collection may raise NotFoundError (old API) or ValueError (newer)
        logger.debug("Collection %s not found (%s), creating new", name, e)
        collection = client.create_collection(
            name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embed_fn,
        )
        logger.info("Created new collection %s", name)

    return collection


# ────────────────────────────────────────────────────────────────
# Batch helpers
# ────────────────────────────────────────────────────────────────


def split_into_batches(seq: List[Any], batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def estimate_processing_time(total_messages: int, batch_size: int, device: str) -> str:
    """Оценивает время обработки в зависимости от устройства"""
    # Примерные скорости (сообщений в секунду)
    speeds = {
        "cuda": 100,  # ~100 сообщений/сек на GPU
        "cpu": 5,  # ~5 сообщений/сек на CPU
    }

    speed = speeds.get(device, 5)
    estimated_seconds = total_messages / speed

    if estimated_seconds < 60:
        return f"{estimated_seconds:.0f} секунд"
    elif estimated_seconds < 3600:
        return f"{estimated_seconds/60:.1f} минут"
    else:
        return f"{estimated_seconds/3600:.1f} часов"


async def gather_messages(
    client: TelegramClient,
    channel: str,
    start: datetime,
    end: datetime,
    limit: Optional[int] = None,
) -> List[Message]:
    """Fetch messages with *start* ≤ date < *end*.

    Telegram's API returns history in reverse-chronological order by default
    (newest → oldest). We leverage that to avoid walking the whole history of
    busy channels: request messages older than *end* and stop once we cross
    *start*."""

    collected: List[Message] = []

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


def _split_text(text: str, chunk_size: int) -> List[str]:
    if chunk_size and chunk_size > 0 and len(text) > chunk_size:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    return [text]


# ────────────────────────────────────────────────────────────────
# Resilience helpers
# ────────────────────────────────────────────────────────────────


async def _gather_with_retries(
    client: TelegramClient,
    channel: str,
    start: datetime,
    end: datetime,
    limit: Optional[int] = None,
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


async def ingest_batches(
    collection_name: str,
    collection,
    messages: List[Message],
    batch_size: int,
    chunk_size: int = 0,
    channel_hint: Optional[str] = None,
    progress_cb: Optional[Any] = None,
    log_every: int = 200,
):
    total = len(messages)
    processed_in_channel = 0
    total_written_chroma = 0
    total_written_bm25 = 0
    import time as _time

    start_ts = _time.time()
    last_log_ts = start_ts
    for batch in tqdm_asyncio(
        split_into_batches(messages, batch_size),
        total=(total // batch_size) + 1,
        desc="Ingesting",
        unit="batch",
    ):
        docs: List[str] = []
        ids: List[str] = []
        metas: List[Dict[str, Any]] = []
        bm25_docs_local: List[Any] = []

        for m in batch:
            text = (m.message or "").strip()
            if not text:
                continue
            parts = _split_text(text, chunk_size)
            for idx, part in enumerate(parts):
                docs.append(part)
                # Детерминированный id: channel:msg:chunk
                ids.append(f"{m.chat_id}:{m.id}:{idx}")
                meta = {
                    "channel_id": m.chat_id,
                    "msg_id": m.id,
                    "date": m.date.isoformat(),
                }
                if (
                    channel_hint
                    and isinstance(channel_hint, str)
                    and channel_hint.startswith("@")
                ):
                    meta["channel_username"] = channel_hint
                if m.reply_to_msg_id is not None:
                    meta["reply_to"] = m.reply_to_msg_id
                views = getattr(m, "views", None)
                if views is not None:
                    meta["views"] = views
                metas.append(meta)

                # BM25 документ по каждому чанку
                try:
                    from adapters.search.bm25_index import BM25Doc
                except Exception:
                    BM25Doc = None  # type: ignore
                if BM25Doc is not None:
                    date_iso = _to_utc_naive(m.date).date().isoformat()
                    date_days = _to_utc_naive(m.date).date().toordinal()
                    bm25_docs_local.append(
                        BM25Doc(
                            doc_id=f"{m.chat_id}:{m.id}:{idx}",
                            text=" ".join(part.split()),
                            channel_id=int(m.chat_id),
                            channel_username=(
                                channel_hint
                                if (
                                    isinstance(channel_hint, str)
                                    and channel_hint.startswith("@")
                                )
                                else None
                            ),
                            date_days=int(date_days),
                            date_iso=date_iso,
                            views=getattr(m, "views", None),
                            reply_to=getattr(m, "reply_to_msg_id", None),
                            msg_id=int(m.id),
                        )
                    )

        try:
            if hasattr(collection, "upsert"):
                collection.upsert(documents=docs, metadatas=metas, ids=ids)
            else:
                collection.add(documents=docs, metadatas=metas, ids=ids)
            total_written_chroma += len(ids)
        except Exception as e:
            logger.exception("Failed on batch, skipping… (%s)", e)

        # BM25: добавление документов (батч)
        try:
            from adapters.search.bm25_index import BM25IndexManager
            from core.settings import get_settings

            settings = get_settings()
            bm25_root = settings.bm25_index_root
            mgr = BM25IndexManager(index_root=bm25_root)
            if bm25_docs_local:
                mgr.add_documents(
                    collection=collection_name, docs=bm25_docs_local, commit_every=1000
                )
                total_written_bm25 += len(bm25_docs_local)
        except Exception as e:
            logger.warning(f"BM25 add_documents failed: {e}")

        processed_in_channel += len(batch)
        # Периодические логи
        should_log = (processed_in_channel % max(1, log_every) == 0) or (
            processed_in_channel >= total
        )
        totals_from_cb = None
        if progress_cb is not None:
            try:
                totals_from_cb = progress_cb(
                    {
                        "processed_in_channel": processed_in_channel,
                        "written_chroma": total_written_chroma,
                        "written_bm25": total_written_bm25,
                        "batch_size": len(batch),
                        "total_in_channel": total,
                    }
                )
            except Exception:
                totals_from_cb = None
        if should_log:
            now = _time.time()
            elapsed = now - start_ts
            step_elapsed = now - last_log_ts
            last_log_ts = now
            speed = (processed_in_channel / elapsed) if elapsed > 0 else 0.0
            if isinstance(totals_from_cb, dict):
                logger.info(
                    "progress channel=%s processed_in_channel=%d/%d processed_total=%s written_chroma=%d written_bm25=%d elapsed_s=%.1f speed_msg_s=%.1f",
                    (channel_hint or "?"),
                    processed_in_channel,
                    total,
                    totals_from_cb.get("processed_total", "-"),
                    total_written_chroma,
                    total_written_bm25,
                    step_elapsed,
                    speed,
                )
            else:
                logger.info(
                    "progress channel=%s processed_in_channel=%d/%d written_chroma=%d written_bm25=%d elapsed_s=%.1f speed_msg_s=%.1f",
                    (channel_hint or "?"),
                    processed_in_channel,
                    total,
                    total_written_chroma,
                    total_written_bm25,
                    step_elapsed,
                    speed,
                )

    return {
        "processed_in_channel": processed_in_channel,
        "written_chroma": total_written_chroma,
        "written_bm25": total_written_bm25,
        "total_in_channel": total,
    }


# ────────────────────────────────────────────────────────────────
# Date helpers
# ────────────────────────────────────────────────────────────────


def _to_utc_naive(dt: datetime) -> datetime:
    """Return timezone-naive UTC datetime for reliable comparison with
    Telegram timestamps (which Telethon returns as naive UTC)."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


# ────────────────────────────────────────────────────────────────
# Checkpoint stubs (not used yet)
# ────────────────────────────────────────────────────────────────


def load_state(
    path: str = os.path.join(
        os.path.dirname(__file__), "sessions", "telegram_ingest.state.json"
    )
) -> Dict[str, Any]:
    try:
        import json

        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            logger.info(
                "[checkpoint] state файл отсутствует: %s (пока не используется)", path
            )
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[checkpoint] не удалось загрузить state: %s", e)
        return {}


def save_state(
    state: Dict[str, Any],
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


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────


async def main():
    args = _parse_args()
    logger.setLevel(args.log_level.upper())

    # Собираем итоговый список каналов
    channels: List[str] = []
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

    # Определяем оптимальное устройство
    if args.device == "auto":
        device = detect_optimal_device()
    else:
        device = args.device
        logger.info(f"🎯 Используем указанное устройство: {device}")

    # Определяем модель embedding
    embed_model = resolve_embedding_model(args.embed_model_key, args.embed_model)

    # Определяем оптимальный размер батча
    if args.batch_size is None:
        batch_size = get_optimal_batch_size(
            device, gpu_multiplier=args.gpu_batch_multiplier
        )
    else:
        batch_size = args.batch_size
        logger.info(f"📊 Используем указанный batch size: {batch_size}")

    start_iso = _to_utc_naive(date_parser.isoparse(args.since))
    end_iso = _to_utc_naive(date_parser.isoparse(args.until))

    logger.info("🔗 Подключение к Telegram…")
    client = await create_telegram_client()
    try:
        logger.info(
            "🔗 Подключение к ChromaDB %s:%s",
            os.getenv("CHROMA_HOST", "localhost"),
            os.getenv("CHROMA_PORT", 8000),
        )

        # Автоскачивание embedding модели если необходимо
        try:
            auto_download = (
                os.getenv("AUTO_DOWNLOAD_EMBEDDING", "true").lower() == "true"
            )
            if auto_download:
                logger.info("📥 Проверка и скачивание embedding модели...")
                import sys as _sys
                from pathlib import Path as _Path

                src_path = _Path(__file__).parent.parent / "src"
                if str(src_path) not in _sys.path:
                    _sys.path.insert(0, str(src_path))

                from utils.model_downloader import download_embedding_model

                cache_dir = os.getenv("TRANSFORMERS_CACHE", "/models/.cache")
                download_embedding_model(embed_model, cache_dir)
        except Exception as e:
            logger.warning(f"Не удалось автоскачать модель: {e}")

        collection = create_chroma_collection(args.collection, embed_model, device)

        total_processed = 0
        total_written_chroma = 0
        total_written_bm25 = 0

        def _progress_cb(batch_stats: Dict[str, Any]):
            nonlocal total_processed, total_written_chroma, total_written_bm25
            total_processed += int(batch_stats.get("batch_size", 0))
            total_written_chroma = int(batch_stats.get("written_chroma", 0))
            total_written_bm25 = int(batch_stats.get("written_bm25", 0))
            logger.debug(
                "progress_total processed_total=%d written_chroma_total=%d written_bm25_total=%d",
                total_processed,
                total_written_chroma,
                total_written_bm25,
            )
            return {
                "processed_total": total_processed,
                "written_chroma_total": total_written_chroma,
                "written_bm25_total": total_written_bm25,
            }

        for ch in channels:
            logger.info(
                "▶ start channel=%s dates=%s→%s", ch, start_iso.date(), end_iso.date()
            )

            msgs = await _gather_with_retries(
                client, ch, start_iso, end_iso, args.max_messages
            )
            logger.info("📨 Получено %s сообщений для %s", len(msgs), ch)
            if not msgs:
                logger.warning("Пусто для %s — пропускаем", ch)
                continue

            estimated_time = estimate_processing_time(len(msgs), batch_size, device)
            est_batches = (len(msgs) + batch_size - 1) // batch_size
            logger.info(
                "⏱️ Оценка: %s, батчей ~%d (batch_size=%d)",
                estimated_time,
                est_batches,
                batch_size,
            )

            logger.info(
                f"🚀 Начинаем обработку канала {ch} на {device.upper()} с batch_size={batch_size}"
            )

            stats = await ingest_batches(
                args.collection,
                collection,
                msgs,
                batch_size,
                chunk_size=int(getattr(args, "chunk_size", 0) or 0),
                channel_hint=ch if isinstance(ch, str) else None,
                progress_cb=_progress_cb,
                log_every=200,
            )

            logger.info(
                "✔ finish channel=%s read=%d written_chroma=%d written_bm25=%d",
                ch,
                stats.get("processed_in_channel", 0),
                stats.get("written_chroma", 0),
                stats.get("written_bm25", 0),
            )

        final_count = collection.count()
        logger.info(
            "🏁 Все каналы обработаны. В коллекции теперь: %s документов",
            final_count,
        )

        # Путь к BM25 индексу (если включён)
        try:
            from adapters.search.bm25_index import BM25IndexManager
            from core.settings import get_settings

            settings = get_settings()
            bm25_root = settings.bm25_index_root
            mgr = BM25IndexManager(index_root=bm25_root)
            handle = mgr.get_or_create(args.collection)
            logger.info(
                "BM25 index path for collection '%s': %s",
                args.collection,
                handle.paths.get("root"),
            )
        except Exception as e:
            logger.warning(f"BM25 unavailable or disabled: {e}")

    except Exception as e:
        logger.error(f"❌ Ошибка во время инжеста: {e}")
        raise
    finally:
        await client.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
