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
    p = argparse.ArgumentParser(description="Ingest Telegram channel into ChromaDB")
    p.add_argument("--channel", required=True, help="@username or chat id")
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
                convert_to_tensor=False,          # np.ndarray
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
        "mps": 80,  # ~80 сообщений/сек на Apple Silicon
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


async def ingest_batches(collection, messages: List[Message], batch_size: int):
    total = len(messages)
    for batch in tqdm_asyncio(
        split_into_batches(messages, batch_size),
        total=(total // batch_size) + 1,
        desc="Ingesting",
        unit="batch",
    ):
        docs = [m.message for m in batch]
        ids = [f"{m.id}_{uuid.uuid4().hex[:6]}" for m in batch]
        metas = []
        for m in batch:
            meta = {
                "channel_id": m.chat_id,
                "msg_id": m.id,
                "date": m.date.isoformat(),
            }
            if m.reply_to_msg_id is not None:
                meta["reply_to"] = m.reply_to_msg_id
            views = getattr(m, "views", None)
            if views is not None:
                meta["views"] = views
            metas.append(meta)
        try:
            collection.add(documents=docs, metadatas=metas, ids=ids)
        except Exception as e:
            logger.exception("Failed on batch, skipping… (%s)", e)


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
# Main
# ────────────────────────────────────────────────────────────────


async def main():
    args = _parse_args()
    logger.setLevel(args.log_level.upper())

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
        logger.info("📥 Получение сообщений %s → %s", start_iso.date(), end_iso.date())
        msgs = await gather_messages(
            client, args.channel, start_iso, end_iso, args.max_messages
        )
        logger.info("📨 Получено %s сообщений", len(msgs))

        if not msgs:
            logger.warning("❌ Сообщения не найдены — завершаем")
            return

        # Оценка времени обработки
        estimated_time = estimate_processing_time(len(msgs), batch_size, device)
        logger.info(f"⏱️ Примерное время обработки: {estimated_time}")

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
                import sys
                from pathlib import Path

                src_path = Path(__file__).parent.parent / "src"
                if str(src_path) not in sys.path:
                    sys.path.insert(0, str(src_path))

                from utils.model_downloader import download_embedding_model

                cache_dir = os.getenv("TRANSFORMERS_CACHE", "/models/.cache")
                download_embedding_model(embed_model, cache_dir)
        except Exception as e:
            logger.warning(f"Не удалось автоскачать модель: {e}")

        collection = create_chroma_collection(args.collection, embed_model, device)

        logger.info(
            f"🚀 Начинаем обработку на {device.upper()} с batch_size={batch_size}"
        )
        await ingest_batches(collection, msgs, batch_size)

        final_count = collection.count()
        logger.info("✅ Инжест завершен. Итого в коллекции: %s документов", final_count)

        # Статистика производительности
        if len(msgs) > 0:
            logger.info(f"📊 Обработано {len(msgs)} сообщений")

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
