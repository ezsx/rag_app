"""
Миграция коллекции news_colbert → news_colbert_v2 с обогащённым payload.

SPEC-RAG-12: Payload Enrichment + Re-ingest.

Использование:
    # Из Docker контейнера (ingest):
    python scripts/migrate_collection.py --create-only
    # Затем re-ingest через обычный pipeline

    # Или standalone (для создания коллекции и индексов):
    python scripts/migrate_collection.py --qdrant-url http://localhost:16333

Этот скрипт НЕ делает re-ingest. Он только создаёт коллекцию с правильной
vector config и payload indexes. Re-ingest выполняется через ingest_telegram.py
с обновлённым _build_point_docs_flat().
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from qdrant_client import AsyncQdrantClient, models

# SPEC-RAG-20a: единый source of truth для payload indexes
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
from adapters.qdrant.store import PAYLOAD_INDEXES  # noqa: E402

logger = logging.getLogger(__name__)

NEW_COLLECTION = "news_colbert_v2"

# Точная копия vector config из существующей news_colbert
VECTORS_CONFIG = {
    "dense_vector": models.VectorParams(
        size=1024,
        distance=models.Distance.COSINE,
    ),
    "colbert_vector": models.VectorParams(
        size=128,
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM,
        ),
    ),
}

SPARSE_VECTORS_CONFIG = {
    "sparse_vector": models.SparseVectorParams(
        modifier=models.Modifier.IDF,
        index=models.SparseIndexParams(on_disk=False),
    ),
}


async def create_collection(client: AsyncQdrantClient) -> None:
    """Создать news_colbert_v2 с полной vector config и payload indexes."""

    exists = await client.collection_exists(NEW_COLLECTION)
    if exists:
        logger.warning("Коллекция '%s' уже существует. Удалить вручную если нужно пересоздать.", NEW_COLLECTION)
        return

    logger.info("Создаю коллекцию '%s'...", NEW_COLLECTION)
    await client.create_collection(
        collection_name=NEW_COLLECTION,
        vectors_config=VECTORS_CONFIG,
        sparse_vectors_config=SPARSE_VECTORS_CONFIG,
    )
    logger.info("Коллекция создана. Создаю %d payload indexes...", len(PAYLOAD_INDEXES))

    failed = []
    for field_name, field_schema in PAYLOAD_INDEXES:
        try:
            await client.create_payload_index(
                collection_name=NEW_COLLECTION,
                field_name=field_name,
                field_schema=field_schema,
            )
            logger.info("  Index: %s ✓", field_name)
        except Exception as exc:
            logger.error("  Index: %s FAILED — %s", field_name, exc)
            failed.append(field_name)

    if failed:
        logger.error("ABORTING: %d index(es) failed: %s. Коллекция НЕ готова к re-ingest.", len(failed), failed)
        # Удаляем битую коллекцию
        await client.delete_collection(NEW_COLLECTION)
        logger.info("Коллекция '%s' удалена.", NEW_COLLECTION)
        sys.exit(1)

    logger.info("Готово. Коллекция '%s' готова к re-ingest.", NEW_COLLECTION)


async def verify_collection(client: AsyncQdrantClient) -> None:
    """Проверить что коллекция создана корректно."""
    info = await client.get_collection(NEW_COLLECTION)
    print(f"\nCollection: {NEW_COLLECTION}")
    print(f"  Points: {info.points_count}")
    print(f"  Vectors: {list(info.config.params.vectors.keys())}")
    print(f"  Sparse: {list(info.config.params.sparse_vectors.keys())}")
    print(f"  Payload schema:")
    for field, schema in (info.payload_schema or {}).items():
        print(f"    {field}: {schema}")


def main():
    parser = argparse.ArgumentParser(description="Создать news_colbert_v2 для SPEC-RAG-12")
    parser.add_argument("--qdrant-url", default="http://localhost:16333",
                        help="URL Qdrant (default: localhost:16333)")
    parser.add_argument("--create-only", action="store_true",
                        help="Только создать коллекцию, без verify")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    client = AsyncQdrantClient(url=args.qdrant_url)

    async def run():
        await create_collection(client)
        if not args.create_only:
            await verify_collection(client)
        await client.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
