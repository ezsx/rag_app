"""
Добавляет ColBERT vectors к существующим points в Qdrant коллекции.
Points уже содержат dense_vector и sparse_vector, но не colbert_vector.

Использование:
    python scripts/add_colbert_vectors.py \
        --collection news_colbert_v2 \
        --qdrant-url http://localhost:16333 \
        --colbert-url http://localhost:8082 \
        --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.request
from typing import Any

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


def colbert_encode_batch(texts: list[str], colbert_url: str, is_query: bool = False) -> list[list[list[float]]]:
    """ColBERT per-token encoding через gpu_server. Batch."""
    body = json.dumps({"texts": texts, "is_query": is_query}).encode()
    req = urllib.request.Request(
        f"{colbert_url}/colbert-encode",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    result = json.loads(urllib.request.urlopen(req, timeout=60).read())
    return result  # list of list of list[float] — [doc][token][128]


def main():
    parser = argparse.ArgumentParser(description="Add ColBERT vectors to existing Qdrant points")
    parser.add_argument("--collection", default="news_colbert_v2")
    parser.add_argument("--qdrant-url", default="http://localhost:16333")
    parser.add_argument("--colbert-url", default="http://localhost:8082")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    client = QdrantClient(url=args.qdrant_url)
    info = client.get_collection(args.collection)
    total = info.points_count
    logger.info("Collection: %s, Points: %d", args.collection, total)

    # Scroll all points, batch encode, update vectors
    offset = None
    processed = 0
    failed = 0
    start_ts = time.time()

    while True:
        points, next_offset = client.scroll(
            collection_name=args.collection,
            offset=offset,
            limit=args.batch_size,
            with_payload=["text"],
            with_vectors=False,
        )

        if not points:
            break

        texts = [p.payload.get("text", "") for p in points]
        ids = [p.id for p in points]

        try:
            colbert_vecs = colbert_encode_batch(texts, args.colbert_url, is_query=False)
        except Exception as exc:
            logger.error("ColBERT encode failed at offset %s: %s", offset, exc)
            failed += len(points)
            offset = next_offset
            if next_offset is None:
                break
            continue

        # Update vectors — добавляем colbert_vector к каждому point
        update_points = []
        for pid, cvec in zip(ids, colbert_vecs):
            if cvec:
                update_points.append(
                    models.PointVectors(
                        id=pid,
                        vector={"colbert_vector": cvec},
                    )
                )

        if update_points:
            client.update_vectors(
                collection_name=args.collection,
                points=update_points,
            )

        processed += len(points)
        elapsed = time.time() - start_ts
        speed = processed / elapsed if elapsed > 0 else 0

        if processed % (args.batch_size * 5) == 0 or next_offset is None:
            logger.info(
                "Progress: %d/%d (%.0f%%) | %.1f pts/s | failed: %d",
                processed, total, processed / total * 100, speed, failed,
            )

        offset = next_offset
        if next_offset is None:
            break

    elapsed = time.time() - start_ts
    logger.info(
        "Done: %d points processed, %d failed, %.1fs total (%.1f pts/s)",
        processed, failed, elapsed, processed / elapsed if elapsed > 0 else 0,
    )

    # Verify
    sample, _ = client.scroll(args.collection, limit=1, with_vectors=True)
    if sample:
        vecs = sample[0].vector
        has_colbert = "colbert_vector" in vecs if isinstance(vecs, dict) else False
        logger.info("Verify: colbert_vector present = %s", has_colbert)


if __name__ == "__main__":
    main()
