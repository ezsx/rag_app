#!/usr/bin/env python3
"""
Monthly cron: вычисление channel profiles → upsert в channel_profiles.

SPEC-RAG-16. Запуск:
  docker compose -f deploy/compose/compose.dev.yml run --rm api \
    python scripts/compute_channel_profiles.py

Или локально:
  QDRANT_URL=http://localhost:16333 EMBEDDING_TEI_URL=http://localhost:8082 \
    python scripts/compute_channel_profiles.py
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
import urllib.request
import uuid as _uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("channel_profiles")

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:16333")
SOURCE_COLLECTION = os.getenv("QDRANT_COLLECTION", "news_colbert_v2")
PROFILES_COLLECTION = "channel_profiles"
EMBEDDING_URL = os.getenv("EMBEDDING_TEI_URL", "http://localhost:8082")
LLM_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080")
MODEL_SAVE_DIR = _PROJECT_ROOT / "datasets" / "bertopic_model"


def _get_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_URL)


def _ensure_profiles_collection(client) -> None:
    """Создать channel_profiles коллекцию если не существует."""
    from qdrant_client import models
    collections = [c.name for c in client.get_collections().collections]
    if PROFILES_COLLECTION in collections:
        logger.info("Коллекция %s существует", PROFILES_COLLECTION)
        return
    client.create_collection(
        collection_name=PROFILES_COLLECTION,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    )
    client.create_payload_index(PROFILES_COLLECTION, "channel", models.PayloadSchemaType.KEYWORD)
    logger.info("Создана коллекция %s", PROFILES_COLLECTION)


def _load_entity_dictionary() -> Dict[str, str]:
    """Загрузить entity_dictionary.json для подсчёта entity_coverage."""
    paths = [
        _PROJECT_ROOT / "datasets" / "entity_dictionary.json",
        Path("/app/datasets/entity_dictionary.json"),
    ]
    for p in paths:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            all_entities = set()

            # Current format: flat dictionary keyed by canonical entity name.
            if isinstance(data, dict) and "categories" not in data:
                for canonical in data.keys():
                    if isinstance(canonical, str) and canonical.strip():
                        all_entities.add(canonical.strip())

            # Legacy format: {"categories": {"org": {"entities": [{"canonical": ...}]}}}
            for category in data.get("categories", {}).values():
                for ent in category.get("entities", []):
                    canonical = ent.get("canonical", "")
                    if canonical:
                        all_entities.add(canonical)

            all_entities.discard("")
            logger.info("Loaded %d canonical entities from dictionary", len(all_entities))
            return {e: e for e in all_entities}
    logger.warning("entity_dictionary.json not found")
    return {}


def _load_bertopic_model():
    """Загрузить fitted BERTopic model из weekly cron."""
    path = MODEL_SAVE_DIR / "model.pkl"
    if not path.exists():
        logger.warning("BERTopic model not found at %s — topics will be empty", path)
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _scroll_all_by_channel(client) -> Dict[str, List[Dict]]:
    """Scroll весь корпус, группировка по channel."""
    channels: Dict[str, List[Dict]] = defaultdict(list)
    offset = None
    batch_size = 256

    logger.info("Scrolling %s...", SOURCE_COLLECTION)
    while True:
        result = client.scroll(
            collection_name=SOURCE_COLLECTION,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = result
        if not points:
            break
        for p in points:
            ch = (p.payload or {}).get("channel", "")
            if ch:
                channels[ch].append(p.payload or {})
        offset = next_offset
        if offset is None:
            break

    total = sum(len(v) for v in channels.values())
    logger.info("Загружено %d постов из %d каналов", total, len(channels))
    return dict(channels)


def _format_top_topics(topic_ids: List[int], topic_model) -> List[str]:
    """Конвертировать topic IDs в human-readable labels через BERTopic."""
    if not topic_ids or topic_model is None:
        return []
    labels = []
    for tid in topic_ids:
        info = topic_model.get_topic(tid)
        if info:
            label = ", ".join(w for w, _ in info[:3])
            labels.append(label)
        else:
            labels.append(f"Topic {tid}")
    return labels


def _top_n_topics(topic_ids: List[int], n: int = 5) -> List[int]:
    """Топ-N topic IDs по частоте (без outlier -1)."""
    counts: Dict[int, int] = {}
    for t in topic_ids:
        if t != -1:
            counts[t] = counts.get(t, 0) + 1
    return [t for t, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]]


def _compute_scores(
    channel: str,
    posts: List[Dict],
    all_known_entities: set,
    total_weeks: int,
    max_posts_across: int,
    entity_first_mentions: Dict[str, datetime],
    post_topics: Optional[List[int]] = None,
    total_topic_count: int = 1,
) -> Dict[str, Any]:
    """Вычислить authority/speed/breadth/volume scores для канала."""

    # Entity coverage
    channel_entities = set()
    for p in posts:
        for e in p.get("entities", []):
            channel_entities.add(e)
    entity_coverage = len(channel_entities & all_known_entities) / max(len(all_known_entities), 1)

    # Consistency: active weeks / total weeks
    active_weeks = set()
    for p in posts:
        yw = p.get("year_week", "")
        if yw:
            active_weeks.add(yw)
    consistency = len(active_weeks) / max(total_weeks, 1)

    # Uniqueness: fraction of original (not forwarded) posts
    original_count = sum(1 for p in posts if not p.get("is_forward", False))
    uniqueness = original_count / max(len(posts), 1)

    # Volume
    volume_norm = len(posts) / max(max_posts_across, 1)

    authority_score = round(
        0.4 * entity_coverage + 0.3 * consistency + 0.2 * uniqueness + 0.1 * volume_norm,
        3,
    )

    # Speed score (v1: top-20 entities only)
    delays = []
    for p in posts:
        post_date_str = p.get("date", "")
        if not post_date_str:
            continue
        try:
            post_date = datetime.fromisoformat(post_date_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, TypeError):
            continue
        for ent in p.get("entities", []):
            global_first = entity_first_mentions.get(ent)
            if global_first:
                delay_days = (post_date - global_first).total_seconds() / 86400
                if delay_days >= 0:
                    delays.append(delay_days)

    if delays:
        delays.sort()
        median_delay = delays[len(delays) // 2]
        max_possible = 270  # ~9 months corpus
        speed_score = round(1.0 - min(median_delay / max_possible, 1.0), 3)
    else:
        speed_score = 0.5  # default

    # Breadth: unique topics / total topics (из BERTopic assignments)
    if post_topics:
        unique_topics = len(set(t for t in post_topics if t != -1))
        breadth_score = round(unique_topics / max(total_topic_count, 1), 3)
        top_topic_ids = _top_n_topics(post_topics, n=5)
    else:
        breadth_score = 0.5
        top_topic_ids = []

    # Volume score (для ranking по metric="volume")
    volume_score = round(volume_norm, 3)

    return {
        "authority_score": authority_score,
        "speed_score": speed_score,
        "breadth_score": breadth_score,
        "volume_score": volume_score,
        "top_topic_ids": top_topic_ids,
        "entity_coverage": round(entity_coverage, 3),
        "consistency": round(consistency, 3),
        "uniqueness": round(uniqueness, 3),
        "volume_norm": round(volume_norm, 3),
    }


def _compute_entity_first_mentions(
    channels_data: Dict[str, List[Dict]],
    top_n: int = 20,
) -> Dict[str, datetime]:
    """Найти first global mention для top-N entities (для speed_score)."""
    entity_counts: Dict[str, int] = defaultdict(int)
    for posts in channels_data.values():
        for p in posts:
            for e in p.get("entities", []):
                entity_counts[e] += 1

    top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_names = {e for e, _ in top_entities}

    first_mentions: Dict[str, datetime] = {}
    for posts in channels_data.values():
        for p in posts:
            date_str = p.get("date", "")
            if not date_str:
                continue
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
            except (ValueError, TypeError):
                continue
            for e in p.get("entities", []):
                if e in top_names:
                    if e not in first_mentions or dt < first_mentions[e]:
                        first_mentions[e] = dt

    return first_mentions


def _generate_profile_summary(channel: str, posts: List[Dict], scores: Dict) -> str:
    """LLM-generated профиль канала (100 слов)."""
    top_ents = defaultdict(int)
    for p in posts:
        for e in p.get("entities", []):
            top_ents[e] += 1
    top5 = sorted(top_ents.items(), key=lambda x: x[1], reverse=True)[:5]

    prompt = (
        f"Составь краткий профиль Telegram-канала @{channel} на основе его AI/ML публикаций.\n"
        f"Постов: {len(posts)}, authority={scores['authority_score']}, "
        f"speed={scores['speed_score']}.\n"
        f"Топ сущности: {', '.join(f'{e}({c})' for e, c in top5)}.\n"
        f"100 слов, на русском. Без вступления."
    )

    payload = json.dumps({
        "model": "qwen3-30b-a3b",
        "messages": [
            {"role": "system", "content": "Ты — аналитик Telegram-каналов."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 250,
        "temperature": 0.3,
    }).encode()

    try:
        req = urllib.request.Request(
            f"{LLM_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        if "<think>" in content:
            import re
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content.strip()
    except Exception as e:
        logger.warning("LLM profile for %s failed: %s", channel, e)
        return f"Канал @{channel}: {len(posts)} постов, authority={scores['authority_score']}"


def _embed_text(text: str) -> List[float]:
    payload = json.dumps({"inputs": [text]}).encode()
    req = urllib.request.Request(
        f"{EMBEDDING_URL}/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data[0]


def main():
    t0 = time.time()
    logger.info("=== Computing channel profiles ===")

    client = _get_client()
    _ensure_profiles_collection(client)

    # Load data
    entity_dict = _load_entity_dictionary()
    all_known_entities = set(entity_dict.keys())
    channels_data = _scroll_all_by_channel(client)

    # Compute total weeks in corpus
    all_weeks = set()
    for posts in channels_data.values():
        for p in posts:
            yw = p.get("year_week", "")
            if yw:
                all_weeks.add(yw)
    total_weeks = len(all_weeks)
    max_posts = max((len(v) for v in channels_data.values()), default=1)

    # Entity first mentions for speed_score
    first_mentions = _compute_entity_first_mentions(channels_data)

    # BERTopic: assign topics to all posts (для breadth + top_topics)
    topic_model = _load_bertopic_model()
    # Build text→topic map for all posts across channels
    post_topic_map: Dict[str, int] = {}  # key = "channel:message_id"
    total_topic_count = 1
    if topic_model is not None:
        logger.info("Assigning topics via BERTopic model (with pre-computed embeddings)...")
        # Scroll с embeddings для transform
        all_texts = []
        all_keys = []
        all_embeddings = []
        for ch, posts in channels_data.items():
            for p in posts:
                text = p.get("text", "")
                msg_id = p.get("message_id", "")
                if text:
                    all_texts.append(text)
                    all_keys.append(f"{ch}:{msg_id}")

        # Нужны embeddings — scroll с vectors
        import numpy as np
        logger.info("Loading embeddings for %d posts...", len(all_texts))
        key_to_idx = {k: i for i, k in enumerate(all_keys)}
        all_embeddings = [None] * len(all_keys)
        offset = None
        while True:
            result = client.scroll(
                collection_name=SOURCE_COLLECTION,
                limit=256, offset=offset,
                with_vectors=["dense_vector"], with_payload=["channel", "message_id"],
            )
            points, next_offset = result
            if not points:
                break
            for p in points:
                vec = p.vector.get("dense_vector") if isinstance(p.vector, dict) else None
                ch = (p.payload or {}).get("channel", "")
                mid = (p.payload or {}).get("message_id", "")
                key = f"{ch}:{mid}"
                idx = key_to_idx.get(key)
                if vec and idx is not None:
                    all_embeddings[idx] = vec
            offset = next_offset
            if offset is None:
                break

        # Filter out posts without embeddings
        valid = [(t, k, e) for t, k, e in zip(all_texts, all_keys, all_embeddings) if e is not None]
        if valid:
            v_texts, v_keys, v_embs = zip(*valid)
            emb_array = np.array(v_embs, dtype=np.float32)
            predicted_topics, _ = topic_model.transform(list(v_texts), emb_array)
            for key, topic_id in zip(v_keys, predicted_topics):
                post_topic_map[key] = topic_id
            total_topic_count = max(len(set(t for t in predicted_topics if t != -1)), 1)
            logger.info("Assigned topics to %d posts, %d unique topics", len(v_texts), total_topic_count)

    # Process each channel
    from qdrant_client import models

    for channel, posts in sorted(channels_data.items()):
        logger.info("Processing %s (%d posts)...", channel, len(posts))

        # Get topic assignments for this channel's posts
        channel_topics = None
        if post_topic_map:
            channel_topics = []
            for p in posts:
                key = f"{channel}:{p.get('message_id', '')}"
                channel_topics.append(post_topic_map.get(key, -1))

        scores = _compute_scores(
            channel, posts, all_known_entities, total_weeks, max_posts, first_mentions,
            post_topics=channel_topics, total_topic_count=total_topic_count,
        )

        # Top entities
        ent_counts: Dict[str, int] = defaultdict(int)
        for p in posts:
            for e in p.get("entities", []):
                ent_counts[e] += 1
        top_ents = [{"entity": e, "count": c}
                    for e, c in sorted(ent_counts.items(), key=lambda x: x[1], reverse=True)[:15]]

        # Post frequency
        weekly_counts: Dict[str, int] = defaultdict(int)
        for p in posts:
            yw = p.get("year_week", "")
            if yw:
                weekly_counts[yw] += 1
        active_weeks_count = len(weekly_counts)
        avg_per_week = round(len(posts) / max(active_weeks_count, 1), 1)

        # Profile summary via LLM
        summary = _generate_profile_summary(channel, posts, scores)
        summary_vector = _embed_text(summary)

        # Upsert
        point_id = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, f"profile:{channel}"))
        client.upsert(
            collection_name=PROFILES_COLLECTION,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=summary_vector,
                    payload={
                        "channel": channel,
                        "display_name": "",  # TODO: fill from channel metadata
                        "total_posts": len(posts),
                        "post_frequency": {
                            "avg_per_week": avg_per_week,
                            "active_weeks": active_weeks_count,
                        },
                        "top_entities": top_ents,
                        "top_topics": _format_top_topics(scores.get("top_topic_ids", []), topic_model),
                        "authority_score": scores["authority_score"],
                        "speed_score": scores["speed_score"],
                        "breadth_score": scores["breadth_score"],
                        "volume_score": scores["volume_score"],
                        "profile_summary": summary,
                        "updated_at": datetime.utcnow().isoformat() + "Z",
                    },
                )
            ],
        )
        logger.info("  → %s: authority=%.3f speed=%.3f breadth=%.3f",
                     channel, scores["authority_score"], scores["speed_score"], scores["breadth_score"])

    logger.info("=== Done in %.1fs (%d channels) ===", time.time() - t0, len(channels_data))


if __name__ == "__main__":
    main()
