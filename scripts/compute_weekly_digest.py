#!/usr/bin/env python3
"""
Weekly cron: BERTopic на полном корпусе → hot_topics digest → upsert в weekly_digests.

SPEC-RAG-16. Запуск:
  docker compose -f deploy/compose/compose.dev.yml run --rm api \
    python scripts/compute_weekly_digest.py [--week 2026-W12]

Или локально (с доступом к Qdrant и gpu_server):
  QDRANT_URL=http://localhost:16333 EMBEDDING_TEI_URL=http://localhost:8082 \
    python scripts/compute_weekly_digest.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Добавляем src/ в PYTHONPATH для импорта проектных модулей
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("weekly_digest")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:16333")
SOURCE_COLLECTION = os.getenv("QDRANT_COLLECTION", "news_colbert_v2")
DIGEST_COLLECTION = "weekly_digests"
EMBEDDING_URL = os.getenv("EMBEDDING_TEI_URL", "http://localhost:8082")
LLM_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080")
MODEL_SAVE_DIR = _PROJECT_ROOT / "datasets" / "bertopic_model"


# ---------------------------------------------------------------------------
# Qdrant helpers (sync, через qdrant_client)
# ---------------------------------------------------------------------------

def _get_qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_URL)


def _ensure_digest_collection(client) -> None:
    """Создать weekly_digests коллекцию если не существует."""
    from qdrant_client import models
    collections = [c.name for c in client.get_collections().collections]
    if DIGEST_COLLECTION in collections:
        logger.info("Коллекция %s уже существует", DIGEST_COLLECTION)
        return
    client.create_collection(
        collection_name=DIGEST_COLLECTION,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    )
    # Payload indexes для фильтрации
    client.create_payload_index(DIGEST_COLLECTION, "period", models.PayloadSchemaType.KEYWORD)
    logger.info("Создана коллекция %s", DIGEST_COLLECTION)


def _scroll_all_posts(client) -> Tuple[List[str], List[List[float]], List[Dict]]:
    """Scroll весь корпус, вернуть (texts, embeddings, payloads)."""
    from qdrant_client import models
    texts, embeddings, payloads = [], [], []
    offset = None
    batch_size = 256

    logger.info("Scrolling %s (full corpus)...", SOURCE_COLLECTION)
    while True:
        result = client.scroll(
            collection_name=SOURCE_COLLECTION,
            limit=batch_size,
            offset=offset,
            with_vectors=["dense_vector"],
            with_payload=True,
        )
        points, next_offset = result
        if not points:
            break
        for p in points:
            text = (p.payload or {}).get("text", "")
            vec = p.vector.get("dense_vector") if isinstance(p.vector, dict) else None
            if text and vec:
                texts.append(text)
                embeddings.append(vec)
                payloads.append(p.payload or {})
        offset = next_offset
        if offset is None:
            break

    logger.info("Загружено %d постов с embeddings", len(texts))
    return texts, embeddings, payloads


# ---------------------------------------------------------------------------
# BERTopic
# ---------------------------------------------------------------------------

def _fit_bertopic(texts: List[str], embeddings: np.ndarray):
    """Full-corpus BERTopic fit. Возвращает (model, topics, probs)."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic.vectorizers import ClassTfidfTransformer

    logger.info("BERTopic fit на %d документах...", len(texts))
    t0 = time.time()

    topic_model = BERTopic(
        umap_model=UMAP(
            n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=20, min_samples=5, prediction_data=True,
        ),
        vectorizer_model=CountVectorizer(
            min_df=3, max_df=0.9, ngram_range=(1, 2),
        ),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        top_n_words=10,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(texts, embeddings)

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    logger.info("BERTopic fit завершён за %.1fs, %d topics", time.time() - t0, n_topics)
    return topic_model, topics, probs


def _save_model(model) -> None:
    """Сохранить BERTopic model для monthly cron."""
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_SAVE_DIR / "model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("BERTopic model сохранён в %s", path)


# ---------------------------------------------------------------------------
# Hot score
# ---------------------------------------------------------------------------

def _compute_hot_topics(
    topic_model,
    topics: List[int],
    payloads: List[Dict],
    week_start: datetime,
    week_end: datetime,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Вычислить hot_score для топиков текущей недели."""

    # Индексы постов этой недели
    week_indices = []
    for i, p in enumerate(payloads):
        post_date_str = p.get("date", "")
        if not post_date_str:
            continue
        try:
            post_date = datetime.fromisoformat(post_date_str.replace("Z", "+00:00"))
            post_date_naive = post_date.replace(tzinfo=None)
        except (ValueError, TypeError):
            continue
        if week_start <= post_date_naive < week_end:
            week_indices.append(i)

    if not week_indices:
        logger.warning("Нет постов за неделю %s — %s", week_start, week_end)
        return []

    # Группировка по topic
    topic_posts: Dict[int, List[int]] = {}
    for idx in week_indices:
        t = topics[idx]
        if t == -1:  # outlier
            continue
        topic_posts.setdefault(t, []).append(idx)

    # Предыдущая неделя для velocity
    prev_start = week_start - timedelta(days=7)
    prev_indices = []
    for i, p in enumerate(payloads):
        post_date_str = p.get("date", "")
        if not post_date_str:
            continue
        try:
            post_date = datetime.fromisoformat(post_date_str.replace("Z", "+00:00"))
            post_date_naive = post_date.replace(tzinfo=None)
        except (ValueError, TypeError):
            continue
        if prev_start <= post_date_naive < week_start:
            prev_indices.append(i)

    prev_topic_counts: Dict[int, int] = {}
    for idx in prev_indices:
        t = topics[idx]
        if t != -1:
            prev_topic_counts[t] = prev_topic_counts.get(t, 0) + 1

    # Compute scores
    max_count = max((len(v) for v in topic_posts.values()), default=1)
    results = []

    for topic_id, post_indices in topic_posts.items():
        count = len(post_indices)
        channels = set(payloads[i].get("channel", "") for i in post_indices)
        channels.discard("")

        volume_norm = count / max_count
        channel_diversity = len(channels) / 36.0

        # Recency: exponential decay from median post date
        dates = []
        for i in post_indices:
            d = payloads[i].get("date", "")
            if d:
                try:
                    dates.append(datetime.fromisoformat(d.replace("Z", "+00:00")).replace(tzinfo=None))
                except (ValueError, TypeError):
                    pass
        if dates:
            median_date = sorted(dates)[len(dates) // 2]
            days_ago = (week_end - median_date).total_seconds() / 86400
            recency = 2.0 ** (-days_ago / 3.0)  # half_life=3 days
        else:
            recency = 0.0

        # Velocity
        prev_count = prev_topic_counts.get(topic_id, 0)
        velocity = (count - prev_count) / max(prev_count, 1)
        velocity = min(velocity, 5.0) / 5.0  # normalize to [0, 1]

        hot_score = round(
            0.3 * volume_norm + 0.3 * channel_diversity + 0.3 * recency + 0.1 * velocity,
            3,
        )

        # Topic label from BERTopic
        topic_info = topic_model.get_topic(topic_id)
        keywords = [w for w, _ in (topic_info or [])[:5]]
        label = ", ".join(keywords[:3]) if keywords else f"Topic {topic_id}"

        # Representative posts
        rep_ids = []
        for i in sorted(post_indices, key=lambda x: payloads[x].get("date", ""), reverse=True)[:3]:
            msg_id = payloads[i].get("message_id")
            channel = payloads[i].get("channel", "")
            if msg_id:
                rep_ids.append(f"{channel}:{msg_id}")

        results.append({
            "topic_id": topic_id,
            "label": label,
            "hot_score": hot_score,
            "post_count": count,
            "channels": sorted(channels),
            "keywords": keywords,
            "representative_post_ids": rep_ids,
        })

    results.sort(key=lambda x: x["hot_score"], reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# Burst detection
# ---------------------------------------------------------------------------

def _detect_bursts(
    topics: List[int],
    payloads: List[Dict],
    week_start: datetime,
    week_end: datetime,
    min_channels: int = 5,
    window_hours: int = 48,
) -> List[Dict]:
    """Обнаружить burst events: topic в min_channels+ каналах за window_hours."""
    bursts = []
    # Group week posts by topic
    topic_channel_times: Dict[int, List[Tuple[str, datetime]]] = {}
    for i, p in enumerate(payloads):
        t = topics[i]
        if t == -1:
            continue
        d = p.get("date", "")
        ch = p.get("channel", "")
        if not d or not ch:
            continue
        try:
            dt = datetime.fromisoformat(d.replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, TypeError):
            continue
        if week_start <= dt < week_end:
            topic_channel_times.setdefault(t, []).append((ch, dt))

    for topic_id, entries in topic_channel_times.items():
        entries.sort(key=lambda x: x[1])
        # Sliding window
        for i, (ch_i, dt_i) in enumerate(entries):
            window_end = dt_i + timedelta(hours=window_hours)
            channels_in_window = set()
            for ch_j, dt_j in entries[i:]:
                if dt_j > window_end:
                    break
                channels_in_window.add(ch_j)
            if len(channels_in_window) >= min_channels:
                bursts.append({
                    "topic_id": topic_id,
                    "channels": len(channels_in_window),
                    "first_seen": dt_i.isoformat() + "Z",
                })
                break  # one burst per topic

    return bursts


# ---------------------------------------------------------------------------
# Top entities for the week
# ---------------------------------------------------------------------------

def _top_entities_for_week(
    payloads: List[Dict],
    week_start: datetime,
    week_end: datetime,
    top_n: int = 10,
) -> List[Dict]:
    """Топ entities за неделю из payload.entities[]."""
    entity_counts: Dict[str, int] = {}
    for p in payloads:
        d = p.get("date", "")
        if not d:
            continue
        try:
            dt = datetime.fromisoformat(d.replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, TypeError):
            continue
        if week_start <= dt < week_end:
            for ent in p.get("entities", []):
                entity_counts[ent] = entity_counts.get(ent, 0) + 1

    ranked = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"entity": e, "count": c} for e, c in ranked]


# ---------------------------------------------------------------------------
# LLM summary generation
# ---------------------------------------------------------------------------

def _generate_summary(hot_topics: List[Dict], top_entities: List[Dict], week_label: str) -> str:
    """Сгенерировать дайджест через LLM (Qwen3-30B)."""
    topics_text = "\n".join(
        f"- {t['label']} (score={t['hot_score']}, posts={t['post_count']}, channels={t['channels']})"
        for t in hot_topics[:5]
    )
    entities_text = ", ".join(f"{e['entity']} ({e['count']})" for e in top_entities[:10])

    prompt = (
        f"Составь краткий дайджест AI/ML новостей за неделю {week_label} "
        f"на основе данных из 36 Telegram-каналов.\n\n"
        f"Горячие темы:\n{topics_text}\n\n"
        f"Топ упоминаний: {entities_text}\n\n"
        f"Напиши 200-300 слов на русском. Без вступления, сразу суть."
    )

    payload = json.dumps({
        "model": "qwen3-30b-a3b",
        "messages": [
            {"role": "system", "content": "Ты — AI-аналитик, пишущий еженедельные дайджесты."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 600,
        "temperature": 0.4,
    }).encode()

    try:
        req = urllib.request.Request(
            f"{LLM_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        # Убираем think блоки если есть
        if "<think>" in content:
            import re
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content.strip()
    except Exception as e:
        logger.warning("LLM summary failed: %s — using fallback", e)
        return f"Дайджест за {week_label}: {topics_text}"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_text(text: str) -> List[float]:
    """Embed текст через gpu_server (Qwen3-Embedding-0.6B)."""
    payload = json.dumps({"inputs": [text]}).encode()
    req = urllib.request.Request(
        f"{EMBEDDING_URL}/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data[0]


# ---------------------------------------------------------------------------
# Upsert digest
# ---------------------------------------------------------------------------

def _upsert_digest(
    client,
    week_label: str,
    week_start: datetime,
    week_end: datetime,
    summary: str,
    summary_vector: List[float],
    hot_topics: List[Dict],
    top_entities: List[Dict],
    burst_events: List[Dict],
    post_count: int,
) -> None:
    """Upsert one point в weekly_digests."""
    from qdrant_client import models
    import uuid as _uuid

    point_id = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, f"digest:{week_label}"))

    client.upsert(
        collection_name=DIGEST_COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector=summary_vector,
                payload={
                    "period": week_label,
                    "date_from": week_start.strftime("%Y-%m-%d"),
                    "date_to": week_end.strftime("%Y-%m-%d"),
                    "post_count": post_count,
                    "summary": summary,
                    "topics": hot_topics,
                    "top_entities": top_entities,
                    "burst_events": burst_events,
                },
            )
        ],
    )
    logger.info("Upserted digest for %s (post_count=%d, topics=%d)",
                week_label, post_count, len(hot_topics))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_week(week_str: Optional[str]) -> Tuple[str, datetime, datetime]:
    """Parse --week arg или вернуть текущую неделю."""
    if week_str:
        # Format: 2026-W12
        year, week_num = week_str.split("-W")
        d = datetime.strptime(f"{year} {week_num} 1", "%G %V %u")
    else:
        d = datetime.utcnow()
        # Monday of current week
        d = d - timedelta(days=d.weekday())
        d = d.replace(hour=0, minute=0, second=0, microsecond=0)

    week_start = d - timedelta(days=d.weekday())
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    week_end = week_start + timedelta(days=7)
    label = f"{week_start.isocalendar()[0]}-W{week_start.isocalendar()[1]:02d}"
    return label, week_start, week_end


def main():
    parser = argparse.ArgumentParser(description="Weekly BERTopic digest")
    parser.add_argument("--week", type=str, default=None, help="ISO week: 2026-W12")
    parser.add_argument("--top-n", type=int, default=10, help="Top N hot topics")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM summary generation")
    args = parser.parse_args()

    week_label, week_start, week_end = _parse_week(args.week)
    logger.info("=== Weekly digest for %s (%s — %s) ===", week_label, week_start.date(), week_end.date())

    t0 = time.time()

    # 1. Scroll full corpus
    client = _get_qdrant_client()
    _ensure_digest_collection(client)
    texts, embeddings_list, payloads = _scroll_all_posts(client)
    embeddings = np.array(embeddings_list, dtype=np.float32)

    # 2. BERTopic fit on full corpus
    topic_model, topics, _ = _fit_bertopic(texts, embeddings)
    _save_model(topic_model)

    # 3. Hot topics for this week
    hot = _compute_hot_topics(topic_model, topics, payloads, week_start, week_end, args.top_n)
    logger.info("Hot topics: %d", len(hot))
    for t in hot[:5]:
        logger.info("  %.3f  %s (%d posts, %d channels)", t["hot_score"], t["label"], t["post_count"], len(t["channels"]))

    # 4. Top entities
    top_ents = _top_entities_for_week(payloads, week_start, week_end)

    # 5. Burst detection
    bursts = _detect_bursts(topics, payloads, week_start, week_end)
    if bursts:
        logger.info("Bursts: %d", len(bursts))

    # 6. LLM summary
    if args.skip_llm:
        summary = f"Дайджест за {week_label} (LLM skipped)"
    else:
        summary = _generate_summary(hot, top_ents, week_label)
    logger.info("Summary: %s...", summary[:100])

    # 7. Embed summary
    summary_vector = _embed_text(summary)

    # 8. Count posts this week
    week_post_count = sum(
        1 for p in payloads
        if p.get("date") and _in_week(p["date"], week_start, week_end)
    )

    # 9. Upsert
    _upsert_digest(
        client, week_label, week_start, week_end,
        summary, summary_vector, hot, top_ents, bursts, week_post_count,
    )

    logger.info("=== Done in %.1fs ===", time.time() - t0)


def _in_week(date_str: str, start: datetime, end: datetime) -> bool:
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
        return start <= dt < end
    except (ValueError, TypeError):
        return False


if __name__ == "__main__":
    main()
