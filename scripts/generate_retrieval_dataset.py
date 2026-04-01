#!/usr/bin/env python3
"""
Генерация retrieval eval датасета из текущей Qdrant коллекции.

Использует локальный LLM (Qwen3.5 на :8080) для генерации натуральных
поисковых вопросов из текстов постов. Стратифицированная выборка по каналам.

Запуск:
    python scripts/generate_retrieval_dataset.py \
        --collection news_colbert_v2 \
        --qdrant-url http://localhost:16333 \
        --llm-url http://localhost:8080 \
        --output datasets/eval_retrieval_calibration.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8")
random.seed(42)

# ─── LLM ─────────────────────────────────────────────────────────────

QUESTION_PROMPT = """Ты генератор поисковых запросов. Дан текст поста из Telegram-канала про AI/ML.
Сгенерируй ОДИН естественный поисковый вопрос на русском языке, на который этот пост отвечает.

Правила:
- Вопрос должен быть таким, какой реальный пользователь задал бы в поисковике
- НЕ копируй текст поста — переформулируй своими словами
- НЕ упоминай Telegram, канал или автора
- Вопрос должен быть конкретным (не "Что нового в AI?")
- Длина: 10-80 символов
- Категория вопроса: {category}

Текст поста ({channel}, {date}):
---
{text}
---

Ответь ТОЛЬКО вопросом, без кавычек и пояснений."""


def generate_question(
    text: str, channel: str, date: str, category: str, llm_url: str,
) -> Optional[str]:
    """Генерирует вопрос через локальный LLM."""
    prompt = QUESTION_PROMPT.format(
        text=text[:800], channel=channel, date=date, category=category,
    )
    body = json.dumps({
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.7,
        "stop": ["\n\n"],
    }).encode()
    req = urllib.request.Request(
        f"{llm_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
        raw = resp["choices"][0]["message"]["content"].strip()
        # Убираем think tags если есть
        raw = re.sub(r"</?think>", "", raw).strip()
        # Убираем кавычки
        raw = raw.strip('"«»""\' ')
        # Убираем нумерацию ("1. Вопрос")
        raw = re.sub(r"^\d+[.)]\s*", "", raw)
        # Валидация
        if len(raw) < 10 or len(raw) > 150:
            return None
        return raw
    except Exception as e:
        print(f"    LLM error: {e}")
        return None


# ─── Qdrant helpers ──────────────────────────────────────────────────

def qdrant_post(url: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{url}/collections/{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def get_channels(qdrant_url: str, collection: str) -> List[dict]:
    resp = qdrant_post(qdrant_url, f"{collection}/facet", {"key": "channel", "limit": 100})
    return resp.get("result", {}).get("hits", [])


def scroll_channel(
    qdrant_url: str, collection: str, channel: str, limit: int = 50,
) -> List[dict]:
    body = {
        "filter": {"must": [{"key": "channel", "match": {"value": channel}}]},
        "limit": limit,
        "with_payload": True,
        "order_by": {"key": "date", "direction": "desc"},
    }
    resp = qdrant_post(qdrant_url, f"{collection}/points/scroll", body)
    return resp.get("result", {}).get("points", [])


# ─── Post selection ──────────────────────────────────────────────────

def select_posts(posts: List[dict], n: int = 3) -> List[dict]:
    """Выбирает N постов: свежий, старый, случайный. Фильтрует chunks и короткие."""
    good = []
    for p in posts:
        pay = p.get("payload", {})
        text = pay.get("text", "")
        pid = str(p.get("id", ""))
        # Пропускаем chunks (id формата channel_id:msg_id:chunk_idx)
        parts = pid.split(":")
        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
            continue
        # Минимум 150 chars текста для осмысленного вопроса
        if len(text) >= 150:
            good.append(p)

    if not good:
        return []
    if len(good) <= n:
        return good

    # Свежий + старый + случайные из середины
    selected = [good[0], good[-1]]
    middle = good[1:-1]
    if middle:
        selected.extend(random.sample(middle, min(n - 2, len(middle))))
    return selected[:n]


def assign_category(payload: dict, idx: int) -> str:
    """Назначает категорию запроса для разнообразия."""
    entities = payload.get("entities") or []
    date = payload.get("date", "")

    # Ротация категорий
    if entities and idx % 4 == 0:
        return "entity_centric"
    if date and idx % 4 == 1:
        return "temporal"
    if idx % 4 == 2:
        return "factual"
    return "topic_search"


# ─── Main ────────────────────────────────────────────────────────────

def generate_dataset(
    qdrant_url: str,
    collection: str,
    llm_url: str,
    posts_per_channel: int = 3,
    max_total: int = 100,
) -> List[dict]:
    channels = get_channels(qdrant_url, collection)
    print(f"Каналов: {len(channels)}")

    candidates: List[dict] = []

    # Собираем посты из всех каналов
    for ch_info in channels:
        channel = ch_info["value"]
        posts = scroll_channel(qdrant_url, collection, channel, limit=30)
        selected = select_posts(posts, n=posts_per_channel)

        for idx, post in enumerate(selected):
            pay = post.get("payload", {})
            candidates.append({
                "channel": channel,
                "message_id": pay.get("message_id"),
                "date": (pay.get("date") or "")[:10],
                "text": pay.get("text", ""),
                "entities": pay.get("entities") or [],
                "category": assign_category(pay, len(candidates) + idx),
            })

    print(f"Кандидатов: {len(candidates)}")

    # Перемешиваем и берём max_total
    random.shuffle(candidates)
    candidates = candidates[:max_total]

    # Генерируем вопросы через LLM
    dataset: List[dict] = []
    failed = 0
    t_start = time.time()

    for i, cand in enumerate(candidates):
        question = generate_question(
            text=cand["text"],
            channel=cand["channel"],
            date=cand["date"],
            category=cand["category"],
            llm_url=llm_url,
        )

        if not question:
            failed += 1
            # Fallback: первое предложение
            lines = [l.strip() for l in cand["text"].split("\n") if len(l.strip()) >= 20]
            question = (lines[0][:80] if lines else cand["text"][:80]) + "?"
            cand["category"] = "fallback"

        item = {
            "id": f"cal_{cand['channel']}_{cand['message_id']}",
            "query": question,
            "category": cand["category"],
            "expected_documents": [f"{cand['channel']}:{cand['message_id']}"],
            "channel": cand["channel"],
            "date": cand["date"],
            "source_text_preview": cand["text"][:150],
        }
        dataset.append(item)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(candidates) - i - 1)
            print(f"  [{i+1}/{len(candidates)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining | {question[:60]}")

    elapsed = time.time() - t_start

    # Статистика
    cats = {}
    for item in dataset:
        cats[item["category"]] = cats.get(item["category"], 0) + 1
    chs = set(item["channel"] for item in dataset)

    print(f"\nСгенерировано: {len(dataset)} вопросов за {elapsed:.0f}s")
    print(f"  Категории: {cats}")
    print(f"  Каналов: {len(chs)}")
    print(f"  LLM failures (fallback): {failed}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate retrieval eval dataset from Qdrant + LLM")
    parser.add_argument("--collection", default="news_colbert_v2")
    parser.add_argument("--qdrant-url", default="http://localhost:16333")
    parser.add_argument("--llm-url", default="http://localhost:8080")
    parser.add_argument("--output", default="datasets/eval_retrieval_calibration.json")
    parser.add_argument("--posts-per-channel", type=int, default=3)
    parser.add_argument("--max-total", type=int, default=100)
    args = parser.parse_args()

    dataset = generate_dataset(
        args.qdrant_url, args.collection, args.llm_url,
        posts_per_channel=args.posts_per_channel,
        max_total=args.max_total,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено в {args.output}")

    # Превью
    print(f"\nПревью (5 вопросов):")
    for item in dataset[:5]:
        print(f"  [{item['category']:15s}] {item['query']}")
        print(f"    → {item['expected_documents']} | {item['date']}")


if __name__ == "__main__":
    main()
