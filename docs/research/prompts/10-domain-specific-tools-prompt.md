# Deep Research: Домен-специфичные инструменты для Telegram AI/ML News RAG-агента

> Это **дополнительный ресерч** к R16. R16 дал generic RAG tools (compare_search, read_post, list_channels, related_posts, summarize_channel). Здесь фокус на инструментах которые **невозможны без понимания домена** — Telegram-каналы, AI/ML новостная лента, кросс-канальные паттерны.

---

## Результат исследования данных в Qdrant

Прямой анализ 13124 документов в коллекции `news_colbert`:

### Payload fields (фактические)

| Field | Type | Описание | Indexed |
|-------|------|----------|---------|
| `text` | string | Полный текст поста (avg 689 chars, max 1493) | Нет |
| `channel` | string | Username канала (36 уникальных) | **Нет** |
| `channel_id` | int | Telegram ID канала | Нет |
| `message_id` | int | ID сообщения в канале | Нет |
| `date` | string (ISO) | Дата публикации (2025-07-01 → 2026-03-18) | **Нет** |
| `url` | string | Прямая ссылка на пост (t.me/channel/id) | Нет |
| `point_id` | string | Внутренний ID (channel:msg_id[:chunk]) | Нет |

**Критично**: НИ ОДНОГО payload индекса. Фильтрация по channel/date работает через full scan.

### Чего НЕТ в payload (но упоминалось в always_on.md)

Эти поля описаны в документации, но **реально отсутствуют** в Qdrant:
- `author` — нет
- `is_forward` — нет
- `reply_to` — нет
- `links[]` — нет (есть только в тексте как URLs)
- `media_types[]` — нет
- `lang` — нет
- `hash` — нет

### Контент-анализ (выборка 500 постов)

**URLs в текстах:**
- 19% постов содержат ссылки
- Top домены: huggingface.co (6), t.me (5), arxiv.org (4), bloomberg.com (3), techcrunch.com (3), openai.com (2), github.com (1)
- Arxiv papers: ~10 уникальных в выборке 500, экстраполяция ~260 на весь корпус

**Entity mentions (regex по 500 постам):**
- OpenAI: 107, GPT-*: 70, Google: 61, NVIDIA: 41, Anthropic: 41
- Gemini: 27, Meta: 26, Яндекс: 25, HuggingFace: 22, DeepSeek: 22, Qwen: 21
- Llama: 4, Claude: 4, Сбер: 4

**Hashtags:** 10% постов. Top: #news (21), #ai (17), #ml (13), #статья (3), #LLM (2), #Qwen (2)

**Cross-channel ссылки:** t.me/gonzo_ML_podcasts (8), t.me/gonzo_ML (2) — минимальное перекрёстное цитирование

**Вопросы в постах:** 15% постов содержат "?"

**Временное распределение:** ~50-75 постов/месяц в выборке, равномерное

### Vectors

| Vector | Dim | Описание |
|--------|-----|----------|
| `dense_vector` | 1024 | Qwen3-Embedding-0.6B |
| `colbert_vector` | multi-vector, 128-dim per token | jina-colbert-v2 |
| (sparse BM25) | — | Qdrant native sparse |

---

## Что мы уже имеем (из R16)

R16 рекомендовал 5 новых tools (приложен полностью). Краткий итог:

| Tool | Тип | Реальный буст |
|------|-----|--------------|
| `list_channels` | Навигация | Минимальный — utility |
| `read_post` | Deep read | Минимальный — utility |
| `related_posts` | Exploration | Средний — новый паттерн |
| `compare_search` | Сравнение | Высокий — новый тип запроса |
| `summarize_channel` | Дайджест | Высокий — новый тип запроса |

**Проблема**: 3 из 5 — utility tools (не дают нового типа поиска). Только 2 реально расширяют возможности агента. Для "жёсткого буста" нужно больше **search-парадигм**, а не утилит.

---

## Что я хочу от этого ресерча

### 1. Домен-специфичные инструменты для Telegram AI/ML News

Опираясь на фактические данные выше, какие **принципиально новые типы поиска** возможны? Не generic RAG tools, а те что эксплуатируют:

**A) Структуру корпуса (36 каналов × 9 месяцев × 13K постов):**
- Каждый канал = один эксперт/редакция с уникальным фокусом
- Временная ось = эволюция тем и технологий
- Вопрос: какие аналитические операции над этой структурой ценны?

**B) Rich text content (URLs, entities, hashtags IN TEXT):**
- 19% постов со ссылками (arxiv, github, huggingface, news sites)
- Entity mentions extractable at query time (не только при ingest)
- **Мы готовы полностью переделать ingest** — добавить любые поля в payload, создать индексы, перезалить все 13K документов. Не рассматривай текущее отсутствие полей как ограничение. Предлагай идеальную схему payload — мы её реализуем.
- Вопрос: можно ли построить tools которые фильтруют/группируют по extracted features? Какая payload schema идеальна для наших tools?

**C) Кросс-канальные паттерны:**
- Одна новость обсуждается в нескольких каналах с разных перспектив
- Viral content = упоминание одной темы в 5+ каналах за короткий период
- Вопрос: как обнаружить "горячие темы" и "expert consensus"?

**D) Temporal patterns:**
- Новость появляется → первый канал → подхватывают другие → обсуждение угасает
- "Что обсуждали на этой неделе" ≠ simple temporal_search (нужна агрегация, не фильтрация)
- Вопрос: какие temporal-аналитические tools реализуемы на Qdrant scroll + client-side?

### Текущий ingest pipeline (для reference)

Код: `scripts/ingest_telegram.py`, функция `_build_point_docs_flat`. Telethon client → Message objects.

Что **уже собирается** в payload:
```python
payload = {
    "text": text,
    "channel": channel_name,
    "channel_id": int(message.chat_id),
    "message_id": int(message.id),
    "date": message.date.isoformat(),
}
if author:  # sender.first_name + last_name
    payload["author"] = author
payload["url"] = f"https://t.me/{channel_name}/{message.id}"
```

Что **доступно в Telethon Message** но НЕ собирается:
- `message.fwd_from` → is_forward, forwarded_from_channel
- `message.reply_to` → reply_to_msg_id (цепочки обсуждений)
- `message.entities` → URLs, mentions, hashtags, bold/italic (Telegram native entities)
- `message.media` → photo, video, document, audio (media_types)
- Текст: extractable regex-ом → arxiv IDs, GitHub URLs, entity names

Мы **готовы переделать ingest** полностью — добавить все нужные поля. Предлагай идеальную payload schema.

### 2. Обогащение payload при ingest

Какие поля стоит добавить при следующем re-ingest для поддержки новых tools:
- `entities[]` — extracted NER (Natasha/spaCy)
- `urls[]` — extracted URLs
- `url_domains[]` — domains from URLs
- `has_arxiv` (bool), `arxiv_ids[]`
- `has_github` (bool)
- `hashtags[]`
- `text_length` (int)
- `lang` (ru/en detection)
- Другие?

Для каждого: стоимость извлечения, нужна ли модель или хватит regex, и какой tool это enables.

### 3. Pre-computed aggregations

R16 правильно отклонил `trending_search` как real-time tool. Но:
- Какие **pre-computed таблицы/коллекции** можно создать cron job-ом?
- Weekly digest, entity co-occurrence matrix, channel similarity, hot topics?
- Какие tools работали бы поверх pre-computed данных?

### 4. Оценка реального буста

Для каждого предложенного tool:
- Какой % существующих eval вопросов он улучшит?
- Какие **новые типы eval вопросов** он позволит добавить?
- Как это выглядит на собеседовании — что можно рассказать?

---

## Ограничения и возможности

1. **Qdrant-only** — нет SQL, нет графов. Scroll + filter + vector search.
2. **Single-agent, max 5 visible tools** — "Less is More", Qwen3-30B-A3B.
3. **Re-ingest НЕ проблема** — мы готовы полностью перезалить 13K docs с новой payload schema. Ingest pipeline написан нами, полный контроль. Можем добавить любые поля, NER, entity extraction, link parsing, language detection — всё что нужно.
4. **Payload indexes создадим** — keyword, datetime, bool, integer. Текущее отсутствие индексов — oversight, не ограничение.
5. **Client-side aggregation OK** — если <500 мс и результат кэшируем.
6. **GPU доступен при ingest** — Qwen3-Embedding (embeddings), Natasha/spaCy (NER), fasttext (lang detect) — всё можем запустить.
7. **No external APIs** — все данные локальные.

---

## Текущие инструменты (для reference)

7 существующих + 5 из R16 = 12 потенциальных tools.
Из них реально search-парадигм: search, temporal_search, channel_search, compare_search, summarize_channel = **5 search tools**.
Остальные 7 = planning (query_plan), enrichment (read_post, related_posts, rerank), synthesis (compose_context, final_answer), navigation (list_channels).

**Вопрос к ресерчу**: можно ли довести search-парадигмы до 7-8 (с dynamic visibility до 4-5 видимых) за счёт домен-специфичных tools?

---

## Формат ответа

1. **Каталог домен-специфичных tools** (таблица + детальные описания с Qdrant API)
2. **Payload enrichment plan** — что добавить при re-ingest
3. **Pre-computed aggregations** — cron jobs и tools поверх них
4. **Оценка буста** для каждого tool
5. **Implementation priority** — что даёт максимальный эффект при минимальных усилиях
6. **Новые eval вопросы** — 5-10 примеров запросов которые покрываются новыми tools

Конкретные решения для нашего домена. Ссылки на papers где применимо. Без generic advice.
