# Review prompt: SPEC-RAG-12 и SPEC-RAG-13

## Задача

Ты — senior reviewer. Проверь две спецификации для RAG-проекта. Найди:
1. **Логические дыры** — что упущено, что сломается
2. **Конфликты** между spec-ами и существующим кодом
3. **Избыточность** — что можно убрать без потери value
4. **Риски** — что может пойти не так при реализации
5. **Альтернативы** — есть ли решения проще/лучше

НЕ хвали. НЕ пересказывай. Только критика и конкретные рекомендации.

---

## Контекст проекта

**rag_app** — self-hosted RAG система. Поисковик по 36 AI/ML Telegram-каналам (13K документов). ReAct агент с native function calling на Qwen3-30B-A3B (MoE, 3B active params, V100).

**Стек**: Qdrant (dense 1024 + sparse BM25 + ColBERT 128-dim, weighted RRF 3:1), Qwen3-Embedding-0.6B, bge-reranker-v2-m3, jina-colbert-v2. Docker + WSL2. Оркестрация через /v1/chat/completions.

**Текущие tools агента** (7 LLM-visible):
- query_plan, search, temporal_search, channel_search, rerank, compose_context, final_answer
- Dynamic visibility: max 4-5 видимых, скрытие по фазе + query signals (regex)
- Constraint: max 5 tools видимых (Qwen3-30B деградирует при >5-7)

**Текущий payload в Qdrant** (7 полей, НИ ОДНОГО индекса):
- text, channel, channel_id, message_id, date (ISO string), url, point_id
- Фильтрация = full scan 13K docs

**Что мы планируем** (2 spec):
- SPEC-RAG-12: обогатить payload (16+ новых полей, entity NER regex, indexes), re-ingest
- SPEC-RAG-13: добавить 5 tools (list_channels, read_post, related_posts, cross_channel_compare, summarize_channel), обновить visibility и system prompt

**Research base**:
- R15: Яндекс конференция — "Less is More", dynamic tool подкладка, FC bottleneck = размер descriptions
- R16: Generic RAG tools — 5 tools рекомендованы, 2 отклонены (trending, author_search)
- R17: Domain-specific tools — entity_tracker, hot_topics, cross_channel_compare, arxiv_tracker, payload enrichment plan

---

## Файлы для review

Приложены:
1. `SPEC-RAG-12-payload-enrichment.md` — полная spec
2. `SPEC-RAG-13-simple-tools.md` — полная spec
3. `R16-deep-rag-agent-tools-expansion.md` — research (generic tools)
4. `R17-deep-domain-specific-tools.md` — research (domain-specific tools)

---

## Конкретные вопросы

1. **Re-ingest risk**: мы пересоздаём коллекцию с 13K docs. Vectors (dense, sparse, ColBERT) уже есть. Нужно ли пересчитывать embeddings или можно обновить только payload? Какой Qdrant API для этого?

2. **Entity dictionary maintenance**: словарь ~500 entities с aliases. Как не превратить это в бесконечный maintenance? Достаточно ли regex tier-1 (без GLiNER)?

3. **Facet API**: spec полагается на Qdrant facet для list_channels и entity counting. Qdrant facet требует keyword index. Есть ли ограничения facet по массивам (entities[])?

4. **query_points_groups**: cross_channel_compare использует group_by="channel". Работает ли это с prefetch + fusion (RRF) или только с single vector?

5. **Dynamic visibility cap at 5**: текущая реализация — priority sort + truncate. Это правильный подход или есть лучше? Что если LLM нужен tool который был отрезан?

6. **System prompt size**: с 12 tools описание всех в system prompt → сколько токенов? Не превышаем ли мы бюджет 2000 токенов на descriptions?

7. **Regression risk**: добавление 5 tools может ухудшить selection accuracy на существующих запросах. Как mitigation кроме smoke test?

8. **Порядок реализации**: правильно ли делать RAG-12 (re-ingest) перед RAG-13 (tools)? Или можно параллельно?

9. **Что мы упустили?** Есть ли очевидные проблемы которые мы не видим из-за tunnel vision?

---

## Формат ответа

Структурированный review:
1. **Critical issues** (блокеры реализации)
2. **Major concerns** (серьёзные проблемы, нужно решить)
3. **Minor suggestions** (nice to have)
4. **Ответы на конкретные вопросы** (1-9)
5. **Вердикт**: ready to implement / needs revision / needs more research
