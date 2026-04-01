# Prompt 33: Codex Pipeline Audit — Find Hidden Problems

## Задача

Провести глубокий аудит RAG agent pipeline на предмет скрытых проблем с обработкой контекста, обрезкой данных, дублированием и потерей информации.

## Контекст

При ревью eval результатов обнаружили что 30-47% найденных документов обрезалось перед тем как попасть к LLM. Модель отвечала на вопросы видя только половину контекста. Это маскировало качество retrieval и ухудшало ответы.

## Что уже нашли (НЕ искать повторно)

1. **compose_context budget слишком мал** — 1200 tokens, обрезал 30-47% документов. Поднято до 4000.
2. **serialize_tool_payload обрезает prompt** — лимит 8000 chars, при budget 4000 tokens (16000 chars) обрежет вдвое.
3. **search дублирует тексты в history** — полные тексты 10-20 docs + те же тексты в compose_context = дубликаты.
4. **rerank scores не используются** — compose_context получает все search_hits, не top-N от rerank.
5. **три слоя обрезки** — compose_context, serialize, trim_messages — не координированы.
6. **coverage threshold** не откалиброван под pplx-embed (калибровался на Qwen3-Embedding).

## Что искать (scope аудита)

### A. Потеря контекста
- Есть ли ещё места где тексты документов обрезаются или теряются?
- Что происходит с ColBERT MaxSim reranked order — сохраняется ли после compose?
- Fetch_docs (системный tool для догрузки текстов) — когда вызывается, работает ли?

### B. Дублирование данных в messages
- Сколько раз одни и те же тексты попадают в chat history?
- Какой реальный размер messages на финальном шаге (до trim)?
- Что trim_messages обрезает — ценные docs или шум?

### C. Token budget accounting
- Посчитать реальный бюджет: system prompt + tools schema + history + compose prompt + output
- Какой максимальный input_tokens мы видим в Langfuse traces?
- Есть ли risk context overflow?

### D. Другие anomalies
- query_plan subqueries — все ли используются или часть теряется?
- Original query injection — корректно ли добавляется?
- Refinement — что именно повторяется и зачем?
- QAService fallback — это вообще нужно?
- Temporal guard — корректно ли работает проверка дат?

## Файлы для чтения

1. `src/services/agent_service.py` — orchestrator (997 строк после decomposition)
2. `src/services/agent/executor.py` — execute_action, normalize_tool_params
3. `src/services/agent/formatting.py` — serialize_tool_payload, trim_messages, format_observation
4. `src/services/tools/compose_context.py` — формирование контекста из документов
5. `src/services/tools/search.py` — search tool, round-robin merge
6. `src/adapters/search/hybrid_retriever.py` — Qdrant query, RRF, ColBERT, channel dedup
7. `src/services/query_planner_service.py` — subquery generation

## Output

Пронумерованный список findings:
- Severity: P0 / P1 / P2 / P3
- Файл и строка
- Что происходит
- Почему это проблема
- Предложенный fix
