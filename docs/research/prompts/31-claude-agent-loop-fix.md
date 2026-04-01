# Prompt: Agent Loop Fix — Tool Repeat Guard + Analytics Completion

## Контекст сессии (2026-03-31)

Длинная сессия: embedding/reranker swap (pplx-embed + Qwen3-Reranker), reingest 13K docs, English system prompt, tool repeat guard. Контекст на пределе.

## Что сделано

1. **Embedding swap**: Qwen3-Embedding-0.6B → pplx-embed-v1-0.6B (bf16, +7 MTEB pts)
2. **Reranker swap**: bge-reranker-v2-m3 → Qwen3-Reranker-0.6B-seq-cls (chat template, +5-8 rerank score)
3. **ColBERT**: jina-colbert-v2 (без изменений)
4. **Reingest**: 13777 docs, все 3 вектора inline, bf16 для embedding (fp16 давал NaN на длинных текстах)
5. **English system prompt**: чёткие path routing (A: retrieval, B: analytics, C: navigation, D: refusal)
6. **Tool repeat guard**: блокирует повторные вызовы analytics/navigation tools
7. **compose.dev.yml**: direct :8082 вместо relay :18082 (mirrored отключён)
8. **gpu_server.py**: новые model paths, bf16 embedding, reranker chat template

## ТЕКУЩАЯ ПРОБЛЕМА (не решена)

**Tool repeat guard ломает agent loop для analytics вопросов.**

### Симптомы
- q22, q23, q26, q28, q29, q30: analytics tool (entity_tracker, arxiv_tracker) вызван 1 раз → модель пытается вызвать его ещё → guard блокирует и вставляет tool response "Tool already called. Use final_answer" → модель НЕ вызывает final_answer → зацикливается на 8 шагов → "Не удалось завершить анализ"
- q21: temporal_search ×2 (guard блокирует второй) → тоже не завершается

### Root cause
Guard вставляет fake tool response `"Tool X already called. Use final_answer."`, но Qwen3.5 не следует этой инструкции в tool response. Модель видит tool response и пытается продолжить работу вместо вызова final_answer.

### Предыдущая проблема (РЕШЕНА)
Без fake tool response → `"Cannot have 2 or more assistant messages at the end of the list"` 400 error от llama-server. Guard с `continue` без tool response ломал message sequence.

### Что НЕ работает
1. `continue` без tool response → 400 error (2 assistant messages подряд)
2. Fake tool response "use final_answer" → модель игнорирует, зацикливается
3. Prompt "NEVER call same tool twice" → модель игнорирует на русском, частично следует на английском

### Варианты фикса (не реализованы)
1. **Forced final_answer после analytics**: если analytics tool вызван и guard блокирует повтор → принудительно генерировать final_answer из последнего tool observation (code-level, не prompt)
2. **analytics_done → skip LLM**: после entity_tracker/arxiv_tracker/hot_topics → сразу формировать final_answer без ещё одного LLM call (как navigation short-circuit)
3. **Убрать guard, решить через prompt**: вернуть guard только для related_posts, для analytics положиться на prompt "use final_answer immediately"
4. **Лимит шагов после analytics**: если analytics_done и step > analytics_step+1 → force finish

## Рабочий стек

- **Embedding**: pplx-embed-v1-0.6B (bf16), path: `/mnt/c/llms/models/pplx-embed-v1-0.6B`
- **Reranker**: Qwen3-Reranker-0.6B-seq-cls, path: `/mnt/c/llms/models/Qwen3-Reranker-0.6B-seq-cls`
- **ColBERT**: jina-colbert-v2, path: `/home/tei-models/jina-colbert-v2`
- **LLM**: Qwen3.5-35B-A3B Q4_K_M на V100
- **Qdrant**: 13777 docs, dense + sparse + ColBERT
- **Langfuse**: localhost:3100, traces с parent-child spans

## Eval метрики (текущие, с guard проблемой)

| Metric | Value | Notes |
|--------|-------|-------|
| KTA | 0.909 | Routing правильный |
| Factual (working Qs) | ~0.90 | Где не упал в loop |
| 400 errors | 0 | Пофикшено |
| Loop failures | 7/36 | analytics tools зацикливаются |
| Latency | 32.3s mean | |

## Eval без loop проблемы (потенциал)

На вопросах где агент НЕ зацикливается — **best metrics ever**: factual 0.907, useful 1.741. Embedding/reranker стек работает отлично. Нужно только пофиксить agent loop completion.

## Файлы для чтения

1. `src/services/agent_service.py` — agent loop, tool repeat guard (строка ~920), _get_step_tools (строка ~2153)
2. `src/services/agent_service.py` — SYSTEM_PROMPT (English, строка ~80)
3. `datasets/tool_keywords.json` — keyword routing
4. `results/eval_new_stack_v3/` — последний eval run с failures
5. `docs/research/reports/R13-deep-tool-router-architecture.md` — fallback patterns

## Ключевые design decisions

- DEC-0039: Qwen3.5-35B-A3B
- DEC-0040: Langfuse v3
- DEC-0041: WSL mirrored отключён по умолчанию
- English system prompt (не зафиксировано в DEC, нужно добавить)

## Pending

1. **ФИКС agent loop** — analytics completion без зацикливания
2. Full eval на пофикшенном стеке → consensus judge
3. Коммит всех изменений (стек, prompt, guard, gpu_server, compose)
4. CRAG-lite spec (после стабилизации)
