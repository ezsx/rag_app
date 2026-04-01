# Prompt 32: Pipeline Cleanup — Context Budget, Serialize, Judge Artifact

## Контекст сессии (2026-03-31)

Большая сессия: analytics loop fix → docs sync → agent_service.py decomposition (2397→997 строк) → eval pipeline refactor → обнаружение pipeline бардака.

## Что сделано в этой сессии

### Fixes
1. **Analytics loop fix**: payload indexes (16 на Qdrant :16333), ANALYTICS-COMPLETE → final_answer only, safety net forced completion
2. **Docs sync**: 12+ файлов обновлены (модели, DECs, glossary). Codex нашёл 5 пропусков.
3. **SPEC-RAG-20a**: ingest safety — PAYLOAD_INDEXES unified в store.py (4→16 indexes)
4. **Observability**: trace.input/output работают, usage_details fix, as_type="tool"
5. **Decomposition**: agent_service.py 2397→997 строк, 8 модулей в agent/
6. **Eval refactor**: убран ClaudeJudge API, добавлен LangfuseTraceExporter, Qdrant enrichment для judge artifact
7. **Новые метрики**: evidence_support + retrieval_sufficiency (judge-based, не source_post_ids)
8. **Codex CLI**: установлен, настроен, flow задокументирован

### Критические находки (НЕ пофикшены полностью)

**compose_context budget** = 1200 tokens → 30-47% документов обрезалось. Поднято до 4000, но serialize_tool_payload (8000 chars) может обрезать обратно при 4000 tokens = 16000 chars. Нужно поднять serialize лимит.

**search дублирует тексты** в history — полные тексты 10-20 docs записываются как tool response, потом compose_context пишет те же тексты ещё раз. При refinement — 4 копии.

**Codex нашёл 3 critical** в decomposition: build_final_payload без ctx (3 call sites), _should_attempt_refinement @staticmethod+self, _temporal_guard без step. Первые два пофикшены, третий пофикшен. Тесты (test_agent_service.py) ссылаются на старые методы — CI broken.

**judge artifact** — не показывает что видела модель. Нужен compose_context prompt в export.

## Что делать в следующей сессии

### Обязательно перед работой
- Прочитать SPEC-RAG-20d (`docs/specifications/active/SPEC-RAG-20d-pipeline-cleanup.md`)
- Прочитать architecture docs (обновлены в этой сессии)
- НЕ действовать по памяти — читать код

### Порядок работы
1. **Codex audit в фоне** — запустить промпт 33 (ниже), пусть ищет ещё бардак
2. **serialize_tool_payload** — поднять лимит, убрать обрезку compose prompt
3. **search tool response** — убрать полные тексты из history, оставить IDs
4. **Rebuild + smoke test** — проверить token usage в Langfuse
5. **Judge artifact** — сохранять compose_context prompt, показывать в markdown
6. **Eval 10 Qs** — сравнить retrieval_sufficiency до/после
7. **Merge Codex findings** — дополнить спеку
8. **Full eval 36 Qs** — baseline с новым pipeline

### Файлы для изменения
- `src/services/agent/formatting.py` — serialize_tool_payload лимит
- `src/services/agent/executor.py` — уже поднят compose budget до 4000
- `src/services/tools/search.py` — убрать полные тексты из response
- `scripts/evaluate_agent.py` — сохранять compose prompt в judge artifact
- `src/tests/test_agent_service.py` — обновить после decomposition

### Стек
- LLM: Qwen3.5-35B-A3B (V100, :8080)
- Embedding: pplx-embed-v1-0.6B (RTX 5060 Ti, :8082)
- Reranker: Qwen3-Reranker-0.6B-seq-cls (:8082)
- ColBERT: jina-colbert-v2 (:8082)
- Qdrant: Docker :16333, collection news_colbert_v2, 13777 docs, 16 indexes
- Langfuse: Docker :3100
- API: Docker :8001
- Codex CLI: v0.117.0, ChatGPT auth

### Eval baseline (10 Qs, consensus Claude+Codex)
- factual: 0.55
- useful: 1.4
- evidence_support: 0.70
- retrieval_sufficiency: 0.65
- KTA: 1.0
