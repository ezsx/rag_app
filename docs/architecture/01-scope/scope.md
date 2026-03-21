## Scope

### Что такое rag_app

**rag_app** — FastAPI-платформа RAG + ReAct агент для поиска и агрегации новостей
из Telegram-каналов.

Пользователь задаёт вопрос на русском → ReAct агент с 5 LLM-инструментами ищет по
проиндексированным 36 Telegram-каналам (13K+ документов) → отвечает с цитатами через SSE стриминг.

**Целевой пользователь**: один пользователь (автор проекта) — для обучения и
демонстрации навыков в applied LLM/AI.

### Что входит в scope

- **Ingest**: загрузка Telegram-каналов → Qdrant (dense + sparse + ColBERT vectors)
- **Retrieval**: гибридный поиск (BM25+Dense → weighted RRF 3:1 → ColBERT MaxSim → cross-encoder rerank → channel dedup)
- **ReAct агент**: native function calling, 5 LLM tools, auto-refinement, grounding
- **API**: FastAPI с JWT auth, `/v1/agent/stream` (SSE), Web UI
- **Evaluation**: agent eval (через LLM) + retrieval eval (прямые Qdrant queries)

### Что НЕ входит в scope

- Multi-user / multi-tenant
- Горизонтальное масштабирование
- Production SLA / uptime guarantees
- Автоматическое переключение каналов / live-инgest

### Ключевые цели проекта

1. Рабочий end-to-end RAG агент на локальных LLM (self-hosted, no managed APIs)
2. Портфолио для Applied LLM Engineer позиций
3. Полный цикл: ingest → retrieval → agent → evaluation с tracked experiments
4. Adaptive retrieval — ключевое отличие от фреймворков (LlamaIndex/LangChain)
