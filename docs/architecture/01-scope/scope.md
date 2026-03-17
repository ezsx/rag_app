## Scope

### Что такое rag_app

**rag_app** — FastAPI-платформа RAG + ReAct агент для поиска и агрегации новостей
из Telegram-каналов.

Пользователь задаёт вопрос на русском → ReAct агент с 7 инструментами ищет по
проиндексированным Telegram-каналам → отвечает с цитатами через SSE стриминг.

**Целевой пользователь**: один пользователь (автор проекта) — для обучения и
демонстрации навыков в applied LLM/AI.

### Что входит в scope

- **Ingest**: загрузка Telegram-каналов → Qdrant (dense + sparse vectors)
- **Retrieval**: гибридный поиск (Qdrant native sparse + dense + RRF + MMR + BGE reranker)
- **ReAct агент**: 7-шаговый цикл с SSE стримингом
- **API**: FastAPI с JWT auth, `/v1/agent/stream`, `/v1/qa/stream`
- **Evaluation**: CLI скрипт для оценки качества агента

### Что НЕ входит в scope

- Multi-user / multi-tenant
- Горизонтальное масштабирование
- Production SLA / uptime guarantees
- Автоматическое переключение каналов / live-инgest

### Ключевые цели проекта

1. Рабочий end-to-end RAG агент на локальных LLM
2. Источник для изучения applied LLM паттернов (собеседования)
3. Полный цикл: ingest → retrieval → agent → evaluation
