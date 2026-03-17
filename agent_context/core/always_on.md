# Always-On Guardrails

Этот файл загружается всегда. Короткий: только правила, которые нельзя нарушать.

## Архитектура
- `docs/ai/project_brief.md` — эталонный overview проекта. Читать при неясности в архитектуре.
- `docs/ai/agent_technical_spec.md` — детальная спецификация ReAct агента.
- LLM, ретриверы и сервисы создаются лениво через `lru_cache` в `src/core/deps.py`.
  Смена настроек требует явного `cache_clear()` через `settings.update_*()`.

## Код и модели
- Основной LLM: **Qwen3-8B GGUF** (V100 SXM2 32GB, llama-server.exe на Windows хосте).
- Embedding: **multilingual-e5-large** через TEI HTTP → WSL2 native (RTX 5060 Ti, порт 8082).
- Reranker: **bge-reranker-v2-m3** через TEI HTTP → WSL2 native (RTX 5060 Ti, порт 8083).
- Хранилище (Phase 1): **Qdrant** (dense + sparse named vectors, native RRF+MMR).
- **Docker GPU blocker**: RTX 5060 Ti недоступна в Docker Desktop (TCC V100 блокирует NVML).
  Embedding/Reranker запускаются нативно в Ubuntu WSL2, Docker обращается через `host.docker.internal`.
  Детали: DEC-0024 в `docs/architecture/11-decisions/decision-log.md`.

## ReAct агент
- Цикл: router_select → query_plan → search → rerank → compose_context → verify → final_answer.
- Coverage threshold: **0.65**; max refinements: **2** (DEC-0019; не менять без ресерча).
- `agent_service.py` — единственный владелец состояния шага; не дублировать логику снаружи.
- SSE стриминг через `/v1/agent/stream` — не ломать контракт событий (thought/tool_invoked/observation/citations/final).

## Deploy и запуск
- **Перед** `docker compose up` — запустить в Ubuntu WSL2:
  - TEI embedding: `docker run -d --gpus all -p 8082:80 ghcr.io/huggingface/text-embeddings-inference:1.9 --model-id intfloat/multilingual-e5-large`
  - TEI reranker: `docker run -d --gpus all -p 8083:80 ghcr.io/huggingface/text-embeddings-inference:1.9 --model-id BAAI/bge-reranker-v2-m3`
  - llama-server.exe: на Windows хосте (V100), порт 8080
- Docker-сервисы (CPU only): `docker compose -f deploy/compose/compose.dev.yml up`
- Ingest: `docker compose -f deploy/compose/compose.dev.yml run --rm ingest --channel @name --since YYYY-MM-DD --until YYYY-MM-DD`
- `.env` в корне репозитория — не коммитить, не логировать plaintext-секреты.

## Тесты
- Основные тесты в `src/tests/`.
- `pytest` в контейнере: `docker compose -f deploy/compose/compose.test.yml run --rm test`.
- Evaluation скрипт: `python scripts/evaluate_agent.py` (требует запущенного API).

## Безопасность
- Не логировать JWT-токены, API-ключи, промпты с PII.
- `SecurityManager` и `sanitize_for_logging` — использовать для всего внешнего ввода.
- Не использовать destructive git-команды без явного запроса.

## Стиль кода
- Python docstring на русском языке, если тело функции длиннее ~5 строк.
- Комментарии на русском для нетривиальной логики (особенно в ReAct цикле, RRF fusion).
