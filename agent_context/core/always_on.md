# Always-On Guardrails

Этот файл загружается всегда. Короткий: только правила, которые нельзя нарушать.

## Архитектура
- `docs/ai/project_brief.md` — эталонный overview проекта. Читать при неясности в архитектуре.
- `docs/ai/agent_technical_spec.md` — детальная спецификация ReAct агента.
- LLM, ретриверы и сервисы создаются лениво через `lru_cache` в `src/core/deps.py`.
  Смена настроек требует явного `cache_clear()` через `settings.update_*()`.

## Код и модели
- Основной LLM: **Qwen3-30B-A3B GGUF** (V100 SXM2 32GB, llama-server.exe на Windows хосте).
- Embedding: **Qwen3-Embedding-0.6B** через TEI HTTP → WSL2 native (RTX 5060 Ti, порт 8082).
- Reranker: **BAAI/bge-m3** (XLMRoberta seq-cls) через gpu_server.py → WSL2 native (RTX 5060 Ti, порт 8082).
  **Временная мера**: целевой реранкер — bge-reranker-v2-m3 (dedicated cross-encoder, +10 nDCG).
- Хранилище (Phase 1): **Qdrant** (dense + sparse named vectors, **weighted RRF** BM25 3:1).
- **Docker GPU blocker**: RTX 5060 Ti недоступна в Docker Desktop (TCC V100 блокирует NVML).
  Embedding/Reranker запускаются нативно через gpu_server.py в Ubuntu WSL2, Docker обращается через `host.docker.internal`.
  Детали: DEC-0024 в `docs/architecture/11-decisions/decision-log.md`.

## ReAct агент
- Оркестрация: native function calling через `/v1/chat/completions`, без regex-парсинга Thought/Action.
- LLM tools schema: `query_plan → search → rerank → compose_context → final_answer`.
- **Dynamic tools**: `final_answer` скрыт до выполнения `search` — LLM не может пропустить поиск.
- **Forced search**: если LLM не вызывает tools, принудительный search с оригинальным запросом.
- **Original query injection**: оригинальный запрос всегда добавляется в subqueries для BM25 keyword match.
- `verify` и `fetch_docs` вызываются системно внутри `AgentService`, не через schema для модели.
- Retrieval-пайплайн: `query_plan → search (BM25 top-100 + dense top-20 → RRF 3:1) → rerank → compose_context`.
- Coverage threshold: **0.65**; max refinements: **2** (DEC-0019; не менять без ресерча).
- `agent_service.py` — единственный владелец состояния шага; не дублировать логику снаружи.
- SSE стриминг через `/v1/agent/stream` — не ломать контракт событий (thought/tool_invoked/observation/citations/final).
- **Recall@5 = 0.70** на quick dataset (10 вопросов). Roadmap к 0.80+: `docs/ai/planning/retrieval_improvement_playbook.md`.

## Deploy и запуск
- **ВАЖНО: Docker GPU НЕ ИСПОЛЬЗУЕТСЯ.** V100 TCC отравляет NVML в WSL2 →
  nvidia-container-cli крашится. Flash Attention не работает на sm_120 (RTX 5060 Ti).
  Подробности: `docs/research/rag-stack/reports/R10-gpu-docker-wsl2-troubleshooting.md`.
- **Порядок запуска:**
  1. llama-server.exe на Windows хосте (V100, порт 8080):
     `--jinja --reasoning-budget 0 --cache-type-k q8_0 --cache-type-v q8_0 -c 16384 --parallel 2`
  2. gpu_server.py нативно в WSL2 (RTX 5060 Ti, порт 8082):
     `source /home/ezsx/infinity-env/bin/activate && CUDA_VISIBLE_DEVICES=0 python scripts/gpu_server.py`
     Embedding (Qwen3-Embedding-0.6B) + Reranker (BGE-M3) в одном процессе.
     PyTorch cu128 + cuBLAS, без Docker, без Flash Attention.
  3. Docker Desktop (CPU only): `docker compose -f deploy/compose/compose.dev.yml up`
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
