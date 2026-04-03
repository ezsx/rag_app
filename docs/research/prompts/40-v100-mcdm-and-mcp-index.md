# Prompt 40: V100 MCDM upgrade + MCP index persistence fix

## Контекст сессии

### Что сделано (SPEC-RAG-28 — code quality 9/10)
Все 7 gaps выполнены, 5 коммитов запушены:
- Gap 1+5: +119 тестов (74→193), SecurityManager comment
- Gap 2+3: MMR dead code -309 строк, HybridRetriever shims удалены
- Gap 6: Exception audit — 9 narrowed, 80 annotated
- Gap 4: Docstrings RU→EN, 20 файлов, legacy refs убраны
- Gap 7: mypy strict check_untyped_defs=true, 0 errors

CI green: ruff + pytest (193) + mypy (strict). Всё запушено.

### Проблема с MCP repo-semantic-search

**Симптомы**: индекс пустой при каждом включении компа, watcher не видит изменения, нужен ручной rebuild.

**Root cause** (из логов Docker):
1. MCP использует **отдельный Qdrant** контейнер (`repo-semantic-qdrant` на порту 6333)
2. При старте `purge_stale_collections()` дропает коллекции если имя изменилось (profile/model/schema)
3. `AUTO_INDEX_ON_START` пытается rebuild, но gpu_server (embedding) ещё не поднят → fail → пустой индекс
4. Watcher показывал `false` в status API, но логи показывают что он **работает** (`semantic_watcher_reindexed`)

**Конфиг**: `C:\cursor_mcp\repo-semantic-mcp\deploy\repo-semantic-search\docker-compose.repo-semantic-search.yml`
**Env**: передаётся через Docker env (inspect показывает `WATCH_ENABLED=1`, `AUTO_INDEX_ON_START=1`, `PROFILE=gpu_pplx`)
**Volume**: `qdrant_repo_semantic_data` — персистентный, данные не теряются. Проблема в `purge_stale_collections()`.

### Решение: V100 TCC→MCDM (R595)

Пользователь решил перейти на MCDM вместо workaround'ов. Это решает **всё**:
- Docker `--gpus all` работает (NVML не отравлен)
- gpu_server.py переносится в Docker compose
- MCP embedding в Docker (не зависит от ручного запуска gpu_server)
- Один `docker compose up` вместо трёх ручных процессов

**Полный гайд**: `fix_tcc.md` в корне репо (gitignored). Содержит 6 частей:
1. Подготовка (backup, экспорт реестра)
2. INF-мод (добавить DEV_1DB5 в DC 595.97 nv_dispwi.inf)
3. Установка (GeForce 595.XX clean install → DC 595.97 modified INF → pnputil)
4. MCDM проверка (nvidia-smi -dm 0, WSL2, Docker GPU)
5. Откат (точка восстановления или ручная переустановка R580)
6. Post-success (gpu_server → Docker compose, MCP на main Qdrant)

**Текущие драйверы**: R580 (GeForce 581.80 + DC 581.42)
**Целевые**: R595 (GeForce 595.XX + DC 595.97 modified)

### Риски
- Модифицированный INF не подписан → testsigning или F7 boot
- V100 SXM2 на 39com адаптере с R595 не тестировалось публично
- MCDM overhead ~5-15% submission latency (для LLM inference некритично)
- Откат: точка восстановления (Checkpoint-Computer)

## Задачи для следующей сессии

### Задача 1: V100 MCDM upgrade (требует физический доступ)

Это делает пользователь руками по `fix_tcc.md`. Claude помогает с:
- Проверкой после каждого шага (`nvidia-smi -L`, WSL2, Docker GPU)
- Troubleshooting если Code 43 или NVML crash
- Модификацией compose файлов после успеха

### Задача 2: Post-MCDM infrastructure update (если MCDM успешен)

1. **gpu_server.py → Docker service**: добавить GPU service в `deploy/compose/compose.dev.yml` с `--gpus device=0` (RTX 5060 Ti)
2. **MCP на main Qdrant**: переключить `SEMANTIC_MCP_QDRANT_URL` на `http://qdrant:6333` (internal) или `http://host.docker.internal:16333`
3. **Убрать отдельный MCP Qdrant**: repo-semantic-qdrant контейнер больше не нужен
4. **Обновить порядок запуска**: llama-server (V100, MCDM) + `docker compose up` (всё остальное)
5. **Обновить CLAUDE.md** и `always_on.md` с новой процедурой

### Задача 3 (если MCDM не сработал): MCP index fix workaround

Без MCDM — фиксим MCP отдельно:
1. Переключить MCP на проектный Qdrant (порт 16333) — убрать отдельный контейнер
2. Фиксить `purge_stale_collections()` — не дропать если profile изменился
3. Добавить retry для auto_index_on_start с ожиданием embedding service

## Hardware reference

- **V100 SXM2 32GB**: Tesla, PCI DEV_1DB5, 39com PCIe адаптер, TCC mode (R580)
- **RTX 5060 Ti**: GeForce, WDDM, WSL2 native (порт 8082 через gpu_server.py)
- **Текущий запуск**: 3 ручных процесса (llama-server Windows + gpu_server WSL2 + Docker CPU)
- **Целевой запуск**: llama-server Windows + `docker compose up` (GPU services + CPU services)
