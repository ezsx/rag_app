# Review SPEC-RAG-15: Entity Analytics Tools

## Задача

Провести code review спецификации SPEC-RAG-15 (entity_tracker + arxiv_tracker).
Проверить корректность, полноту и согласованность с существующим кодом.

## Файлы для чтения (по приоритету)

### 1. Сама спецификация
- `docs/specifications/active/SPEC-RAG-15-entity-analytics-tools.md` — **ГЛАВНЫЙ ФАЙЛ для review**

### 2. Код — существующие паттерны (сверить что spec им следует)
- `src/services/tools/list_channels.py` — reference Facet API tool (sync bridge pattern)
- `src/services/tools/cross_channel_compare.py` — reference complex tool (multi-vector, groups)
- `src/services/agent_service.py` — ключевые секции:
  - `AGENT_TOOLS` (~строка 63) — текущие 11 tool schemas
  - `AgentState` (~строка 354) — state machine fields
  - `_get_step_tools()` (~строка 1679) — dynamic visibility logic
  - `_apply_action_state()` (~строка 1297) — state updates after tool execution
  - `SYSTEM_PROMPT` (~строка 29) — промпт агента
  - Forced search logic в `stream_agent_response()` (~строка 900+)
- `src/core/deps.py` — tool registration pattern (closure wrappers, ~строка 188)
- `src/services/tools/__init__.py` — текущие импорты

### 3. Данные
- `datasets/entity_dictionary.json` — 91 entity, 6 категорий
- `datasets/eval_golden_v1.json` — текущий golden dataset (25 вопросов)
- `scripts/payload_enrichment.py` — extract_from_text(), entity extraction logic
- `scripts/migrate_collection.py` — payload indexes (16 keyword indexes)

### 4. Research basis
- `docs/research/reports/R17-deep-domain-specific-tools.md` — entity_tracker/arxiv_tracker design
- `docs/research/reports/R16-deep-rag-agent-tools-expansion.md` — visibility, "Less is More"

### 5. Completed spec (формат reference)
- `docs/specifications/completed/SPEC-RAG-13-simple-tools.md` — аналогичная spec для 4 tools

## Чеклист review

### A. Корректность Qdrant API
- [ ] `facet()` вызовы — правильные параметры? `facet_filter` vs `query_filter`?
- [ ] `scroll()` в arxiv_tracker lookup — правильный фильтр? `MatchAny` vs `MatchValue` для массива?
- [ ] `Range` filter для year_week — работает ли string range ("2025-W48" ≤ "2026-W10")?
- [ ] Sync bridge: `hybrid_retriever._run_sync()` — правильный паттерн? Сверить с list_channels.py

### B. State machine
- [ ] `analytics_done` flag — не ломает ли существующие flow (nav, search)?
- [ ] `arxiv_tracker(lookup)` → `search_count++` — корректно ли? Hits попадут в rerank/compose?
- [ ] Forced search bypass — проверить что `analytics_done` в правильном месте
- [ ] ANALYTICS-COMPLETE фаза — нет ли конфликта с NAV-COMPLETE?

### C. Dynamic visibility
- [ ] Keywords — покрывают ли реальные запросы? Нет ли ложных срабатываний?
- [ ] Hard cap 5 tools — не нарушается ли с добавлением analytics?
- [ ] Взаимодействие с existing keyword routing (temporal, channel, compare)

### D. Output format
- [ ] `summary` field — достаточно информативен для LLM?
- [ ] `data` structure — консистентен между modes?
- [ ] `arxiv_tracker(lookup)` hits — совместимы с citation pipeline (id, text, meta, dense_score)?
- [ ] Error handling — все edge cases покрыты? (пустой entity, unknown mode)

### E. Tool schemas для LLM
- [ ] Descriptions ≤50 слов? Достаточно информативны?
- [ ] `mode` enum — values понятны LLM? Не будет путать modes?
- [ ] `entities` (array) vs `entity` (string) — нет ли путаницы?
- [ ] `period_from`/`period_to` format — LLM будет генерировать "2025-W48"?
- [ ] `category` enum — только org/model, а где framework/technique/conference/tool?

### F. Golden dataset
- [ ] q22, q23 updates — корректны?
- [ ] Новые q26-q30 — answerable с текущими данными?
- [ ] Покрывают все modes обоих tools?
- [ ] `forbidden_tools` — логичны?

### G. Интеграция
- [ ] 13 tools total — правильно? Перечислить все
- [ ] Import paths — правильные?
- [ ] ToolRunner timeout (10 sec) — достаточно для facet queries?
- [ ] Совместимость с SSE event contract (step_started/tool_invoked/observation)

### H. Пропущенное / рекомендации
- [ ] Есть ли missing edge cases?
- [ ] Нужны ли unit tests для Facet API mocking?
- [ ] Нужен ли cache для entity_tracker(top)?
- [ ] Entity name matching — case-sensitive? "openai" vs "OpenAI"?

## Формат ответа

Для каждого найденного issue:
```
### [SEVERITY] Краткое описание
**Где**: файл/секция в spec
**Проблема**: что не так
**Fix**: что изменить
```

Severity levels:
- **[CRITICAL]** — приведёт к runtime error или неправильному поведению
- **[IMPORTANT]** — работать будет, но с багами или неоптимально
- **[SUGGESTION]** — улучшение, не обязательно

В конце — общая оценка (готово к имплементации / нужны правки / нужна переработка).

---

## Подключение MCP серверов

Если MCP серверы не подключены или сломаны — вот текущий рабочий конфиг.

### Файл: `~/.codex/config.toml`

```toml
model = "gpt-5.4"
model_reasoning_effort = "high"
personality = "pragmatic"

[windows]
sandbox = "elevated"

# --- MCP Servers ---

[mcp_servers.serena]
command = "uvx"
args = ["--from", "git+https://github.com/oraios/serena", "serena", "start-mcp-server", "--context", "codex"]

[mcp_servers.ast-grep]
command = "uvx"
args = ["--from", "git+https://github.com/ast-grep/ast-grep-mcp", "ast-grep-server"]

[mcp_servers.code-index]
command = "uvx"
args = ["code-index-mcp"]

[mcp_servers.ripgrep]
command = "node"
args = ["C:/Users/scdco/.mcp/ripgrep-win/server.mjs"]
```

### repo-semantic-search — ВАЖНО

repo-semantic-search работает через WSL2 и требует **HTTP transport** для Codex (stdio не поддерживается).

**Вариант 1: Отдельный HTTP-инстанс** (рекомендуется)

Добавить в `config.toml`:
```toml
[mcp_servers.repo-semantic-search]
url = "http://127.0.0.1:8011/mcp"
```

Запустить HTTP-инстанс в WSL2 (отдельный от stdio для Claude Code):
```bash
wsl -d Ubuntu-22.04 -e bash -c '
source /home/ezsx/infinity-env/bin/activate
export PYTHONPATH="/mnt/c/cursor_mcp/repo-semantic-mcp:/mnt/c/cursor_mcp/repo-semantic-mcp/libs"
export SEMANTIC_MCP_QDRANT_URL="http://localhost:6333"
export SEMANTIC_MCP_EMBEDDING_BACKEND="http://localhost:8082"
export SEMANTIC_MCP_TRANSPORT="streamable-http"
export SEMANTIC_MCP_HTTP_HOST="0.0.0.0"
export SEMANTIC_MCP_HTTP_PORT="8011"
export SEMANTIC_MCP_AUTO_INDEX_ON_START="false"
export SEMANTIC_MCP_WATCH_ENABLED="false"
python /mnt/c/cursor_mcp/repo-semantic-mcp/apps/repo-semantic-mcp/main.py 2>/tmp/repo_semantic_http.log
'
```

**Вариант 2: Без repo-semantic-search**

Если не хочется поднимать HTTP-инстанс — просто убрать секцию `[mcp_servers.repo-semantic-search]` из config.toml. Для review задач хватит serena + ripgrep + встроенного чтения файлов.

### Проверка MCP

После настройки конфига запустить Codex и проверить:
```
codex --mcp-status
```

Если серверы не подключаются:
1. Проверить что Docker Desktop запущен (Qdrant на порту 6333/16333)
2. Проверить что gpu_server запущен (порт 8082) — нужен для repo-semantic-search
3. Перезапустить Codex

### Зависимости (порядок запуска)

1. **Docker Desktop** → Qdrant контейнер
2. **gpu_server.py** в WSL2 (порт 8082) — только если нужен repo-semantic-search
3. **repo-semantic-search HTTP** (опционально) → порт 8011
4. **Codex** — подхватит MCP серверы из config.toml
