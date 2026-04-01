# Agent Module — ReAct Агент

## Ключевые файлы

- `src/services/agent_service.py` — основной класс `AgentService`, system prompt, AGENT_TOOLS, dynamic visibility
  - **RequestContext** (ContextVar) — per-request state isolation (SPEC-RAG-17 FIX-01)
- `src/services/tools/tool_runner.py` — `ToolRunner` реестр + запуск с таймаутом
- `src/services/tools/` — 15 LLM-visible инструментов + 2 системных
- `src/schemas/agent.py` — схемы: `AgentRequest`, `AgentStepEvent`, `ToolRequest`, `AgentAction`
- `src/core/deps.py` — DI: `get_agent_service`, `get_tool_runner`, wrapper-функции для всех tools
- `src/api/v1/endpoints/agent.py` — SSE endpoint `/v1/agent/stream`

## Цикл ReAct (native function calling)

```
Инициализация (AgentState: coverage=0.0, refinement_count=0, search_count=0, navigation_answered=False, analytics_done=False)
    ↓
Цикл по шагам (до max_steps=8):
    _get_step_tools(agent_state)  — phase-based visibility (max 5 tools)
    yield step_started {step, visible_tools, request_id}
    LLM.chat_completion(messages, tools=step_tools)
    parse(content + tool_calls)
    emit thought event (если есть content)

    Guard: forced search bypass
      - если нет tool_calls И search_count==0 И НЕ navigation_answered И НЕ analytics_done И НЕ refusal → forced search

    Для каждого tool_call:
      _normalize_tool_params() — temporal/channel → search с filters
      temporal guard: если date_from/date_to вне корпуса → refusal
      _execute_action() → ToolRunner.run()
      _apply_action_state() — обновляет search_count, hits, coverage, navigation_answered, analytics_done
      emit tool_invoked, observation

      if compose_context → check coverage, maybe refinement
      if analytics_done and hits не нужны → final_answer доступен без search
    ↓
Fallback (max_steps exceeded): error answer
```

## 15 LLM tools + 2 системных (SPEC-RAG-13 + SPEC-RAG-15 + SPEC-RAG-16)

### Pre-search phase
| Инструмент | Файл | Назначение |
|-----------|------|-----------|
| `query_plan` | `tools/query_plan.py` | Декомпозиция запроса на 3-5 подзапросов |
| `search` | `tools/search.py` | Широкий гибридный поиск (BM25+dense→RRF 3:1→ColBERT) |
| `temporal_search` | `tools/search.py` | Поиск с date filter (маппится на search + filters) |
| `channel_search` | `tools/search.py` | Поиск в конкретном канале (маппится на search + filters) |
| `cross_channel_compare` | `tools/cross_channel_compare.py` | Qdrant query_points_groups по каналам |
| `summarize_channel` | `tools/summarize_channel.py` | Qdrant scroll + dedup, temporal window от latest post |
| `list_channels` | `tools/list_channels.py` | Qdrant Facet API на channel field |

### Post-search phase
| Инструмент | Файл | Назначение |
|-----------|------|-----------|
| `rerank` | `tools/rerank.py` | Qwen3-Reranker-0.6B-seq-cls cross-encoder |
| `compose_context` | `tools/compose_context.py` | Сборка контекста с цитатами, coverage |
| `final_answer` | `tools/final_answer.py` | Финальный payload с sources |
| `related_posts` | `tools/related_posts.py` | Qdrant RecommendQuery — похожие посты |
| `entity_tracker` | `tools/entity_tracker.py` | Facet analytics по сущностям: top, timeline, compare, co_occurrence |
| `arxiv_tracker` | `tools/arxiv_tracker.py` | Facet/lookup по arxiv-статьям: top papers, кто обсуждал |
| `hot_topics` | `tools/hot_topics.py` | Горячие темы за неделю из коллекции `weekly_digests` |
| `channel_expertise` | `tools/channel_expertise.py` | Профиль экспертизы канала из коллекции `channel_profiles` |

### Системные (не в LLM schema)
| Инструмент | Файл | Назначение |
|-----------|------|-----------|
| `fetch_docs` | `tools/fetch_docs.py` | Догрузка полных текстов по id |
| `verify` | `tools/verify.py` | Верификация финального ответа |

## Dynamic Tool Visibility (phase-based)

```python
if nav_done and not search_done:
    # NAV-COMPLETE: only final_answer
    visible = {"final_answer"}
elif analytics_done and not search_done:
    # ANALYTICS-COMPLETE: можно сразу отвечать или продолжить с search
    visible = {"final_answer", "search", "entity_tracker", "arxiv_tracker"}
elif search_done:
    # POST-SEARCH: synthesis tools
    visible = {"rerank", "compose_context", "final_answer", "related_posts"}
else:
    # PRE-SEARCH: query_plan + search + signal-based + keyword-routed specialized
    visible = {"query_plan", "search"}
    # + temporal_search (если есть даты в query)
    # + channel_search, summarize_channel (если есть канал)
    # + cross_channel_compare (если "сравни", "vs")
    # + list_channels (если "какие каналы", "сколько постов")
    # + entity_tracker, arxiv_tracker (если keywords из datasets/tool_keywords.json)
    # + hot_topics (если "тренды", "горячие темы", "дайджест")
    # + channel_expertise (если "экспертиза канала", "о чём пишет")
    # Hard cap: max 5, убираем по eviction priority
```

## Refusal Policy (P1, 2026-03-24)

System prompt содержит explicit refusal rules:
- Даты вне июля 2025 — марта 2026 → отказ без поиска
- Несуществующая сущность не найдена → "нет в базе", без подмены
- Temporal guard в `_execute_action`: если date_from/date_to полностью вне корпуса → empty hits
- Forced search bypass: если LLM content содержит refusal markers → не форсить search

## SSE события (контракт `/v1/agent/stream`)

```
step_started   — {step, visible_tools, request_id, max_steps, query}
thought        — мысль агента (text)
tool_invoked   — {tool, input, step}
observation    — результат инструмента {content, success, took_ms}
citations      — из compose_context {citations, coverage}
final          — финальный ответ {answer, citations, coverage, request_id}
error          — при ошибке
```

## Вспомогательные коллекции Qdrant (SPEC-RAG-16)

- `weekly_digests` — горячие темы по неделям (используется `hot_topics` tool)
- `channel_profiles` — профили экспертизы каналов (используется `channel_expertise` tool)

## Настройки (settings.py)

```
agent_max_steps       = 15
agent_default_steps   = 8
coverage_threshold    = 0.75     # LANCER nugget coverage (3/4 nuggets)
max_refinements       = 1        # targeted по uncovered nuggets
agent_tool_temp       = 0.2
agent_final_max_tokens = 512
```

## Eval (SPEC-RAG-14, Phase 3.3)

- Golden dataset: `datasets/eval_golden_v1.json` (25 Qs, 6 categories)
- Eval script: `scripts/evaluate_agent.py` — tool tracking, failure attribution, LLM judge
- Key Tool Accuracy: **0.955** | Strict Recall@5: ~0.43 | Manual judge factual: **0.52**
- Подробности: `docs/specifications/active/SPEC-RAG-14-evaluation-pipeline.md`
