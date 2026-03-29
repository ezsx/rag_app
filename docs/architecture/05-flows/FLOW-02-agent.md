## FLOW-02: Agent Request → SSE Stream

### Problem
Пользователь задаёт вопрос. Агент должен найти релевантные данные в Telegram-архиве
и дать ответ с цитатами. Весь процесс транслируется через SSE для наблюдаемости.

### Contract
```
POST /v1/agent/stream
Authorization: Bearer <jwt>
Content-Type: application/json

{"query": "string", "max_steps": 8, "planner": true}

→ text/event-stream
  event: thought       {"thought": "...", "step": N, "request_id": "..."}
  event: tool_invoked  {"tool": "...", "input": {...}, "step": N, "request_id": "..."}
  event: observation   {"tool": "...", "output": {...}, "ok": bool, "step": N}
  event: citations     {"citations": [...], "coverage": float, "step": N}
  event: final         {"answer": "...", "step": N, "total_steps": N, "request_id": "..."}
```

### Actors
- **Client** — browser (Web UI) / evaluate_agent.py
- **FastAPI** — HTTP layer, SSE stream
- **AgentService** — ReAct loop owner, native function calling, adaptive tool selection
- **QuerySignals** — rule-based pre-validator (<1ms): extracts dates, channels, entities
- **ToolRunner** — tool registry + timeout execution
- **LLM** — Qwen3-30B-A3B GGUF (llama-server, V100; `--jinja --reasoning-budget 0`). Выбирает из 15 LLM tools (search/temporal_search/channel_search/entity_tracker/arxiv_tracker/hot_topics/channel_expertise/...)
- **HybridRetriever** — Qdrant: BM25+Dense → weighted RRF 3:1 → ColBERT MaxSim → channel dedup
- **RerankerService** — bge-reranker-v2-m3 cross-encoder (GPU, RTX 5060 Ti)
- **QueryPlannerService** — тот же LLM endpoint (не отдельный CPU процесс)

### Sequence

```mermaid
sequenceDiagram
  autonumber
  participant C as Client
  participant API as FastAPI
  participant Agent as AgentService
  participant LLM as Qwen3-30B-A3B V100
  participant Tools as ToolRunner
  participant Qdrant as HybridRetriever

  C->>API: POST /v1/agent/stream
  API->>API: JWT verify
  API->>Agent: stream_agent_response

  Note over Agent: request_id, AgentState init

  loop ReAct Cycle
    Agent->>LLM: generate_step with tools schema
    Note over LLM: Native function calling<br/>tools parameter + tool_choice
    LLM-->>Agent: tool_calls response

    Agent->>Agent: extract tool name + arguments
    Agent-->>C: SSE thought + tool_invoked

    Agent->>Tools: run tool

    alt tool = search / temporal_search / channel_search / cross_channel_compare / summarize_channel
      Note over Tools: All subqueries, round-robin merge
      loop for each subquery
        Tools->>Qdrant: search_with_plan
        Note over Qdrant: BM25 top-100 + dense top-20<br/>weighted RRF 3:1<br/>ColBERT MaxSim rerank
        Qdrant-->>Tools: candidates per query
      end
      Tools->>Tools: round-robin merge + channel dedup
    else tool = entity_tracker / arxiv_tracker(top)
      Tools->>Qdrant: Facet API / point-level aggregations
      Qdrant-->>Tools: analytics summary + data
      Note over Agent: analytics_done=True<br/>forced search bypass
    else tool = arxiv_tracker(lookup)
      Tools->>Qdrant: Scroll / filter by arxiv_id
      Qdrant-->>Tools: hits[] deduped by root_message_id
      Note over Agent: analytics_done=True,<br/>search_count++ for rerank/compose path
    else tool = hot_topics
      Tools->>Qdrant: Scroll weekly_digests collection
      Qdrant-->>Tools: topic clusters + trends
      Note over Agent: analytics_done=True<br/>forced search bypass
    else tool = channel_expertise
      Tools->>Qdrant: Scroll channel_profiles collection
      Qdrant-->>Tools: channel profile + expertise areas
      Note over Agent: analytics_done=True<br/>forced search bypass
    else tool = rerank
      Tools->>Tools: cross-encoder rerank via gpu_server
    else tool = compose_context
      Tools->>Tools: build prompt + citations
      Note over Tools: composite coverage from cosine similarity
    else tool = final_answer
      Tools-->>Agent: answer text
    end

    Tools-->>Agent: AgentAction result
    Agent-->>C: SSE observation

    alt analytics-only answer ready
      Agent-->>C: SSE final
      Note over Agent: final_answer without compose_context/citations
    else compose_context done
      Agent->>Agent: check coverage
      Agent-->>C: SSE citations

      alt coverage below 0.65 AND refinements left
        Note over Agent: trigger refinement search
      else coverage below 0.30
        Note over Agent: abort, insufficient information
      end
    end

    alt final_answer received
      Agent-->>C: SSE final
      Note over Agent: break loop
    end
  end
```

### Adaptive Tool Selection

LLM видит **специализированные search и analytics tools** и сам выбирает нужный:

| Tool | Когда LLM выбирает | Qdrant filter |
|------|-------------------|---------------|
| `search` | Общие/сравнительные запросы (default) | Нет фильтров |
| `temporal_search` | "в январе 2026", "последние новости" | DatetimeRange по date |
| `channel_search` | "gonzo_ml про трансформеры" | MatchValue по channel |
| `entity_tracker` | "кто чаще упоминается", "динамика", "что упоминается вместе" | Facet по entities[] / year_week |
| `arxiv_tracker` | "какие статьи обсуждали", "кто обсуждал paper 1706..." | Facet/Scroll по arxiv_ids[] |
| `hot_topics` | "главные темы недели", "тренды за последний месяц" | Scroll по weekly_digests collection |
| `channel_expertise` | "на чём специализируется канал", "профиль gonzo_ml" | Scroll по channel_profiles collection |

### Dynamic Tool Visibility

Четыре фазы:
1. **PRE-SEARCH**: `query_plan` + `search` + signal/keyword-routed specialized tools
2. **POST-SEARCH**: `rerank`, `compose_context`, `final_answer`, `related_posts`
3. **NAV-COMPLETE**: `list_channels` ответил, search не нужен → только `final_answer`
4. **ANALYTICS-COMPLETE**: `entity_tracker`/`arxiv_tracker`/`hot_topics`/`channel_expertise` уже ответили → `final_answer` доступен без search

- Hard cap: максимум 5 видимых tools, лишние убираются по eviction priority из `datasets/tool_keywords.json`
- Если LLM не вызывает tools → **forced search** с оригинальным запросом пользователя, кроме negative intent / analytics short-circuit
- Оригинальный запрос **всегда** добавляется в subqueries (BM25 keyword match)
- Rule-based pre-validator (`query_signals.py`) извлекает даты/каналы/entities за <1ms

### Multi-Query Search Detail

```
query_plan → subqueries: ["query A", "query B", "query C"]
  + original user query (всегда добавляется)

for each subquery:
  → search_with_plan(subquery, plan)
    → Qdrant: BM25 top-100 + dense top-20 → weighted RRF (3:1)
    → ColBERT MaxSim rerank (top candidates)
  → append to per_query_results[]

Round-robin merge:
  for rank 0..max_len:
    for each per_query_results:
      if not seen → append to all_candidates

Channel dedup: max 2 docs from same channel → diversity
```

### Coverage Check Detail

```
compose_context(hit_ids, query) → composite_coverage (float 0-1)
  │
  ├── coverage >= 0.65  →  proceed to final_answer
  ├── coverage < 0.65 AND refinements < 2
  │     └── refinements += 1 → ещё один search + compose_context
  ├── coverage < 0.65 AND refinements >= 2
  │     └── proceed anyway (best effort)
  └── coverage < 0.30
        └── abort: вернуть "insufficient information" (не галлюцинировать)

Composite formula (compute в compose_context, НЕ в AgentService):
  scores = [doc.cosine_sim for doc in retrieved_docs]  # из Qdrant with_vectors=True
  max_sim           × 0.25
  mean_top_k        × 0.20
  term_coverage     × 0.20   # доля ключевых слов query в тексте документов
  doc_count_adequacy× 0.15   # docs с cosine > 0.55 / target_k
  score_gap         × 0.15   # 1 - (top1-topk) / top1
  above_threshold   × 0.05
```

### Ключевые инварианты

- RULE 4 (system prompt): FinalAnswer запрещён без предшествующего compose_context
- `AgentState` живёт ровно столько, сколько один запрос
- При отключении клиента (`is_disconnected()`) — loop прерывается
- Все tool timeout через `ToolRunner._run_with_timeout()`
- `compose_context` получает `query` как параметр (необходим для term_coverage)
- `verify` и `fetch_docs` — системные вызовы внутри AgentService, не LLM tools

### Request Isolation

`AgentService` использует `contextvars.ContextVar` для per-request изоляции состояния
(`_current_step`, `_current_request_id`). Реализовано в SPEC-RAG-17 — заменяет прежние shared class attributes (бывший OPEN-01).
