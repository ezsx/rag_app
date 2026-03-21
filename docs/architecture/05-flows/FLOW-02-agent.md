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
- **AgentService** — ReAct loop owner, native function calling
- **ToolRunner** — tool registry + timeout execution
- **LLM** — Qwen3-30B-A3B GGUF (llama-server, V100; `--jinja --reasoning-budget 0`)
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
  participant LLM as Qwen3-30B-A3B (V100)
  participant Tools as ToolRunner
  participant Qdrant as HybridRetriever

  C->>API: POST /v1/agent/stream {query, max_steps}
  API->>API: require_read(jwt) → user verified
  API->>Agent: stream_agent_response(AgentRequest)

  Note over Agent: request_id = uuid4(), AgentState(coverage=0, refinements=0)

  loop ReAct Cycle (max max_steps iterations)
    Agent->>LLM: _generate_step(messages, tools=[query_plan,search,rerank,compose_context,...])
    Note over LLM: Native function calling via /v1/chat/completions<br/>tools parameter + tool_choice
    LLM-->>Agent: response.choices[0].message.tool_calls

    Agent->>Agent: extract tool_call → name, arguments
    Agent-->>C: SSE: thought {thought, step}
    Agent-->>C: SSE: tool_invoked {tool, input, step}

    Agent->>Tools: run(request_id, step, ToolRequest)
    Note over Tools: httpx.AsyncClient timeout

    alt tool = search
      Note over Tools: Итерация по ВСЕМ subqueries (round-robin merge)
      loop for each subquery in plan
        Tools->>Qdrant: search_with_plan(subquery, search_plan)
        Note over Qdrant: BM25 top-100 + dense top-20<br/>→ RrfQuery(weights=[1.0, 3.0])<br/>→ ColBERT MaxSim rerank (if available)
        Qdrant-->>Tools: List[Candidate] per query
      end
      Tools->>Tools: round-robin merge (interleave, dedup by id)
      Tools->>Tools: channel_dedup(max 2 per channel)
    else tool = rerank
      Tools->>Tools: reranker.rerank(query, candidates) via gpu_server /rerank
    else tool = compose_context
      Tools->>Tools: build prompt + citations from hit_ids
      Note over Tools: cosine_sim per doc → composite coverage<br/>max_sim×0.25 + mean_top_k×0.20 + ...<br/>(НЕ RRF-скоры)
    else tool = final_answer
      Tools-->>Agent: {answer: "..."}
    end

    Tools-->>Agent: AgentAction(ok, data, took_ms)
    Agent-->>C: SSE: observation {output, ok, step}

    alt tool = compose_context
      Agent->>Agent: check composite coverage
      Agent-->>C: SSE: citations {citations, coverage}

      alt coverage < 0.65 AND refinements < max_refinements (=2)
        Note over Agent: refinements += 1 → trigger refinement search
      else coverage < 0.30
        Note over Agent: abort → "insufficient information" (не галлюцинировать)
      end
    end

    alt tool = final_answer
      Agent-->>C: SSE: final {answer, steps, request_id}
      Note over Agent: break loop
    end
  end
```

### Dynamic Tool Visibility

- `final_answer` **скрыт** до выполнения `search` — LLM не может пропустить поиск
- Если LLM не вызывает tools → **forced search** с оригинальным запросом пользователя
- Оригинальный запрос **всегда** добавляется в subqueries (BM25 keyword match)

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

### Техдолг

- `AgentService._current_step` и `_current_request_id` — shared class attributes
  вместо per-request local → `contextvars.ContextVar` (OPEN-01, R06)
