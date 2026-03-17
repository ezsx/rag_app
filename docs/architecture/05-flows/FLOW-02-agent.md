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
- **Client** — browser / evaluate_agent.py
- **FastAPI** — HTTP layer, SSE stream
- **AgentService** — ReAct loop owner
- **ToolRunner** — tool registry + timeout execution
- **LLM** — Qwen3-8B GGUF (llama-server, V100; thinking mode отключён)
- **HybridRetriever** — Qdrant prefetch(dense+sparse) + RRF + MMR
- **RerankerService** — BGE reranker
- **QueryPlannerService** — тот же LLM endpoint (не отдельный CPU процесс)

### Sequence

```mermaid
sequenceDiagram
  autonumber
  participant C as Client
  participant API as FastAPI
  participant Agent as AgentService
  participant LLM as Qwen3-8B (V100)
  participant Tools as ToolRunner
  participant Qdrant as QdrantRetriever

  C->>API: POST /v1/agent/stream {query, max_steps}
  API->>API: require_read(jwt) → user verified
  API->>Agent: stream_agent_response(AgentRequest)

  Note over Agent: request_id = uuid4(), AgentState(coverage=0, refinements=0)

  loop ReAct Cycle (max max_steps iterations)
    Agent->>LLM: _generate_step(conversation_history, system_prompt)
    Note over LLM: EN system prompt + /no_think\nThought: ...\nAction: tool_name {params}
    LLM-->>Agent: raw text (без <think> блоков)

    Agent->>Agent: _parse_action(text) → ToolRequest
    Agent-->>C: SSE: thought {thought, step}
    Agent-->>C: SSE: tool_invoked {tool, input, step}

    Agent->>Tools: run(request_id, step, ToolRequest)
    Note over Tools: httpx.AsyncClient timeout

    alt tool = search
      Tools->>Qdrant: query_points(prefetch=[dense,sparse], FusionQuery(RRF))
      Note over Qdrant: dense+sparse prefetch → RRF → MMR → candidates
      Qdrant-->>Tools: List[ScoredPoint] с cosine_sim (with_vectors=True)
      Tools->>Tools: reranker.rerank(query, candidates)
    else tool = compose_context
      Tools->>Tools: build prompt + citations from hit_ids
      Note over Tools: cosine_sim per doc → composite coverage\nmax_sim×0.25 + mean_top_k×0.20 + ...\n(НЕ RRF-скоры)
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

    alt text contains FinalAnswer:
      Agent->>LLM: _generate_final(context, query)
      Note over LLM: verify step (optional)
      Agent-->>C: SSE: final {answer, steps, request_id}
      Note over Agent: break loop
    end
  end
```

### Coverage Check Detail

```
compose_context(hit_ids, query) → composite_coverage (float 0-1)
  │
  ├── coverage >= 0.65  →  proceed to verify → final_answer
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
- При отключении клиента (`is_disconnected()`) — loop прерывается, генерация останавливается
- Все tool timeout через `ToolRunner._run_with_timeout()` → никогда не висим вечно
- `compose_context` получает `query` как параметр (необходим для term_coverage сигнала)

### Техдолг

- `AgentService._current_step` и `_current_request_id` — shared class attributes
  вместо per-request local → `contextvars.ContextVar` (OPEN-01, R06)
- LLM вызов через httpx.AsyncClient — промежуточный фикс OPEN-02.
  Финальный фикс: AsyncOpenAI + vLLM после Proxmox (DEC-0021)
