## API Contracts

### Auth

Все защищённые endpoints требуют `Authorization: Bearer <jwt>`.

```
POST /v1/auth/admin
Body: {"key": "<ADMIN_KEY>"}
→ {"access_token": "...", "token_type": "bearer"}
```

JWT содержит: `sub`, `roles: ["admin", "write", "read"]`, `exp`.

---

### Agent Stream

```
POST /v1/agent/stream
Auth: require_read
Body: AgentRequest

AgentRequest {
  query:           str        (1–1000 chars, required)
  model_profile:   str | null (зарезервировано, не используется)
  tools_allowlist: list[str] | null
  planner:         bool       (default: true)
  max_steps:       int        (1–15, default: 8)
}

Response: text/event-stream
  event: thought
  data: {"thought": str, "step": int, "request_id": str}

  event: tool_invoked
  data: {"tool": str, "input": dict, "step": int, "request_id": str}

  event: observation
  data: {"tool": str, "output": dict, "ok": bool, "took_ms": int,
         "step": int, "request_id": str}

  event: citations
  data: {"citations": list[Citation], "coverage": float,
         "step": int, "request_id": str}

  event: final
  data: {"answer": str, "step": int, "total_steps": int, "request_id": str,
         "citations": list[Citation] | null}
```

**Примечание**: поле `coverage` в событии `citations` — composite coverage (0–1),
вычисленное из 5 сигналов (cosine similarity, term_coverage, etc.), не RRF-скоры.

---

### QA Stream (baseline)

```
POST /v1/qa/stream
Auth: не требуется (TODO: добавить)
Body: {"query": str, "include_context": bool}

Response: text/event-stream
  event: token   {"token": str}
  event: done    {"total_tokens": int}
  event: context {"context": list[str]}  (if include_context=true)
```

---

### Системные эндпоинты

```
GET  /health           → {"status": "ok"}
GET  /v1/models        → список доступных LLM моделей
POST /v1/models/switch → переключить текущую LLM модель
GET  /v1/search        → прямой поиск в Qdrant (debug)
POST /v1/ingest        → не реализован в API, только CLI
```

---

### Error Responses

| Code | When |
|------|------|
| 401 | Отсутствует или невалидный JWT |
| 403 | Недостаточно прав |
| 422 | Невалидное тело запроса (Pydantic validation) |
| 503 | LLM недоступна (llama-server не запущен) / Qdrant недоступен |
