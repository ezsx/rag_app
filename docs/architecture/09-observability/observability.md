## Observability

### Логирование

Все компоненты логируют через `logging.getLogger(__name__)`.

**Обязательные поля в структурированных логах** (ToolRunner trace):
```json
{"req": "request_id", "step": 3, "tool": "search", "ok": true, "took_ms": 412}
{"req": "request_id", "step": 3, "tool": "search", "ok": false, "took_ms": 15001, "error": "timeout>15000ms"}
```

### SSE Events как трейсинг

SSE-поток является единственным инструментом трейсинга в реальном времени.
Клиент может наблюдать полный ReAct цикл через события:
`thought → tool_invoked → observation → citations → final`

### Tool Latency

`ToolRunner` измеряет `took_ms` для каждого инструмента.
Evaluation скрипт собирает `agent_latency_sec` (end-to-end).

Ориентиры по latency (Phase 1, Qwen3-8B Q8_0 на V100):
| Компонент | Ожидаемо |
|-----------|---------|
| LLM step (Qwen3-8B Q8_0) | 1–3 сек |
| Qdrant query_points (hybrid) | 50–200 мс |
| BGE reranker (16 docs) | 200–500 мс |
| compose_context (с coverage) | < 50 мс |

### Composite Coverage

`compose_context` вычисляет и логирует сигналы composite coverage:
```json
{
  "composite_coverage": 0.71,
  "signals": {
    "max_sim": 0.84,
    "mean_top_k": 0.72,
    "term_coverage": 0.65,
    "doc_count_adequacy": 0.80,
    "score_gap": 0.55,
    "above_threshold_ratio": 0.60
  }
}
```

### Метрики evaluate_agent.py

| Метрика | Описание |
|---------|---------|
| `latency.agent.mean` | Средняя latency всех запросов |
| `latency.agent.p95` | 95-й перцентиль latency |
| `latency.agent.max` | Максимум |
| `recall.mean_recall_at_5` | Средний recall@5 |
| `recall.queries_with_full_recall` | Запросы с полным recall |
| `coverage.mean_composite` | Средний composite coverage |
| `errors.agent` | Количество ошибок агента |

### Health Endpoints

```
GET /health  →  {"status": "ok"}
```

Нет метрик-эндпоинтов (Prometheus/OpenTelemetry) — за пределами текущего scope.
