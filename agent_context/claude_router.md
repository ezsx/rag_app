# Claude Router

Этот файл загружается всегда, но сам остаётся маленьким.
Доменные модули читаются только через Read tool по совпадению задачи.

## Правила маршрутизации

- Если задача про **ReAct агент**, шаги цикла, инструменты агента, AgentService,
  ToolRunner, coverage/refinement, verify, compose_context:
  - прочитай `agent_context/modules/agent.md`

- Если задача про **retrieval pipeline**, Qdrant, HybridRetriever,
  RRF fusion, MMR, reranker, QueryPlanner, TEI embedding:
  - прочитай `agent_context/modules/retrieval.md`

- Если задача про **Telegram ingest**, парсинг каналов, скрипты ingestion,
  Qdrant коллекции, eval dataset, evaluation скрипт, метрики качества:
  - прочитай `agent_context/modules/ingest_eval.md`

- Если задача про **API слой**, FastAPI endpoints, SSE стриминг, JWT/auth,
  rate limiting, DI/deps, схемы (schemas):
  - прочитай `agent_context/modules/api.md`

## Комбинирование

- Если задача затрагивает несколько доменов — читай только нужные модули.
- Не загружай все модули заранее.
