# Claude Router

Этот файл загружается всегда. Trigger rules — **обязательные**, не рекомендательные.
Доменные модули **must load** при совпадении задачи — не "можно прочитать", а "обязан прочитать".

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

- Если задача про **документацию**, структуру docs/, создание/перемещение файлов:
  - прочитай `docs/architecture/00-meta/02-documentation-governance.md`

- Если задача про **debugging**, failure analysis, баг, неожиданное поведение:
  - прочитай `agent_context/modules/debugging_protocol.md`

- Если задача требует **параллельной работы** Claude + Codex, dispatch нескольких задач,
  или нужно организовать write+review workflow:
  - прочитай `agent_context/modules/parallel_agents.md`

- Если задача про **эксперимент**, eval прогон, ablation, A/B сравнение pipeline,
  измерение метрик, запуск evaluate_retrieval/evaluate_agent:
  - прочитай `experiments/PROTOCOL.md`
  - прочитай `experiments/baseline.yaml`
  - прочитай `experiments/log.md`

## Комбинирование

- Если задача затрагивает несколько доменов — читай только нужные модули.
- Не загружай все модули заранее.
