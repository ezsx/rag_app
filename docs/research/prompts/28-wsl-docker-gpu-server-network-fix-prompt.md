# Deep Research: Docker ↔ WSL2 gpu_server Networking Fix (No Relay Hacks)

> **Цель**: Найти production-grade способ дать Docker-контейнерам `rag_app` стабильный доступ к WSL2-native `gpu_server.py` (embedding + reranker + ColBERT) **без деградации retrieval pipeline** и, по возможности, **без ad-hoc relay hacks**. Нужен конкретный рекомендуемый fix для нашей среды, а не общий обзор.

---

## Контекст проекта

### Что это
`rag_app` — FastAPI RAG/agent система:
- Docker Desktop: API + Qdrant (CPU)
- Windows Host: `llama-server.exe` на V100 (`:8080`)
- WSL2 native Ubuntu: `gpu_server.py` на RTX 5060 Ti (`:8082`)

`gpu_server.py` обслуживает **три** runtime capability:
- `POST /embed`
- `POST /rerank`
- `POST /colbert-encode`

### Почему это важно
Retrieval pipeline зависит от этого сервера:
- dense embeddings
- cross-encoder reranker
- ColBERT query encoding

Если reranker / ColBERT недоступны, агент **не падает**, но retrieval silently деградирует:
- работает только dense + sparse RRF fallback
- quality ухудшается
- agent выглядит “живым”, но pipeline уже не production-grade

---

## Текущая инфраструктура

### Реальная топология
1. **Windows Host**
   - `llama-server.exe` → `http://host.docker.internal:8080`

2. **WSL2 Ubuntu**
   - `gpu_server.py` слушает `0.0.0.0:8082`
   - внутри WSL доступны:
     - `GET /health`
     - `POST /embed`
     - `POST /rerank`
     - `POST /colbert-encode`

3. **Docker Desktop**
   - API контейнер не видит WSL-native `:8082` напрямую через `host.docker.internal:8082`
   - для обхода был заведён Windows relay на `:18082`

### Текущий compose
В `compose.dev.yml` сейчас:
- `EMBEDDING_TEI_URL=http://host.docker.internal:18082`
- `RERANKER_TEI_URL=http://host.docker.internal:18082`

То есть compose предполагает, что один relay на `18082` проксирует **весь** `gpu_server`.

---

## Что уже проверено руками

### 1. Внутри WSL `gpu_server.py` работает корректно
Из WSL:
- `http://127.0.0.1:8082/health` → `200`
- `http://127.0.0.1:8082/embed` → `200`
- `http://127.0.0.1:8082/rerank` → `200`
- `http://127.0.0.1:8082/colbert-encode` → `200`

### 2. Из Docker через relay `18082` работает только embedding path
Из API контейнера:
- `http://host.docker.internal:18082/health` → `200`
- `http://host.docker.internal:18082/embed` → `200`
- `http://host.docker.internal:18082/rerank` → `404`
- `http://host.docker.internal:18082/colbert-encode` → `404`

### 3. Из Docker прямой доступ к `host.docker.internal:8082` не работает
Из API контейнера:
- `http://host.docker.internal:8082/health` → connection refused

### 4. Retrieval pipeline реально страдает
В логах API:
- `POST http://host.docker.internal:18082/rerank` → `404 Not Found`
- `ColBERT query encoding failed: HTTP Error 404: Not Found`

Итог:
- analytics tools работают
- agent выглядит живым
- но reranker и ColBERT path сейчас broken

---

## Что уже НЕ нужно предлагать

Не нужно предлагать как “фикс”:

1. **Просто отключить reranker / ColBERT**
   - это деградация, а не решение

2. **Оставить relay как есть**
   - он обслуживает только `/embed`, а не весь runtime path

3. **“Переехать всем в Docker GPU”**
   - это уже исследовано; V100 TCC/NVML poisoning делает этот путь непрактичным в нашей конфигурации

4. **“Переключить всё на localhost:8082”**
   - Docker контейнеры это не видят

---

## Главный вопрос исследования

Как в нашей конкретной среде сделать так, чтобы Docker-контейнеры стабильно и production-grade ходили к WSL2-native `gpu_server.py` со **всеми** endpoint-ами:
- `/embed`
- `/rerank`
- `/colbert-encode`

и при этом:
- не ломать Windows-host `llama-server.exe`
- не ломать VPN / mirrored networking setup
- не ухудшать retrieval quality
- не вводить fragile hand-made routing, который снова сломается через неделю

---

## Что хочу получить

### 1. Root cause analysis

Объясни:
- почему `host.docker.internal:8082` недоступен из Docker, хотя `gpu_server.py` жив в WSL
- почему текущий relay даёт `200` на `/embed`, но `404` на `/rerank` и `/colbert-encode`
- это limitation mirrored networking, limitation конкретного relay, limitation Docker Desktop, limitation Windows↔WSL port forwarding, или комбинация факторов?

### 2. Option table

Сравни 4-6 реалистичных вариантов, например:
- direct access по WSL IP
- Windows `netsh interface portproxy`
- полноценный reverse proxy на Windows
- отдельные relay ports для каждого endpoint
- запуск proxy/bridge внутри Docker network
- отказ от mirrored networking

Для каждого варианта дай:
- works / doesn’t work
- latency impact
- stability
- ops complexity
- security implications
- compatibility с нашей связкой `Windows Host + WSL2 + Docker Desktop + V100 + RTX 5060 Ti`

### 3. Recommended fix

Выбери **один** основной путь и, если нужно, один backup path.

Нужен не abstract advice, а конкретный ответ:
- что именно менять
- где именно менять
- какие env vars / ports / routes использовать
- как потом верифицировать, что `embed + rerank + colbert` реально доступны из контейнера

### 4. Implementation plan

Дай пошаговый plan:
1. infra change
2. compose/env change
3. verification commands
4. rollback plan

### 5. Risk assessment

Ответь отдельно:
- насколько это решение устойчиво к reboot / Docker restart / WSL restart
- не сломает ли это `llama-server.exe` на Windows
- не сломает ли это `repo-semantic-search` / local tooling
- есть ли hidden pitfalls для mirrored mode

---

## Дополнительные вопросы

1. Есть ли production-grade способ использовать **один** stable hostname/port для всех endpoint-ов `gpu_server.py`, чтобы compose не приходилось “знать” про внутренности proxy?

2. Если direct WSL IP — хороший путь, насколько этот IP стабилен при reboot / restart WSL? Нужен ли discovery step?

3. Если нужен proxy, что лучше:
- layer-4 portproxy
- layer-7 reverse proxy
- маленький Python/Go relay
- Windows IIS / nginx / Caddy

4. Можно ли сделать это так, чтобы:
- Docker видел `gpu_server`
- host tools тоже видели `gpu_server`
- не было special-case path только для `/embed`

5. Какой вариант наиболее “production-looking” на собесе, а не просто “у меня локально завелось”?

---

## Артефакты для проверки

Пожалуйста, опирайся не только на этот prompt, но и верифицируй по репо:
- `scripts/gpu_server.py`
- `deploy/compose/compose.dev.yml`
- `src/core/settings.py`
- `src/adapters/tei/reranker_client.py`
- `src/adapters/search/hybrid_retriever.py`
- `docs/architecture/11-decisions/decision-log.md` (`DEC-0038`)

Важно:
- не принимай текущий relay как корректный design just because it exists
- challenge assumptions
- ищи **настоящий infra fix**, а не workaround, который просто скрывает деградацию

---

## Формат ответа

Структурированный отчёт:

1. **Root Cause**
2. **Options Table**
3. **Recommended Fix**
4. **Exact Implementation Steps**
5. **Verification Checklist**
6. **Risks / Tradeoffs**
7. **Why this is better than relay-only workaround**
