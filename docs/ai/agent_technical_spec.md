# Техническая спецификация ReAct агента

> **Статус:** Phase 1 (Qdrant + TEI HTTP)
> **Актуально на:** 2026-03-17
> **LLM:** Qwen3-8B GGUF (llama-server.exe, V100)
> **Embedding:** multilingual-e5-large (TEI HTTP, WSL2 native)
> **Reranker:** bge-reranker-v2-m3 (TEI HTTP, WSL2 native)
> **Store:** Qdrant (dense + sparse, native RRF+MMR)

## Обзор

ReAct агент реализован в `src/services/agent_service.py` и представляет собой систему пошагового мышления-действия-наблюдения с детерминированной логикой проверки покрытия и верификации ответов. Агент использует SSE стриминг для наблюдаемости и автоматически выполняет refinement раунды при недостаточном покрытии контекста.

**Основная модель**: Qwen3-8B GGUF (V100 SXM2 32GB, 8k контекст, llama-server.exe)

---

## Архитектура компонентов

### AgentService (`src/services/agent_service.py`)

**Основной класс**: `AgentService`

**Зависимости**:
- `llm_factory: Callable` - фабрика для создания LLM (через `get_llm()`)
- `tool_runner: ToolRunner` - реестр и выполнитель инструментов
- `settings: Settings` - конфигурация системы
- `qa_service: Optional[QAService]` - fallback для классического QA

**Внутреннее состояние**:
- `_current_request_id: Optional[str]` - ID текущего запроса
- `_current_step: int` - текущий номер шага
- `_current_query: Optional[str]` - текущий запрос пользователя
- `_last_search_hits: List[Dict]` - последние результаты поиска
- `_last_search_route: str` - последний использованный маршрут поиска
- `_last_plan_summary: Dict` - сводка последнего плана запроса
- `_last_compose_citations: List[Dict]` - последние цитаты из compose_context

### AgentState (внутренний класс)

```python
class AgentState:
    coverage: float = 0.0           # Citation coverage последнего compose_context
    refinement_count: int = 0       # Количество выполненных refinement раундов
    max_refinements: int = 2        # Максимум refinement раундов (DEC-0019)
```

---

## Алгоритм работы (детерминированная логика)

### Основной цикл ReAct

1. **Инициализация**:
   - Создается `AgentState` для отслеживания coverage и refinement_count
   - Валидация и санитизация входного запроса через `SecurityManager`
   - Генерация `request_id` (UUID)

2. **Цикл по шагам** (до `max_steps`):
   - **Генерация шага**: LLM генерирует следующий шаг через `_generate_step()`
   - **Парсинг ответа**: `_parse_llm_response()` извлекает `thought`, `action`, `final_answer`
   - **Отправка события `thought`**: если есть мысль агента
   - **Проверка финального ответа**: если есть `FinalAnswer` → переход к верификации
   - **Выполнение действия**: если есть `action` → `_execute_action()` → выполнение через `ToolRunner`
   - **Отправка событий**: `tool_invoked` → `observation` → `citations` (если compose_context)

3. **Детерминированная проверка покрытия** (после `compose_context`):
   - Извлекается `citation_coverage` из результата инструмента
   - Проверка: `coverage < coverage_threshold` (по умолчанию 0.65, DEC-0019) И `refinement_count < max_refinements` (по умолчанию 2, DEC-0019)
   - Если условие выполнено → автоматический refinement раунд:
     - Системно генерируется `thought` о необходимости дополнительного поиска
     - Выполняется `_perform_refinement()` с расширенными параметрами (`k = hybrid_top_bm25 * 2`, максимум 200)
     - `refinement_count` увеличивается на 1
     - Цикл продолжается БЕЗ инкремента шага (refinement не считается отдельным шагом)

4. **Верификация финального ответа** (перед `FinalAnswer`):
   - Если `enable_verify_step=True` (по умолчанию):
     - Вызывается `_verify_answer()` через инструмент `verify`
     - Проверяется `confidence >= 0.6` (threshold)
     - Если `confidence < 0.6` И `refinement_count < max_refinements`:
       - Выполняется verification refinement (дополнительный поиск для верификации)
       - Цикл продолжается для дополнительной проверки
     - Если `confidence < 0.6` И достигнут лимит refinement:
       - Добавляется предупреждение "(⚠️ Answer not verified with high confidence)"

5. **Пост-обработка финального ответа**:
   - Удаление артефактов ReAct (повторные Thought/Action маркеры)
   - Проверка языка ответа (должен соответствовать языку запроса)
   - Если русский запрос, но ответ содержит CJK символы → регенерация на русском
   - Вызов инструмента `final_answer` для унификации payload

6. **Fallback при превышении max_steps**:
   - Если достигнут `max_steps` без финального ответа:
     - Используется `QAService.answer()` для классического QA
     - Возвращается ответ с метаданными `fallback: true`

---

## Инструменты (7 базовых)

Все инструменты регистрируются в `ToolRunner` через `core.deps.get_agent_service()`.

### 1. router_select
**Назначение**: Выбирает оптимальный маршрут поиска (bm25/dense/hybrid)

**Параметры**:
```json
{"query": "string"}
```

**Возвращает**: `{"route": "hybrid|bm25|dense"}`

---

### 2. query_plan
**Назначение**: Создает план поиска с декомпозицией запроса и фильтрами

**Параметры**:
```json
{"query": "string"}
```

**Возвращает**:
```json
{
  "plan": {
    "normalized_queries": ["string", ...],
    "k_per_query": int,
    "fusion": "rrf|mmr",
    "filters": {
      "date_from": "YYYY-MM-DD",
      "date_to": "YYYY-MM-DD",
      "channel": "string"
    }
  }
}
```

**Особенности**:
- Использует QueryPlannerService с GBNF грамматикой (опционально)
- Таймаут: 15 секунд (по умолчанию)
- Кеширование планов (TTL ~10 минут)

---

### 3. search
**Назначение**: Выполняет гибридный поиск по коллекции с RRF слиянием

**Параметры**:
```json
{
  "queries": ["string", ...],
  "route": "hybrid|bm25|dense",
  "k": int,
  "filters": {
    "date_from": "YYYY-MM-DD",
    "date_to": "YYYY-MM-DD",
    "channel": "string"
  }
}
```

**Возвращает**:
```json
{
  "hits": [
    {
      "id": "string",
      "text": "string",
      "metadata": {...},
      "score": float
    }
  ],
  "route_used": "hybrid|bm25|dense",
  "total_found": int
}
```

**Особенности**:
- Поддерживает гибридный поиск (Qdrant prefetch+FusionQuery(RRF), dense+sparse) с RRF слиянием
- Fallback на BM25 при недоступности Qdrant
- Использует HybridRetriever → Qdrant native RRF (prefetch dense + sparse → FusionQuery)

---

### 4. rerank
**Назначение**: Переранжирует документы по релевантности через TEI HTTP reranker (bge-reranker-v2-m3)

**Параметры**:
```json
{
  "query": "string",
  "docs": ["text1", "text2", ...],
  "top_n": int
}
```

**Возвращает**:
```json
{
  "indices": [2, 0, 1],
  "scores": [0.95, 0.87, 0.72],
  "top_n": 3
}
```

**Особенности**:
- TEI HTTP → bge-reranker-v2-m3 (sync bridge over async TEIRerankerClient, sigmoid-normalized scores)
- Batch processing с размером батча `reranker_batch_size` (по умолчанию 16)
- Автоматически извлекает `docs` из последних `search` hits если не переданы

---

### 5. fetch_docs
**Назначение**: Получает документы по списку ID из Qdrant

**Параметры**:
```json
{"ids": ["id1", "id2", ...]}
```

**Возвращает**:
```json
{
  "docs": [
    {
      "id": "string",
      "text": "string",
      "metadata": {...}
    }
  ]
}
```

---

### 6. compose_context
**Назначение**: Собирает контекст из документов с цитированием и вычисляет citation coverage

**Параметры** (нормализуются автоматически):
```json
{
  "hit_ids": ["id1", "id2", ...],  // опционально, берутся из последних search hits
  "docs": [{id, text, metadata}],   // автоматически заполняется из hit_ids
  "max_tokens_ctx": int             // по умолчанию 1200, ограничение для предотвращения overflow
}
```

**Возвращает**:
```json
{
  "prompt": "string",               // форматированный контекст с цитатами [1], [2], ...
  "citations": [
    {
      "id": "string",
      "index": int,
      "metadata": {...}
    }
  ],
  "contexts": ["string", ...],      // список текстов контекстов
  "citation_coverage": float        // len(citations) / len(docs), метрика покрытия
}
```

**Особенности**:
- **Lost-in-the-middle mitigation**: переупорядочивает документы (наиболее релевантные в начало и конец)
- **Citation coverage**: автоматически вычисляется как доля документов, включенных в контекст
- Ограничение по токенам: `max_tokens_ctx * 4` символов (приблизительно)
- Автоматически вызывает `fetch_docs` для документов без текста

---

### 7. verify
**Назначение**: Проверяет утверждения через поиск в базе знаний с confidence scoring

**Параметры**:
```json
{
  "query": "string",                // исходный запрос пользователя
  "claim": "string",                // утверждение для проверки
  "top_k": int                      // количество документов для поиска (по умолчанию 3)
}
```

**Возвращает**:
```json
{
  "verified": bool,                 // true если confidence >= threshold
  "confidence": float,              // средняя уверенность по документам (0.0-1.0)
  "threshold": float,                // порог верификации (по умолчанию 0.6)
  "evidence": ["string", ...],       // фрагменты найденных документов
  "documents_found": int,            // количество найденных документов
  "used_docs": int                   // количество использованных документов
}
```

**Алгоритм confidence**:
- Для каждого документа вычисляется confidence на основе расстояния (distance) или ранга
- Если есть distance: `confidence = max(0.0, 1.0 - min(distance, 2.0) / 2.0)`
- Если нет distance: `confidence = max(0.3, 1.0 - idx * 0.1)` (чем выше ранг, тем выше confidence)
- Средняя confidence сравнивается с threshold (0.6)

---

### 8. final_answer (системный инструмент)
**Назначение**: Унифицирует финальный ответ с метаданными

**Параметры**:
```json
{
  "answer": "string",
  "citations": [{id, index, metadata}],
  "coverage": float,
  "refinements": int,
  "verification": {
    "verified": bool,
    "confidence": float,
    "documents_found": int
  },
  "route": "string",
  "plan": {...}
}
```

**Возвращает**: Нормализованный payload для финального события

---

## Системный промпт

Агент использует детальный системный промпт (`self.system_prompt`) с:

1. **Строгая языковая политика (RU)**:
   - Определение языка пользователя из первого сообщения
   - Если русский → строго русский язык (без латиницы, без CJK)
   - Разрешенные символы: кириллица U+0400-U+04FF, цифры, пунктуация
   - Опциональные русские метки: Мысль/Действие/Наблюдение/Ответ

2. **Формат ReAct**:
   ```
   Thought: [reasoning]
   Action: [tool_name] {"param": "value"}
   Observation: [tool result]
   FinalAnswer: [final answer]
   ```

3. **Правила работы**:
   - Всегда начинать с Thought
   - ОБЯЗАТЕЛЬНО: после каждого успешного search вызывать compose_context с hit_ids
   - НИКОГДА не генерировать FinalAnswer без compose_context
   - Передавать ТОЛЬКО указанные параметры инструментам
   - Соответствовать языку запроса во всех выводах

4. **Детерминированная логика системы**:
   - После compose_context система автоматически проверяет citation coverage (>=80%)
   - При недостаточном покрытии система выполняет дополнительный поиск
   - Перед финальным ответом система может верифицировать утверждения
   - Максимум 2 дополнительных раунда (DEC-0019)

---

## Метрики и их вычисление

### Citation Coverage

**Определение**: Доля документов из результатов поиска, включенных в финальный контекст.

**Формула**: `citation_coverage = len(citations) / len(docs)` где:
- `citations` - список документов, включенных в контекст (после ограничения по токенам)
- `docs` - все документы из результатов поиска

**Порог**: `coverage_threshold = 0.65 (DEC-0019)` (по умолчанию)

**Использование**:
- Автоматическая проверка после каждого `compose_context`
- Если `coverage < threshold` → запуск refinement раунда

**Ограничения**:
- Coverage может быть < 1.0 из-за ограничения по токенам (`max_tokens_ctx`)
- Не учитывает релевантность включенных документов (только количество)

---

### Verification Confidence

**Определение**: Уверенность в том, что финальный ответ подтверждается источниками из базы знаний.

**Формула**: Средняя confidence по топ-K документам, найденным для утверждения:
- Для каждого документа: `confidence = f(distance, rank)`
- Средняя: `avg_confidence = sum(confidences) / len(confidences)`

**Порог**: `threshold = 0.6` (в инструменте verify)

**Использование**:
- Проверка перед финальным ответом (если `enable_verify_step=True`)
- Если `confidence < threshold` → возможен verification refinement

---

### Latency (Время выполнения)

**Измерение**: Время от начала запроса до финального ответа (включая все шаги, refinement, верификацию)

**Компоненты**:
- Генерация шагов LLM
- Выполнение инструментов (с таймаутами)
- Refinement раунды
- Верификация

**Метаданные**: Каждый инструмент возвращает `took_ms` в `ToolMeta`

---

### Refinement Count

**Определение**: Количество выполненных refinement раундов для текущего запроса.

**Ограничение**: `max_refinements = 2 (DEC-0019)` (по умолчанию)

**Типы refinement**:
1. **Coverage refinement**: При `coverage < threshold` после `compose_context`
2. **Verification refinement**: При `confidence < threshold` после верификации

---

## SSE Стриминг и события

Агент использует Server-Sent Events (SSE) для пошаговой передачи результатов.

### Типы событий

1. **`step_started`**: Начало нового шага
   ```json
   {
     "step": int,
     "request_id": "uuid",
     "max_steps": int,
     "query": "string"
   }
   ```

2. **`thought`**: Мысль агента (может быть system-generated)
   ```json
   {
     "content": "string",
     "step": int,
     "request_id": "uuid",
     "system_generated": bool,        // опционально
     "verification": bool,             // опционально
     "refinement": bool                 // опционально
   }
   ```

3. **`tool_invoked`**: Вызов инструмента
   ```json
   {
     "tool": "string",
     "input": {...},
     "step": int,
     "request_id": "uuid",
     "refinement": bool,               // опционально
     "verification_refinement": bool   // опционально
   }
   ```

4. **`observation`**: Результат инструмента
   ```json
   {
     "content": "string",
     "success": bool,
     "step": int,
     "request_id": "uuid",
     "took_ms": int,
     "refinement": bool,               // опционально
     "verification_refinement": bool   // опционально
   }
   ```

5. **`citations`**: Цитаты из compose_context
   ```json
   {
     "citations": [{id, index, metadata}],
     "coverage": float,
     "step": int,
     "request_id": "uuid"
   }
   ```

6. **`final`**: Финальный ответ с метаданными
   ```json
   {
     "answer": "string",
     "citations": [{id, index, metadata}],
     "coverage": float,
     "refinements": int,
     "verification": {
       "verified": bool,
       "confidence": float,
       "documents_found": int
     },
     "step": int,
     "total_steps": int,
     "request_id": "uuid",
     "fallback": bool,                 // опционально
     "error": "string"                 // опционально
   }
   ```

---

## Настройки и конфигурация

### Основные настройки агента (`Settings`)

```python
# Включение/выключение агента
enable_agent: bool = True

# Лимиты шагов
agent_max_steps: int = 15          # максимальное количество шагов
agent_default_steps: int = 8       # значение по умолчанию для max_steps
agent_token_budget: int = 2000     # лимит токенов для LLM

# Таймауты
agent_tool_timeout: float = 15.0   # таймаут выполнения инструментов (секунды)

# Детерминированная логика
coverage_threshold: float = 0.65  # порог citation coverage для refinement
max_refinements: int = 2          # максимум refinement раундов
enable_verify_step: bool = True   # включение верификации финального ответа

# Параметры декодирования LLM
agent_tool_temp: float = 0.2       # temperature для шагов инструментов
agent_tool_top_p: float = 0.9
agent_tool_top_k: int = 40
agent_tool_repeat_penalty: float = 1.15
agent_tool_max_tokens: int = 64

agent_final_temp: float = 0.3      # temperature для финальных ответов
agent_final_top_p: float = 0.9
agent_final_max_tokens: int = 512
```

### Переменные окружения

Все настройки могут быть переопределены через переменные окружения:
- `ENABLE_AGENT=true`
- `AGENT_MAX_STEPS=15`
- `AGENT_DEFAULT_STEPS=8`
- `AGENT_TOKEN_BUDGET=2000`
- `AGENT_TOOL_TIMEOUT=15.0`
- `COVERAGE_THRESHOLD=0.65`
- `MAX_REFINEMENTS=2`
- `ENABLE_VERIFY_STEP=true`

---

## Обработка ошибок и fallback

### Типы ошибок и обработка

1. **Security violations** (входной запрос):
   - Валидация через `SecurityManager.validate_input()`
   - Возврат события `final` с `error: "security_violation"`

2. **Ошибки парсинга LLM ответа**:
   - Невалидный JSON в параметрах инструмента → fallback в `raw_input`
   - Невалидный формат ReAct → логирование и продолжение

3. **Ошибки выполнения инструментов**:
   - Таймаут инструмента → `ToolResponse.ok = False`, `meta.error = "timeout>Xms"`
   - Ошибка выполнения → изоляция через `ToolRunner`, логирование, продолжение работы

4. **Превышение max_steps**:
   - Fallback через `QAService.answer()` (классический QA)
   - Возврат ответа с `fallback: true`

5. **Ошибки LLM генерации**:
   - Логирование и возврат сообщения об ошибке в `final` событии

### Логирование

Все вызовы инструментов логируются в JSON формате:
```json
{
  "req": "request_id",
  "step": int,
  "tool": "tool_name",
  "ok": bool,
  "took_ms": int,
  "error": "string"  // опционально
}
```

---

## Особенности реализации

### Нормализация параметров инструментов

Метод `_normalize_tool_params()` автоматически нормализует параметры для совместимости:

- **compose_context**: 
  - Извлекает `hit_ids` → автоматически заполняет `docs` из `_last_search_hits`
  - Если документы без текста → вызывает `fetch_docs`
  - Ограничивает `max_tokens_ctx` для предотвращения overflow

- **fetch_docs**: 
  - Нормализует `hit_ids`/`doc_ids` → `ids`

- **rerank**: 
  - Автоматически добавляет `docs` из последних hits если не переданы
  - Добавляет `query` из текущего запроса если не передано

- **verify**: 
  - Нормализует `k` → `top_k`

### Ограничение истории разговора

Для предотвращения overflow контекста используется скользящее окно:
- Максимум `MAX_HISTORY_ITEMS = 10` элементов истории
- Сохраняется первый элемент (исходный запрос) + последние 10 элементов

### Языковая пост-обработка

Если русский запрос, но ответ содержит CJK символы:
1. Определяется необходимость регенерации
2. Выполняется регенерация через LLM с промптом на русском
3. Fallback на простую санитизацию (удаление CJK символов)

### Lost-in-the-Middle Mitigation

В `compose_context` реализована стратегия переупорядочивания документов:
- Наиболее релевантные документы размещаются в начало и конец контекста
- Менее релевантные документы размещаются в середину
- Это помогает модели не "терять" важную информацию в середине длинного контекста

---

## API Endpoint

### POST `/v1/agent/stream`

**Параметры запроса** (`AgentRequest`):
```json
{
  "query": "string",                    // обязательно, 1-1000 символов
  "collection": "string",              // опционально
  "model_profile": "string",           // опционально
  "tools_allowlist": ["string"],        // опционально
  "planner": true,                      // использовать ли планировщик
  "max_steps": 8                        // 1-15, по умолчанию 8
}
```

**Ответ**: SSE stream с событиями типа `step_started`, `thought`, `tool_invoked`, `observation`, `citations`, `final`

**Аутентификация**: Требуется JWT токен с правами `read`

---

## Интеграция с системой

### Создание AgentService

Через `core.deps.get_agent_service()`:
1. Создается `ToolRunner` с таймаутом по умолчанию
2. Регистрируются все 7 базовых инструментов с инъекцией зависимостей
3. Создается `AgentService` с LLM фабрикой, ToolRunner, Settings, QAService

### Зависимости инструментов

Инструменты получают зависимости через замыкания/инъекцию:
- `hybrid_retriever: HybridRetriever` - через `get_hybrid_retriever()` (Qdrant prefetch+RRF)
- `settings: Settings` - через `get_settings()`
- `query_planner: QueryPlannerService` - через `get_query_planner()`
- `reranker: RerankerService` - через `get_reranker()`
- `qa_service: QAService` - через `get_qa_service()`

### Горячее переключение

При изменении коллекции/модели через `Settings.update_collection()` / `update_llm_model()`:
- Сбрасываются кеши зависимостей (`@lru_cache`)
- AgentService пересоздается с новыми настройками

---

## Текущие ограничения и известные проблемы

1. **Финальный ответ формируется через обходной путь**: Не через инструмент FinalAnswer в промпте, а через системную обработку `FinalAnswer` токена

2. **Ответ модели может содержать устаревшую информацию**: Промпт может упоминать старые модели (Claude/PaLM) вместо актуальных

3. **Citation coverage (Phase 1)**: наивный doc count ratio. SPEC-RAG-07 заменит на 5-signal composite metric

4. **Verification confidence может быть неточным**: Основан на эвристиках расстояния/ранга, не на семантическом сходстве

5. **collections.py endpoint отключён** (Phase 0 ChromaDB код, ожидает SPEC-RAG-06)

---

## Планируемые улучшения

1. Оптимизация таймаутов на основе реальных метрик
2. Добавление поддержки стриминга токенов ответа
3. Улучшение метрик оценки качества (Citation Precision, Recall@5)
4. Автоматическая оценка качества ответов через eval скрипт

---

## Ссылки на код

- **Основной класс**: `src/services/agent_service.py:29` (AgentService)
- **Схемы**: `src/schemas/agent.py`
- **API endpoint**: `src/api/v1/endpoints/agent.py:20`
- **Инструменты**: `src/services/tools/`
- **Настройки**: `src/core/settings.py:91-120`
- **Зависимости**: `src/core/deps.py` (get_agent_service)

