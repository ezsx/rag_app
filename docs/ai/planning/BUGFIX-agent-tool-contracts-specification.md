# BUGFIX: Исправление контрактов инструментов агента

**Дата:** 2025-10-05  
**Приоритет:** Критический  
**Тип:** Bugfix — поиск причины и минимальное исправление с тестом на регрессию  

---

## Executive Summary

Обнаружено **5 критических расхождений** между системным промптом агента, реальными сигнатурами инструментов и логикой нормализации параметров. Это приводит к каскаду ошибок:

1. **compose_context**: LLM генерирует параметры `query`, `hits`, `hit_ids`, но функция их не принимает
2. **fetch_docs**: LLM передает `hit_ids`, но функция ожидает `ids`/`doc_ids`
3. **QueryPlannerService._generate_plan**: Метод вызывается, но не существует
4. **Validation error step=0**: Verification refinement нарушает схему Pydantic (step >= 1)
5. **Context window overflow**: Prompt от compose_context превышает лимит модели (6112 > 4096 токенов)

---

## Диагностика

### Трейс ошибок из логов

```
Строка 806-809: ERROR - Error in _verify_answer: 1 validation error for AgentAction
step
  Input should be greater than or equal to 1 [type=greater_than_equal, input_value=0, input_type=int]
```

```
Строка 827-828: "compose_context() got an unexpected keyword argument 'query'"
Строка 837-838: "compose_context() got an unexpected keyword argument 'hits'"
Строка 847-848: "fetch_docs() got an unexpected keyword argument 'hit_ids'"
```

```
Строка 948-952: Ошибка генерации шага 8: Requested tokens (6112) exceed context window of 4096
Строка 956-960: QueryPlannerService.make_plan failed: 'QueryPlannerService' object has no attribute '_generate_plan'
```

### Корневые причины

#### 1. **Системный промпт не синхронизирован с реальными сигнатурами**

**Файл:** `src/services/agent_service.py:46-83`

Промпт описывает инструменты неточно:

```python
# Текущее описание (НЕВЕРНО):
- compose_context: собирает контекст из документов с цитированием
- fetch_docs: получает документы по списку ID
```

**Реальные сигнатуры:**

```python
# src/services/tools/compose_context.py:14-19
def compose_context(
    docs: List[Dict[str, Any]],
    max_tokens_ctx: int = 1800,
    citation_format: str = "footnotes",
    enable_lost_in_middle_mitigation: bool = True,
) -> Dict[str, Any]

# src/services/tools/fetch_docs.py:11-16
def fetch_docs(
    retriever: Retriever,
    ids: Optional[List[str]] = None,
    window: Optional[List[int]] = None,
    doc_ids: Optional[List[str]] = None,
) -> Dict[str, Any]
```

**Решение:** Обновить системный промпт с точными параметрами инструментов.

---

#### 2. **Нормализация параметров частично реализована в agent_service.py**

**Файл:** `src/services/agent_service.py:296-314, 640-676`

Логика нормализации для `compose_context` существует (строки 640-676), но:
- LLM всё равно генерирует лишние параметры (`query`, `hits`, `hit_ids`)
- После нормализации параметры `docs` корректны, но лог показывает ошибку **до** нормализации

**Проблема:** Нормализация срабатывает только внутри `_execute_action`, но LLM генерирует неверный формат ещё на этапе `_generate_step`.

**Решение:** 
1. Явно документировать нормализацию в промпте
2. Добавить ранний парсинг и нормализацию **перед** вызовом `tool_runner.run`
3. Использовать fallback для невалидных параметров

---

#### 3. **QueryPlannerService._generate_plan отсутствует**

**Файл:** `src/services/query_planner_service.py:360`

Метод вызывается, но не реализован:

```python
try:
    plan = self._generate_plan(query)  # ← MISSING METHOD
except Exception as exc:
    logger.error("QueryPlannerService.make_plan failed: %s", exc, exc_info=True)
    raise
```

**Поиск grep:** Метод `_generate_plan` не найден ни в одном файле.

**Предполагаемая реализация:** Метод должен вызывать LLM с GBNF грамматикой и парсить JSON план.

**Решение:** Восстановить/реализовать метод `_generate_plan` на основе существующей логики:
1. Использовать `_build_prompt` (строка 374)
2. Вызывать `self.llm` с GBNF грамматикой (если включено)
3. Парсить JSON с fallback на `_fallback_plan`
4. Применять `post_validate`

---

#### 4. **Validation error step=0 в verify**

**Файл:** `src/services/agent_service.py:788-798`

Verification вызывает tool_runner с `step=0`:

```python
result = self.tool_runner.run(
    request_id_for_verify,
    0,  # ← INVALID: step must be >= 1
    ToolRequest(
        tool="verify",
        input={...},
    ),
)
```

**Схема:** `src/schemas/agent.py:42`
```python
step: int = Field(..., ge=1, description="Порядковый номер шага")
```

**Решение:** Использовать корректный step (текущий шаг или -1 для системных вызовов, но тогда изменить схему).

---

#### 5. **Context window overflow (6112 > 4096)**

**Файл:** `src/services/agent_service.py:530-540`

Prompt генерируется из:
```python
prompt = f"""Системная инструкция: {self.system_prompt}

Контекст разговора:
{history_text}

Продолжи рассуждение, следуя формату ReAct (Thought/Action/Observation или FinalAnswer):"""
```

**Проблема:**
- `self.system_prompt` ~2000 символов
- `history_text` накапливается с каждым шагом (8 шагов × ~600 символов = 4800 символов)
- `compose_context` возвращает до 7200 символов (1800 токенов × 4 символа/токен)

**Лимит модели:** 4096 токенов (~16384 символа)

**Решение:**
1. Сократить `max_tokens_ctx` в `compose_context` до 1200 токенов
2. Реализовать скользящее окно для `conversation_history` (последние 5 шагов)
3. Сжимать системный промпт (убрать примеры/детали)

---

## Комплексный план исправления

### Фаза 1: Восстановление QueryPlannerService._generate_plan

**Файл:** `src/services/query_planner_service.py`

**Действие:** Реализовать отсутствующий метод после строки 331:

```python
def _generate_plan(self, query: str) -> SearchPlan:
    """Генерирует план поиска через LLM с GBNF грамматикой."""
    prompt = self._build_prompt(query)
    
    try:
        # Вызов LLM с GBNF грамматикой (если включено)
        if self._gbnf_grammar:
            response = self.llm(
                prompt,
                max_tokens=512,
                temperature=0.1,
                top_p=0.95,
                grammar=self._gbnf_grammar,
                stop=["</s>", "\n\n"],
            )
        else:
            response = self.llm(
                prompt,
                max_tokens=512,
                temperature=0.1,
                top_p=0.95,
                stop=["</s>", "\n\n"],
            )
        
        raw_text = response["choices"][0]["text"].strip()
        
        # Извлекаем JSON блок
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not json_match:
            logger.warning("No JSON found in LLM response, using fallback")
            return self._fallback_plan(query)
        
        raw_json = json.loads(json_match.group(0))
        
        # Применяем post_validate для санитизации
        plan = self.post_validate(raw_json, query, self.settings)
        
        # Кешируем результат
        if self.settings.enable_cache:
            cache_key = f"plan:{hash(query)}"
            self._plan_cache[cache_key] = plan
        
        return plan
        
    except Exception as e:
        logger.error(f"_generate_plan failed for query='{query[:80]}': {e}")
        return self._fallback_plan(query)
```

**Тест:** Вызвать `planner.make_plan("Что известно про новые модели LLM?")` и проверить, что возвращается `SearchPlan` без ошибок.

---

### Фаза 2: Синхронизация системного промпта с реальными сигнатурами

**Файл:** `src/services/agent_service.py:46-83`

**Действие:** Обновить системный промпт с точными контрактами:

```python
self.system_prompt = """Ты — ReAct агент, который помогает отвечать на вопросы пользователей, используя доступные инструменты.

ФОРМАТ РАБОТЫ:
Мысли пошагово, используя следующий формат:

Thought: [твоё размышление о том, что нужно сделать]
Action: [название_инструмента] {"param": "value"}
Observation: [результат выполнения инструмента]

Повторяй этот цикл до получения достаточной информации для ответа.

Когда у тебя есть достаточно информации для полного ответа:
FinalAnswer: [твой итоговый ответ пользователю]

ДОСТУПНЫЕ ИНСТРУМЕНТЫ И ИХ КОНТРАКТЫ:

1. router_select: выбирает оптимальный маршрут поиска
   Параметры: {"query": "string"}

2. query_plan: создает план поиска
   Параметры: {"query": "string"}

3. search: выполняет гибридный поиск по коллекции
   Параметры: {"queries": ["string"], "route": "hybrid|bm25|dense", "k": int}
   Возвращает: {"hits": [{"id": "...", "text": "...", "metadata": {...}}], "route_used": "..."}

4. rerank: переранжирует документы
   Параметры: {"query": "string", "docs": ["text1", "text2", ...], "top_n": int}

5. fetch_docs: получает документы по списку ID
   Параметры: {"ids": ["id1", "id2", ...]}

6. compose_context: собирает контекст из документов с цитированием
   Параметры: {"hit_ids": ["id1", "id2", ...]}
   ВАЖНО: Используй hit_ids из последних результатов search. Система автоматически преобразует ids в docs.
   Возвращает: {"prompt": "...", "citations": [...], "citation_coverage": float}

7. verify: проверяет утверждения через поиск в базе знаний
   Параметры: {"query": "string", "claim": "string", "k": int}

ПРАВИЛА:
1. Всегда начинай с Thought
2. Используй инструменты для получения информации
3. После search сохраняй hit_ids для compose_context
4. Передавай ТОЛЬКО указанные параметры инструментам
5. Завершай с FinalAnswer

ДЕТЕРМИНИРОВАННАЯ ЛОГИКА СИСТЕМЫ:
- После compose_context система автоматически проверит покрытие цитирований (>=80%)
- Если покрытие недостаточно, система выполнит дополнительный раунд поиска
- Перед финальным ответом система может верифицировать утверждения
- Максимум 1 дополнительный раунд поиска для избежания бесконечных циклов

Будь точным, логичным и полезным."""
```

**Критические изменения:**
1. Добавлены точные контракты для каждого инструмента
2. Явно указано, что `compose_context` принимает `hit_ids` (нормализация внутри)
3. Убраны лишние примеры (сокращение на ~500 символов)

---

### Фаза 3: Улучшение нормализации параметров

**Файл:** `src/services/agent_service.py:640-676`

**Действие:** Рефакторить нормализацию в отдельный метод для переиспользования:

Добавить после строки 687:

```python
def _normalize_tool_params(
    self, tool_name: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Нормализует параметры инструментов для совместимости."""
    
    if tool_name == "compose_context":
        # Убираем лишние параметры, которые LLM может генерировать
        params.pop("query", None)
        params.pop("hits", None)
        
        # Извлекаем hit_ids
        hit_ids: List[str] = params.pop("hit_ids", []) or []
        
        # Получаем последние hits из поиска
        last_hits = getattr(self, "_last_search_hits", [])
        hits_by_id = {
            hit.get("id"): hit
            for hit in last_hits
            if isinstance(hit, dict) and hit.get("id")
        }
        
        # Выбираем документы по hit_ids или берём все последние hits
        selected_hits: List[Dict[str, Any]] = []
        if hit_ids:
            for hid in hit_ids:
                match = hits_by_id.get(hid)
                if match:
                    selected_hits.append(match)
        
        if not selected_hits:
            selected_hits = [hit for hit in last_hits if isinstance(hit, dict)]
        
        # Нормализуем в формат docs
        normalized_docs: List[Dict[str, Any]] = []
        for doc in selected_hits:
            normalized_docs.append({
                "id": doc.get("id"),
                "text": (
                    doc.get("text")
                    or doc.get("snippet")
                    or doc.get("meta", {}).get("text")
                    or doc.get("metadata", {}).get("text")
                    or ""
                ),
                "metadata": doc.get("metadata") or doc.get("meta", {}),
            })
        
        params["docs"] = normalized_docs
        
        # Ограничиваем max_tokens_ctx для предотвращения overflow
        params.setdefault("max_tokens_ctx", 1200)
    
    elif tool_name == "fetch_docs":
        # Нормализация hit_ids → ids
        hit_ids = params.pop("hit_ids", None)
        doc_ids = params.pop("doc_ids", None)
        if hit_ids is not None and "ids" not in params:
            params["ids"] = hit_ids
        elif doc_ids is not None and "ids" not in params:
            params["ids"] = doc_ids
    
    elif tool_name == "rerank":
        # Автоматически добавляем docs из последних hits если не переданы
        if "docs" not in params:
            hits_payload = params.pop("hits", None)
            if not hits_payload:
                hits_payload = getattr(self, "_last_search_hits", [])
            if hits_payload:
                params["docs"] = [
                    item.get("snippet") or item.get("text") or ""
                    for item in hits_payload
                    if item
                ]
        params.setdefault("query", self._current_query or "")
    
    return params
```

**Обновить `_execute_action` (строка 578-687):**

Заменить строки 640-676 на:

```python
# Нормализуем параметры через единый метод
params = self._normalize_tool_params(tool_name, params)
```

**Обновить блок `elif action_text.lower().startswith("compose_context"):` (строки 296-314):**

Упростить до:

```python
elif action_text.lower().startswith("compose_context"):
    # Нормализация будет выполнена в _execute_action через _normalize_tool_params
    pass
```

---

### Фаза 4: Исправление validation error step=0

**Файл:** `src/services/agent_service.py:788-798`

**Действие:** Передавать корректный step при verify:

```python
# Вместо:
result = self.tool_runner.run(
    request_id_for_verify,
    0,  # ← ОШИБКА
    ToolRequest(...)
)

# Использовать:
# Получаем текущий step из контекста или используем фейковый step для системных вызовов
verify_step = self._current_step if hasattr(self, '_current_step') else 1
result = self.tool_runner.run(
    request_id_for_verify,
    verify_step,
    ToolRequest(...)
)
```

**Добавить атрибут `_current_step` в класс AgentService:**

В `__init__` (строка 32-43):

```python
self._current_request_id: Optional[str] = None
self._current_step: int = 1  # ← Добавить
```

Обновлять в основной петле (после строки 453):

```python
step += 1
self._current_step = step  # ← Добавить
```

**Альтернативное решение (если нужны системные вызовы вне петли):**

Изменить схему `AgentAction` для поддержки step=0:

```python
# src/schemas/agent.py:42
step: int = Field(..., ge=0, description="Порядковый номер шага (0 для системных вызовов)")
```

**Рекомендация:** Использовать первое решение (корректный step), т.к. изменение схемы может сломать существующую логику.

---

### Фаза 5: Предотвращение context window overflow

**Файл:** `src/services/agent_service.py:511-541`

**Действие 1:** Реализовать скользящее окно для conversation_history:

После строки 517:

```python
# Собираем промпт
history_text = "\n".join(conversation_history)

# ДОБАВИТЬ: Ограничение истории скользящим окном
MAX_HISTORY_ITEMS = 10  # Последние 5 Thought-Action-Observation триплетов
if len(conversation_history) > MAX_HISTORY_ITEMS:
    # Сохраняем первый элемент (исходный запрос) и последние MAX_HISTORY_ITEMS
    conversation_history = [conversation_history[0]] + conversation_history[-MAX_HISTORY_ITEMS:]
    history_text = "\n".join(conversation_history)
```

**Действие 2:** Сократить системный промпт (см. Фазу 2) — экономия ~500 символов.

**Действие 3:** Снизить max_tokens_ctx в compose_context до 1200 (см. Фазу 3).

**Действие 4:** Увеличить context_size для основной LLM через переменную окружения:

```bash
# .env или docker-compose.yml
LLM_CONTEXT_SIZE=8192  # Вместо 4096
```

**Обоснование:** Qwen2.5-7B-Instruct поддерживает контекст до 32k токенов, но в логах установлено 4096. Увеличение до 8192 даст запас без значительного увеличения latency.

**Обновить `src/core/deps.py:96`:**

```python
n_ctx = int(os.getenv("LLM_CONTEXT_SIZE", "8192"))  # Вместо "4096"
```

---

### Фаза 6: Тестирование и валидация

**Создать тест:** `src/tests/test_agent_tool_contracts.py`

```python
import pytest
from services.agent_service import AgentService
from services.tools.compose_context import compose_context
from services.tools.fetch_docs import fetch_docs
from services.query_planner_service import QueryPlannerService
from schemas.agent import AgentRequest, ToolRequest
from core.deps import get_agent_service, get_query_planner

@pytest.fixture
def agent_service():
    return get_agent_service()

@pytest.fixture
def query_planner():
    return get_query_planner()

class TestToolContracts:
    """Тесты на соответствие контрактов инструментов."""
    
    def test_compose_context_accepts_docs_only(self):
        """compose_context должен принимать только docs параметр."""
        docs = [
            {"id": "1", "text": "test text", "metadata": {}},
            {"id": "2", "text": "another text", "metadata": {}},
        ]
        
        result = compose_context(docs=docs, max_tokens_ctx=1200)
        
        assert "prompt" in result
        assert "citations" in result
        assert "citation_coverage" in result
        assert len(result["citations"]) == 2
    
    def test_compose_context_rejects_invalid_params(self):
        """compose_context должен игнорировать лишние параметры."""
        docs = [{"id": "1", "text": "test", "metadata": {}}]
        
        # Должны быть проигнорированы: query, hits, hit_ids
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            compose_context(docs=docs, query="test", hits=10)
    
    def test_fetch_docs_accepts_ids(self, agent_service):
        """fetch_docs должен принимать ids параметр."""
        # Mock retriever
        from unittest.mock import Mock
        mock_retriever = Mock()
        mock_retriever.get_by_ids.return_value = [
            {"id": "1", "text": "doc1", "metadata": {}}
        ]
        
        result = fetch_docs(retriever=mock_retriever, ids=["1"])
        
        assert "docs" in result
        assert len(result["docs"]) == 1
    
    def test_agent_normalize_tool_params_compose_context(self, agent_service):
        """AgentService должен нормализовать параметры compose_context."""
        # Имитируем последние hits
        agent_service._last_search_hits = [
            {"id": "1", "text": "test1", "metadata": {}},
            {"id": "2", "text": "test2", "metadata": {}},
        ]
        
        # LLM генерирует невалидные параметры
        params = {
            "query": "test query",
            "hits": 10,
            "hit_ids": ["1", "2"],
            "docs": [],  # Пустой, должен быть заполнен
        }
        
        normalized = agent_service._normalize_tool_params("compose_context", params)
        
        assert "query" not in normalized
        assert "hits" not in normalized
        assert "hit_ids" not in normalized
        assert "docs" in normalized
        assert len(normalized["docs"]) == 2
        assert normalized["max_tokens_ctx"] == 1200
    
    def test_query_planner_generate_plan_exists(self, query_planner):
        """QueryPlannerService._generate_plan должен существовать."""
        assert hasattr(query_planner, '_generate_plan')
        assert callable(getattr(query_planner, '_generate_plan'))
    
    def test_query_planner_make_plan_success(self, query_planner):
        """QueryPlannerService.make_plan должен возвращать SearchPlan."""
        plan = query_planner.make_plan("Что известно про новые модели LLM?")
        
        assert plan is not None
        assert hasattr(plan, 'normalized_queries')
        assert hasattr(plan, 'must_phrases')
        assert len(plan.normalized_queries) >= 1
    
    def test_agent_verify_step_validation(self, agent_service):
        """_verify_answer должен передавать корректный step >= 1."""
        agent_service._current_step = 5
        agent_service._current_request_id = "test-req"
        
        # Mock tool_runner
        from unittest.mock import Mock
        mock_runner = Mock()
        mock_output = Mock()
        mock_output.ok = True
        mock_output.data = {"verified": True, "confidence": 0.9}
        mock_action = Mock()
        mock_action.output = mock_output
        mock_runner.run.return_value = mock_action
        
        agent_service.tool_runner = mock_runner
        
        import asyncio
        result = asyncio.run(agent_service._verify_answer(
            "test answer",
            ["Human: test query"]
        ))
        
        # Проверяем, что step >= 1
        call_args = mock_runner.run.call_args
        assert call_args[0][1] >= 1  # Второй аргумент - step
    
    def test_agent_context_window_limit(self, agent_service):
        """conversation_history должна быть ограничена скользящим окном."""
        # Имитируем длинную историю (20 элементов)
        history = [f"Item {i}" for i in range(20)]
        
        # После применения скользящего окна должно остаться <= 11 элементов
        # (1 исходный запрос + 10 последних)
        # Проверим логику внутри _generate_step
        
        agent_service._current_request_id = "test-req"
        
        # Mock LLM factory
        from unittest.mock import Mock
        mock_llm = Mock()
        mock_response = {
            "choices": [{"text": "Thought: test"}]
        }
        mock_llm.return_value = mock_response
        agent_service.llm_factory = lambda: mock_llm
        
        import asyncio
        asyncio.run(agent_service._generate_step(history, "test-req", 1))
        
        # Проверяем, что prompt не превышает разумный размер
        call_args = mock_llm.call_args
        prompt = call_args[0][0]
        
        # Примерная оценка: системный промпт ~3000 символов + история
        # Не должно превышать ~20000 символов (5000 токенов)
        assert len(prompt) < 20000, f"Prompt too long: {len(prompt)} chars"
```

**Запуск тестов:**

```bash
pytest src/tests/test_agent_tool_contracts.py -v
```

**Регрессионный тест с реальным сценарием:**

```python
# src/tests/test_agent_integration.py

import pytest
from httpx import AsyncClient
from main import app

@pytest.mark.asyncio
async def test_agent_full_flow():
    """Интеграционный тест: полный ReAct цикл без ошибок."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Предполагается, что коллекция ml_mult5 уже загружена
        
        # Выбор коллекции
        resp = await client.post("/v1/collections/select", json={"collection": "ml_mult5"})
        assert resp.status_code == 200
        
        # Запрос к агенту
        resp = await client.post(
            "/v1/agent/stream",
            json={
                "query": "Что известно про новые модели LLM?",
                "max_steps": 8,
            },
            headers={"Authorization": "Bearer test-token"},
        )
        
        assert resp.status_code == 200
        
        # Парсим SSE события
        events = []
        for line in resp.text.split("\n"):
            if line.startswith("data: "):
                import json
                events.append(json.loads(line[6:]))
        
        # Проверки
        assert len(events) > 0, "No events received"
        
        # Должно быть финальное событие
        final_events = [e for e in events if e.get("type") == "final"]
        assert len(final_events) == 1, "Expected exactly one final event"
        
        final = final_events[0]["data"]
        assert "answer" in final
        assert "request_id" in final
        
        # НЕ должно быть ошибок с контрактами
        error_events = [
            e for e in events
            if e.get("type") == "observation" and "Ошибка" in e.get("data", {}).get("content", "")
        ]
        
        contract_errors = [
            e for e in error_events
            if "unexpected keyword argument" in e["data"]["content"]
        ]
        
        assert len(contract_errors) == 0, f"Contract errors found: {contract_errors}"
        
        # НЕ должно быть validation errors
        validation_errors = [
            e for e in error_events
            if "validation error" in e["data"]["content"].lower()
        ]
        
        assert len(validation_errors) == 0, f"Validation errors found: {validation_errors}"
        
        # НЕ должно быть context window overflow
        overflow_errors = [
            e for e in error_events
            if "exceed context window" in e["data"]["content"]
        ]
        
        assert len(overflow_errors) == 0, f"Context window overflow: {overflow_errors}"
```

---

## Чеклист изменений

### Файлы для правки

1. ✅ **src/services/query_planner_service.py**
   - [ ] Добавить метод `_generate_plan` после строки 331
   - [ ] Тест: `pytest src/tests/test_query_planner.py::test_generate_plan_exists`

2. ✅ **src/services/agent_service.py**
   - [ ] Обновить системный промпт (строки 46-83) с точными контрактами
   - [ ] Добавить метод `_normalize_tool_params` после строки 687
   - [ ] Обновить `_execute_action` для использования `_normalize_tool_params`
   - [ ] Упростить блоки нормализации (строки 296-314, 640-676)
   - [ ] Добавить атрибут `_current_step` в `__init__` (строка 43)
   - [ ] Обновлять `_current_step` в основной петле (после строки 453)
   - [ ] Исправить `_verify_answer` для использования `_current_step` вместо 0 (строка 790)
   - [ ] Добавить скользящее окно для `conversation_history` в `_generate_step` (после строки 517)
   - [ ] Тесты:
     - `pytest src/tests/test_agent_tool_contracts.py::test_agent_normalize_tool_params_compose_context`
     - `pytest src/tests/test_agent_tool_contracts.py::test_agent_verify_step_validation`
     - `pytest src/tests/test_agent_tool_contracts.py::test_agent_context_window_limit`

3. ✅ **src/core/deps.py**
   - [ ] Обновить `LLM_CONTEXT_SIZE` с 4096 на 8192 (строка 96)
   - [ ] Тест: проверить через `get_llm().n_ctx == 8192`

4. ✅ **src/tests/test_agent_tool_contracts.py** (новый файл)
   - [ ] Создать файл с unit-тестами на контракты инструментов
   - [ ] Запуск: `pytest src/tests/test_agent_tool_contracts.py -v`

5. ✅ **src/tests/test_agent_integration.py** (новый файл)
   - [ ] Создать файл с интеграционным тестом полного ReAct цикла
   - [ ] Запуск: `pytest src/tests/test_agent_integration.py -v`

6. ✅ **docker-compose.yml** или **.env**
   - [ ] Добавить/обновить `LLM_CONTEXT_SIZE=8192` в environment для контейнера api

### Порядок выполнения

1. **Фаза 1:** Восстановление `_generate_plan` (критично для работы плана)
2. **Фаза 2:** Синхронизация системного промпта (предотвращает генерацию невалидных параметров)
3. **Фаза 3:** Улучшение нормализации (исправляет ошибки с контрактами)
4. **Фаза 4:** Исправление step=0 (validation error)
5. **Фаза 5:** Предотвращение overflow (context window)
6. **Фаза 6:** Тестирование и валидация

### Критерии приёмки

✅ Все тесты из `test_agent_tool_contracts.py` проходят  
✅ Интеграционный тест `test_agent_full_flow` проходит без ошибок контрактов  
✅ Нет логов с "unexpected keyword argument" при работе агента  
✅ Нет validation errors "step should be greater than or equal to 1"  
✅ Нет ошибок "Requested tokens exceed context window"  
✅ QueryPlannerService.make_plan работает без "object has no attribute '_generate_plan'"  

---

## Риски и mitigation

### Риск 1: Изменение системного промпта может сломать существующее поведение
**Вероятность:** Средняя  
**Mitigation:** 
- Сохранить старый промпт в `self.system_prompt_legacy`
- Добавить feature flag `USE_LEGACY_SYSTEM_PROMPT` для отката
- A/B тест на 10% трафика перед полным раскатыванием

### Риск 2: LLM всё равно может генерировать невалидные параметры
**Вероятность:** Средняя  
**Mitigation:**
- Нормализация параметров в `_normalize_tool_params` отработает fallback
- Добавить логирование всех невалидных параметров для анализа
- Рассмотреть GBNF грамматику для генерации Action (future work)

### Риск 3: Увеличение LLM_CONTEXT_SIZE до 8192 может увеличить latency
**Вероятность:** Низкая  
**Mitigation:**
- Qwen2.5-7B-Instruct оптимизирована для контекста до 32k
- Реальное использование ~6000 токенов (не критично)
- Мониторинг latency в production: `llama_perf_context_print`

### Риск 4: Метод _generate_plan может работать нестабильно
**Вероятность:** Средняя  
**Mitigation:**
- Фолбэк на `_fallback_plan` при любых ошибках
- Кеширование успешных планов (TTL 10 мин)
- Логирование всех случаев использования fallback для анализа

---

## Метрики успеха

### До исправления (baseline из логов)
- ❌ 5 ошибок "unexpected keyword argument" на 1 запрос
- ❌ 1 validation error step=0
- ❌ 1 context window overflow на шаге 8
- ❌ 1 AttributeError '_generate_plan' в fallback
- ⚠️ Fallback через QAService в 100% случаев при max_steps

### После исправления (target)
- ✅ 0 ошибок "unexpected keyword argument"
- ✅ 0 validation errors
- ✅ 0 context window overflow
- ✅ 0 AttributeError '_generate_plan'
- ✅ Fallback через QAService < 5% случаев
- ✅ Успешное завершение ReAct петли >= 90% случаев

### Мониторинг
```python
# Добавить в src/services/agent_service.py
logger.info(
    "ReAct петля завершена | request_id=%s | steps=%s | coverage=%.2f | refinements=%s | fallback=%s",
    request_id,
    step,
    agent_state.coverage,
    agent_state.refinement_count,
    used_fallback
)
```

---

## Дополнительные улучшения (future work)

### 1. GBNF грамматика для Action генерации
**Проблема:** LLM может генерировать произвольный JSON в Action  
**Решение:** Использовать GBNF грамматику для строгой генерации Action блоков  
**Пример:**
```python
action_grammar = """
root ::= action
action ::= toolname ws "{" ws params ws "}"
toolname ::= "search" | "compose_context" | "fetch_docs" | ...
params ::= param | param ws "," ws params
param ::= key ws ":" ws value
key ::= "\"" [a-z_]+ "\""
value ::= string | number | array
...
"""
```

### 2. Динамическая валидация параметров через Pydantic
**Проблема:** Ручная нормализация подвержена ошибкам  
**Решение:** Определить Pydantic схемы для каждого инструмента  
**Пример:**
```python
class ComposeContextInput(BaseModel):
    hit_ids: List[str] = Field(default_factory=list)
    max_tokens_ctx: int = Field(default=1200, ge=100, le=3000)

class FetchDocsInput(BaseModel):
    ids: List[str] = Field(..., min_items=1)

# В tool_runner.run:
schema = TOOL_SCHEMAS.get(req.tool)
if schema:
    validated_input = schema(**req.input)
    req.input = validated_input.dict()
```

### 3. Автоматическое обновление системного промпта из docstrings
**Проблема:** Системный промпт может устареть при изменении сигнатур  
**Решение:** Генерировать описания инструментов из docstrings  
**Пример:**
```python
def compose_context(docs: List[Dict[str, Any]], max_tokens_ctx: int = 1800) -> Dict[str, Any]:
    """
    Собирает контекст из документов с цитированием.
    
    Args:
        docs: Список документов [{id, text, metadata}]
        max_tokens_ctx: Максимальное количество токенов контекста
    
    Returns:
        {"prompt": str, "citations": List, "citation_coverage": float}
    """
    ...

# Генерация промпта:
system_prompt = build_system_prompt_from_tools(tool_runner.get_all_tools())
```

---

## Заключение

Данный bugfix устраняет **5 критических проблем** в контрактах инструментов агента:

1. ✅ Восстановление отсутствующего метода `_generate_plan`
2. ✅ Синхронизация системного промпта с реальными сигнатурами
3. ✅ Рефакторинг нормализации параметров
4. ✅ Исправление validation error step=0
5. ✅ Предотвращение context window overflow

**Ожидаемый результат:** Успешное завершение ReAct петли без ошибок контрактов в >= 90% случаев.

**Время на реализацию:** 4-6 часов (включая тестирование)

**Приоритет:** Критический (блокирует работу агента)

---

## Ссылки

- Логи с ошибками: строки 780-962 терминала
- Код агента: `src/services/agent_service.py:1-850`
- Инструменты: `src/services/tools/*.py`
- Схемы: `src/schemas/agent.py`
- Зависимости: `src/core/deps.py:469-570`
- Research: `docs/ai/research/react-rag-research.md`
- ADR: `docs/ai/adr/ADR-20250103-agentic-react-rag.md`

