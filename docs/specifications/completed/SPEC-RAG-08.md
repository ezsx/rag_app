# SPEC-RAG-08: Native Function Calling + Qwen3-30B-A3B MoE

> **Версия:** 1.0
> **Дата:** 2026-03-17
> **Статус:** Draft
> **Цель:** Заменить текстовый ReAct-парсинг (regex на Thought/Action/FinalAnswer) на native function calling
> через `/v1/chat/completions` с `tools` schema. Обновить LLM на Qwen3-30B-A3B MoE Q4_K_M.
> **Источники:** R-07 (Block 4: function calling), R-08 (Qwen3-30B-A3B MoE recommendation),
> [llama.cpp function-calling docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)

---

## 0. Мотивация

### Почему ReAct regex сломан

Qwen3 документация: *"For reasoning models like Qwen3, it is not recommended to use tool call template
based on stopwords, such as ReAct, because the model may output stopwords in the thought section."*

Текущие проблемы (все задокументированы в коммитах `4c89a93`, `53163ce`, `6663307`):
1. Thinking preamble (англоязычный CoT без `<think>` тегов) ломает regex-парсер
2. Пустые шаги: LLM генерирует thinking → parser возвращает (None, None, None) → step wasted
3. Forced FinalAnswer: пришлось программно форсить ответ вместо нормального agent flow
4. Refinement loop: LLM игнорирует промпт-инструкции и бесконечно вызывает search

### Что даёт native function calling

- Сервер сам разделяет reasoning и tool calls (через Jinja2 chat template)
- Structured output: `tool_calls[0].name` + `tool_calls[0].arguments` (JSON)
- Tool results: `role: "tool"` messages — стандарт OpenAI API
- Не нужен regex, strip thinking, forced FinalAnswer
- Parallel tool calls (опционально)

### Почему Qwen3-30B-A3B MoE

Per R-08: 30B total / 3B active per token. ArenaHard 91.0. На V100 32GB:
~18GB weights + ~5GB KV (16K p=2 Q8) = ~20GB. Generation: 80-120 tok/s (3B active).
Быстрее текущего 8B при качестве 30B-класса.

---

## 1. Изменяемые файлы

| Файл | Характер |
|------|----------|
| `src/adapters/llm/llama_server_client.py` | Добавить `chat_completion()` метод |
| `src/services/agent_service.py` | Полный rewrite main loop + tool dispatch |
| `src/core/settings.py` | Обновить defaults (model name, sampling params) |
| `src/core/deps.py` | Обновить `get_agent_service()` — tools как JSON schema |
| `deploy/compose/compose.dev.yml` | Обновить LLM env vars |
| `agent_context/core/always_on.md` | Обновить модель и команду запуска |
| `AGENTS.md` | Обновить модель |
| `docs/ai/agent_technical_spec.md` | Обновить архитектуру агента |
| `docs/ai/project_brief.md` | Обновить модель |
| `docs/ai/modules/src/services/agent_service.py.md` | Обновить |
| `docs/ai/modules/src/adapters/llm/llama_server_client.py.md` | Обновить (если существует) |

### Что НЕ менять

- `src/services/tools/*` — сами tool-функции остаются (search, rerank, compose_context и т.д.)
- `src/services/tools/tool_runner.py` — ToolRunner остаётся для изоляции и таймаутов
- `src/api/v1/endpoints/agent.py` — SSE endpoint не меняется
- `src/schemas/agent.py` — AgentRequest/AgentStepEvent не меняются
- SSE event contract: `thought/tool_invoked/observation/citations/final` — НЕ ломать

---

## 2. `src/adapters/llm/llama_server_client.py` — новый метод

Добавить `chat_completion()` рядом с существующим `__call__()` (который остаётся для qa_service
и query_planner, использующих `/v1/completions`).

```python
def chat_completion(
    self,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    presence_penalty: float = 1.5,
    stop: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Вызывает /v1/chat/completions с поддержкой function calling.

    Returns:
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "текст ответа" | null,
                    "tool_calls": [{"name": "...", "arguments": "{...}"}] | null
                },
                "finish_reason": "stop" | "tool"
            }],
            "usage": {...}
        }
    """
    payload: Dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
    }
    if tools:
        payload["tools"] = tools
    if stop:
        payload["stop"] = stop
    if seed is not None:
        payload["seed"] = seed

    resp = self._session.post(
        f"{self.base_url}/v1/chat/completions",
        json=payload,
        timeout=self.timeout,
    )
    resp.raise_for_status()
    return resp.json()
```

**Sampling parameters** (per R-08, Qwen3 non-thinking recommendations):
- `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`
- Не использовать `temperature=0` — вызывает repetition loops в Qwen3

---

## 3. Tools Schema — JSON определения

Определить в `agent_service.py` или в отдельном файле `src/services/tools/tool_schemas.py`.

Каждый tool описывается как OpenAI-совместимая JSON schema:

```python
AGENT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_plan",
            "description": "Декомпозирует сложный запрос на 3-5 подзапросов с фильтрами. "
                           "Вызывай первым для планирования поиска.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Исходный запрос пользователя"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Выполняет гибридный поиск (dense+sparse RRF) по коллекции Telegram-каналов. "
                           "Возвращает список документов с ID, текстом и метаданными.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список поисковых запросов (1-5 штук)"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Количество результатов (default: 10)",
                        "default": 10
                    }
                },
                "required": ["queries"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rerank",
            "description": "Переранжирует документы по реальной семантической близости к запросу. "
                           "Вызывай после search для повышения точности.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Запрос для ранжирования"
                    },
                    "docs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Тексты документов для ранжирования"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Количество лучших документов (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query", "docs"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compose_context",
            "description": "Собирает контекст из документов с цитированием [1], [2] и т.д. "
                           "Вызывай после search/rerank. Возвращает prompt с citations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hit_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "ID документов из результатов search"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Формирует финальный ответ пользователю. ОБЯЗАТЕЛЬНО вызывай после compose_context. "
                           "Ответ должен быть на русском языке и ссылаться на источники [1], [2] и т.д.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Текст ответа на русском языке с цитатами [1], [2]"
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Номера использованных источников [1, 2, 3]"
                    }
                },
                "required": ["answer", "sources"]
            }
        }
    }
]
```

**Примечание:** `router_select` убрать — per R-07, если один search backend, это лишний шаг.
`verify` оставить опционально, но убрать из основного flow (вызывать только системно после final_answer).
`fetch_docs` — вызывается внутри compose_context автоматически, не нужен как отдельный tool для LLM.

Итого tools для LLM: **query_plan, search, rerank, compose_context, final_answer** (5 tools).
verify и fetch_docs остаются в ToolRunner для системных вызовов.

---

## 4. `src/services/agent_service.py` — rewrite main loop

### 4.1 Системный промпт

```python
SYSTEM_PROMPT = """Ты — RAG-агент для поиска информации в базе новостей из Telegram-каналов.

ПОРЯДОК РАБОТЫ:
1. query_plan — декомпозируй запрос на подзапросы
2. search — найди документы по подзапросам
3. rerank — переранжируй документы (query=исходный запрос, docs=тексты из search)
4. compose_context — собери контекст из лучших документов
5. final_answer — дай ответ СТРОГО на основе контекста

ПРАВИЛА:
- Отвечай ТОЛЬКО на русском языке
- Каждое утверждение в ответе ОБЯЗАТЕЛЬНО подкрепляй ссылкой [1], [2] и т.д.
- НЕ придумывай факты — если в контексте нет информации, скажи об этом
- В final_answer перечисли номера использованных источников в поле sources
- После compose_context ОБЯЗАТЕЛЬНО вызови final_answer (не search повторно)
"""
```

### 4.2 Main loop (псевдокод)

```python
async def stream_agent_response(self, request: AgentRequest):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request.query},
    ]

    for step in range(1, max_steps + 1):
        yield step_started_event(step)

        # Вызов LLM через chat/completions с tools
        llm = self.llm_factory()
        response = llm.chat_completion(
            messages=messages,
            tools=AGENT_TOOLS,
            max_tokens=self.settings.agent_tool_max_tokens,
            temperature=0.7,
            top_p=0.8,
        )

        choice = response["choices"][0]
        message = choice["message"]
        finish_reason = choice["finish_reason"]

        # Если LLM сгенерировал текст (мысль или финальный ответ)
        if message.get("content"):
            yield thought_event(message["content"], step)
            # Если finish_reason == "stop" и нет tool_calls → это финальный ответ
            if finish_reason == "stop" and not message.get("tool_calls"):
                yield final_event(message["content"], step)
                return

        # Если LLM вызвал tool
        if message.get("tool_calls"):
            # Добавляем assistant message в историю
            messages.append(message)

            for tool_call in message["tool_calls"]:
                tool_name = tool_call["name"]
                tool_args = json.loads(tool_call["arguments"])

                yield tool_invoked_event(tool_name, tool_args, step)

                # Нормализация параметров + выполнение через ToolRunner
                normalized_args = self._normalize_tool_params(tool_name, tool_args)
                result = self.tool_runner.run(request_id, tool_name, normalized_args)

                observation = self._format_observation(result, tool_name)
                yield observation_event(observation, result.ok, step)

                # Добавляем tool result в messages
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": observation,
                })

                # Специальная обработка compose_context → citations event
                if tool_name == "compose_context" and result.ok:
                    yield citations_event(...)

                    # Системная проверка coverage + refinement (как раньше)
                    coverage = result.data.get("citation_coverage", 0)
                    if self._should_attempt_refinement(coverage, agent_state):
                        # ... refinement logic ...
                        pass

                # Специальная обработка final_answer
                if tool_name == "final_answer" and result.ok:
                    yield final_event(result.data, step)
                    return

    # Fallback при исчерпании шагов
    yield fallback_event(request.query)
```

### 4.3 Что удаляется из agent_service.py

- `_generate_step()` — заменяется на `llm.chat_completion()`
- `_parse_llm_response()` — больше не нужен (structured tool_calls)
- Regex strip `<think>` preamble — не нужен (сервер сам фильтрует)
- `system_prompt` в `__init__` (строки 48-121) — заменяется на SYSTEM_PROMPT
- Forced FinalAnswer блок (строки 695-750) — не нужен
- CJK logit_bias hack — не нужен (chat template обрабатывает language)

### 4.4 Что остаётся

- `AgentState` — coverage tracking, refinement_count, low_coverage_disclaimer
- `_normalize_tool_params()` — нормализация параметров инструментов
- `_format_observation()` — форматирование результатов
- `_should_attempt_refinement()` — проверка coverage
- `_perform_refinement()` — системный refinement
- `_verify_answer()` — системная верификация (вызывается после final_answer)
- SSE event yield logic — те же events

---

## 5. Settings & Deploy

### 5.1 `src/core/settings.py`

```python
# LLM model
self.llm_model_name: str = os.getenv("LLM_MODEL_NAME", "qwen3-30b-a3b")

# Sampling для non-thinking mode (Qwen3 рекомендации)
self.agent_tool_temp: float = float(os.getenv("AGENT_TOOL_TEMP", "0.7"))
self.agent_tool_top_p: float = float(os.getenv("AGENT_TOOL_TOP_P", "0.8"))
self.agent_tool_top_k: int = int(os.getenv("AGENT_TOOL_TOP_K", "20"))
self.agent_tool_presence_penalty: float = float(os.getenv("AGENT_TOOL_PRESENCE_PENALTY", "1.5"))
```

### 5.2 `deploy/compose/compose.dev.yml`

```yaml
environment:
  - LLM_MODEL_NAME=qwen3-30b-a3b
```

### 5.3 llama-server launch command

```bash
set GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1

llama-server.exe ^
  -hf unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M ^
  -c 16384 --parallel 2 ^
  --flash-attn on ^
  --cache-type-k q8_0 --cache-type-v q8_0 ^
  -ngl 99 --main-gpu 0 ^
  --jinja ^
  --reasoning-budget 0
```

### 5.4 Скачать модель

```bash
PYTHONIOENCODING=utf-8 huggingface-cli download unsloth/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf --local-dir models/
```

---

## 6. Документация

Обновить в соответствии с изменениями:

### 6.1 `docs/ai/agent_technical_spec.md`
- Архитектура: ReAct regex → native function calling
- Tools: 5 tools (query_plan, search, rerank, compose_context, final_answer)
- router_select убран (один search backend)
- verify — системный, не tool для LLM
- Модель: Qwen3-30B-A3B MoE (было: Qwen3-8B)
- Sampling: temperature 0.7, top_p 0.8 (было: 0.3, 0.9)

### 6.2 `docs/ai/project_brief.md`
- Обновить модель LLM
- Обновить архитектуру агента

### 6.3 `agent_context/core/always_on.md`
- Обновить модель и команду запуска
- Обновить описание агента

### 6.4 `AGENTS.md`
- Обновить модель

### 6.5 `docs/ai/modules/src/services/agent_service.py.md`
- Полное обновление: function calling, tools schema, main loop

---

## 7. Тесты

Файл: `src/tests/test_agent_function_calling.py`

Тесты с мокированным LLM:
1. `test_tool_call_parsing` — LLM возвращает tool_call → агент вызывает правильный tool
2. `test_final_answer_stops_loop` — final_answer tool_call → агент завершается
3. `test_text_response_as_thought` — content без tool_calls → emit thought event
4. `test_multi_step_flow` — search → rerank → compose_context → final_answer
5. `test_refinement_triggers` — compose_context с low coverage → системный refinement

Не запускать pytest.

---

## 8. Чеклист реализации

### LLM Client
- [ ] `LlamaServerClient.chat_completion()` — новый метод для `/v1/chat/completions`
- [ ] Поддержка `tools`, `messages`, sampling params

### Agent Service
- [ ] `AGENT_TOOLS` — JSON schema для 5 tools
- [ ] `SYSTEM_PROMPT` — новый системный промпт с grounding instructions
- [ ] Main loop переписан: messages-based, tool_calls dispatch
- [ ] `_generate_step()` удалён
- [ ] `_parse_llm_response()` удалён
- [ ] Regex strip thinking удалён
- [ ] Forced FinalAnswer удалён
- [ ] CJK logit_bias hack удалён
- [ ] SSE events сохранены: thought, tool_invoked, observation, citations, final
- [ ] AgentState, coverage, refinement — сохранены
- [ ] _normalize_tool_params — обновлён под новые имена параметров

### Settings & Deploy
- [ ] `settings.py` — sampling params обновлены (temp 0.7, top_p 0.8)
- [ ] `compose.dev.yml` — LLM_MODEL_NAME обновлён
- [ ] llama-server команда с `--jinja --reasoning-budget 0 --cache-type-k q8_0 --cache-type-v q8_0`

### Documentation
- [ ] `agent_technical_spec.md` — обновлён
- [ ] `project_brief.md` — обновлён
- [ ] `always_on.md` — обновлён
- [ ] `AGENTS.md` — обновлён
- [ ] `agent_service.py.md` — обновлён

### Tests
- [ ] `test_agent_function_calling.py` — создан, не запускался

### Что НЕ менять
- [ ] Проверить: SSE event contract не сломан
- [ ] Проверить: ToolRunner API не изменён
- [ ] Проверить: tool функции (search, rerank и т.д.) не изменены
- [ ] Проверить: schemas/agent.py не изменён
