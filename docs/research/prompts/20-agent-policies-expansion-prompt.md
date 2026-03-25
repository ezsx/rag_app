# Задача: расширить agent_policies в datasets/tool_keywords.json

## Контекст

`datasets/tool_keywords.json` теперь содержит две секции:
- `tool_keywords` — keyword routing для dynamic visibility (уже заполнен, не трогать)
- `agent_policies` — refusal markers, negative intent markers, refusal trim patterns, eviction order

Policies загружаются в `agent_service.py` через `_load_policy(name)` и используются в:
1. **Forced search bypass** — решение форсить ли search если LLM отказал без поиска
2. **Refusal trim** — обрезание альтернатив после refusal в final answer
3. **Hard cap eviction** — порядок удаления tools при >5 visible

## Что нужно сделать

Расширить `values` в каждой policy-секции. Текущие значения — минимальные, найденные при анализе 30 eval вопросов. Нужно покрыть больше edge cases.

### 1. `refusal_markers`

**Как используется**: substring match по `content.lower()` (ответ LLM). Если найден маркер — LLM пытается отказать.

**Текущие**:
```json
["нет в базе", "отсутству", "нет данных", "вне периода", "не содержит информац", "не найден"]
```

**Что добавить**: подумать о других способах как Qwen3 может сформулировать отказ:
- "не удалось найти"
- "информация отсутствует"
- "не располагаю данными"
- "в базе данных нет"
- "к сожалению"
- EN варианты?
- Другие формулировки из русскоязычных LLM

**Правила**:
- Stems/substrings (как в tool_keywords)
- НЕ слишком короткие — "нет" ловит всё подряд
- Проверять на false positives: "В базе нет данных" — refusal, "В базе нет одной модели, но есть другие" — не refusal (но это edge case)

### 2. `negative_intent_markers`

**Как используется**: substring match по `query.lower()` (вопрос пользователя). Если вопрос содержит negative intent И LLM дал refusal → НЕ форсим search. Для обычных factual вопросов — форсим search даже при refusal.

**Текущие**:
```json
["существует ли", "выходила ли", "есть ли модель", "бывает ли", "была ли", "выпускал ли", "создавал ли"]
```

**Что добавить**: другие формулировки "проверки существования":
- "правда ли что"
- "действительно ли"
- "есть ли информация"
- "знаешь ли"
- Формы на "ли" с другими глаголами
- Out-of-range temporal patterns ("в 2024", "в 2023", "до 2025")

**Правила**:
- Должны ловить ТОЛЬКО запросы где пользователь спрашивает "существует ли X" / "было ли Y"
- НЕ ловить обычные factual: "Какие модели выпускали?" (это не negative intent)
- "ли" в конце — хороший маркер вопросительности

### 3. `refusal_alt_patterns`

**Как используется**: substring match по `answer.lower()`. Если ответ содержит refusal marker + alt_pattern → обрезаем ответ до refusal. Не даём LLM предлагать альтернативы после отказа.

**Текущие**:
```json
["однако", "но в базе", "при этом", "вместе с тем", "тем не менее", "если вас интересует", "зато", "впрочем"]
```

**Что добавить**: другие способы LLM перехода к альтернативе:
- "вместо этого"
- "могу предложить"
- "возможно вам будет интересно"
- "стоит отметить"
- "хотя"
- "но я нашёл"
- "в качестве альтернативы"

**Правила**:
- Маркеры должны быть достаточно специфичны
- "но" без продолжения — слишком короткий
- "хотя" может быть ОК — обычно вводит альтернативу

### 4. `eviction_order`

**Как используется**: при >5 visible tools удаляем tools по порядку пока не останется <=5.

**Текущие**:
```json
["arxiv_tracker", "entity_tracker", "list_channels", "summarize_channel", "search"]
```

**Что менять**: порядок уже хороший (analytics evicted first, search last). Но при добавлении новых tools нужно будет обновлять. Пока можно оставить как есть или добавить `temporal_search` и `channel_search` на случай если signal-based routing добавит их вместе с keyword-based.

## Файлы для чтения

1. `datasets/tool_keywords.json` — текущий файл (расширить agent_policies секцию)
2. `src/services/agent_service.py`:
   - `_load_policy()` (~строка 60) — как загружаются policies
   - Forced search bypass (~строка 700) — как используются refusal_markers + negative_intent_markers
   - `_trim_refusal_alternatives()` (~строка 1553) — как используются refusal_alt_patterns
   - `_get_step_tools()` eviction (~строка 1942) — как используется eviction_order
3. `results/raw/eval_results_20260325-192924.json` — raw eval results (посмотреть реальные refusals и alt-responses от Qwen3)

## Валидация

После расширения проверить:

```python
import json
d = json.load(open("datasets/tool_keywords.json"))
policies = d["agent_policies"]

# Все policies имеют values
for name, policy in policies.items():
    assert "values" in policy, f"{name} missing values"
    assert len(policy["values"]) > 0, f"{name} empty"
    print(f"{name}: {len(policy['values'])} values")

# Проверить что refusal_markers не ловят нормальные ответы
normal_answers = [
    "OpenAI выпустила GPT-5 с новыми возможностями",
    "В январе 2026 обсуждались агенты и трансформеры",
    "Канал gonzo_ml пишет про ML/AI технологии",
]
for ans in normal_answers:
    ans_lower = ans.lower()
    markers = policies["refusal_markers"]["values"]
    matched = [m for m in markers if m in ans_lower]
    assert not matched, f"False positive refusal: '{ans}' matched {matched}"

print("All checks passed")
```

## Формат

Сохранить в `datasets/tool_keywords.json` → секция `agent_policies`. Тот же формат: `values` + `comments`.
НЕ трогать секцию `tool_keywords` — она уже проверена и работает.
