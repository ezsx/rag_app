# Задача: расширить datasets/tool_keywords.json

## Контекст

`datasets/tool_keywords.json` — файл keyword routing для dynamic visibility RAG-агента. Когда пользователь задаёт вопрос, agent подбирает видимые tools по substring match: `keyword in query.lower()`.

Файл уже содержит 4 tools с базовыми keywords. Нужно:
1. **Расширить keywords** для существующих 4 tools
2. **Добавить keywords** для ещё 2 tools: `temporal_search` и `channel_search` (сейчас они роутятся только через signal-based logic, но keyword backup полезен)

## Как работает routing

```python
query_lower = "какие ai-компании чаще всего упоминаются в каналах?"
for tool_name, keywords in tool_keywords.items():
    if any(kw in query_lower for kw in keywords):
        visible_tools.add(tool_name)
```

**Substring match**, не exact match. `"упомина"` ловит "упоминается", "упоминался", "упоминаний".

## Правила для keywords

1. **Stems, не полные слова** — русская морфология даёт десятки форм. Используй минимальный stem покрывающий все формы:
   - "популярн" → популярный/популярных/популярности/популярностью
   - "обсужда" → обсуждают/обсуждается/обсуждаемый/обсуждали
   - НЕ "обсуждается" — это только одна форма

2. **Не слишком короткие** — "стат" ловит "статья", но и "статус", "состав". Лучше "стать" + "статей" отдельно.

3. **Двуязычные** — пользователи могут писать на русском и английском. Добавляй EN варианты.

4. **Каждый keyword — в comments** с объяснением какие формы покрывает.

5. **False positives допустимы** (лучше показать лишний tool, чем скрыть нужный) — но не слишком агрессивные. Проверяй мысленно: "сработает ли на обычных запросах типа 'что такое GPT-5?'"

## Существующие 13 tools агента

| Tool | Описание | Routing |
|------|----------|---------|
| `query_plan` | Декомпозиция запроса | Всегда видим в PRE-SEARCH |
| `search` | Общий поиск | Всегда видим в PRE-SEARCH |
| `temporal_search` | Поиск по дате/периоду | Signal-based (dates) → **добавить keywords** |
| `channel_search` | Поиск в конкретном канале | Signal-based (channel names) → **добавить keywords** |
| `cross_channel_compare` | Сравнение каналов | **Keyword-based** (уже есть) |
| `summarize_channel` | Дайджест канала | Signal-based (channel + keyword?) |
| `list_channels` | Список каналов | **Keyword-based** (уже есть) |
| `entity_tracker` | Аналитика сущностей | **Keyword-based** (уже есть) |
| `arxiv_tracker` | Аналитика arxiv papers | **Keyword-based** (уже есть) |
| `rerank` | Переранжирование | POST-SEARCH only |
| `compose_context` | Сборка контекста | POST-SEARCH only |
| `final_answer` | Финальный ответ | POST-SEARCH only |
| `related_posts` | Похожие посты | POST-SEARCH only |

**Только первые 9 нуждаются в keyword routing** (POST-SEARCH tools не роутятся по keywords).
`query_plan` и `search` всегда видны — им keywords не нужны.

## Что нужно сделать

### 1. Расширить существующие 4 tools

Подумай о запросах которые пользователь может задать и которые должны триггерить каждый tool. Примеры для вдохновения:

**cross_channel_compare:**
- "как разные каналы обсуждали X"
- "мнения экспертов о Y"
- "X vs Y в каналах"

**list_channels:**
- "покажи каналы"
- "что за каналы"
- "есть канал X?"

**entity_tracker:**
- "какие компании популярны"
- "динамика упоминаний DeepSeek"
- "что связано с NVIDIA"
- "топ моделей"
- "рейтинг AI-компаний"

**arxiv_tracker:**
- "какие papers обсуждались"
- "кто цитировал статью X"
- "научные работы по теме Y"

### 2. Добавить новые tools

**temporal_search** — запросы с датами/периодами:
- "что обсуждалось в январе 2026"
- "новости за последнюю неделю"
- "события на CES / GTC / NeurIPS"
- "что было в марте"

**channel_search** — запросы про конкретный канал:
- "что писал gonzo_ml про X"
- "в канале Y найди Z"
- "посты автора X"

**summarize_channel** — запросы на дайджест:
- "дайджест канала X"
- "последние посты X"
- "обзор канала"

### 3. Формат

Сохранить в `datasets/tool_keywords.json` тот же формат:

```json
{
  "tool_name": {
    "keywords": ["stem1", "stem2", ...],
    "comments": {
      "stem1": "объяснение какие формы покрывает",
      "stem2": "..."
    }
  }
}
```

## Файлы для чтения

1. `datasets/tool_keywords.json` — текущий файл (расширить его)
2. `src/services/agent_service.py` → `SYSTEM_PROMPT` (~строка 29) — описания tools для LLM
3. `src/services/agent_service.py` → `AGENT_TOOLS` (~строка 75) — tool schemas с descriptions
4. `datasets/eval_golden_v1.json` — 30 реальных запросов (проверить coverage)

## Валидация

После расширения проверь на всех 30 запросах из golden dataset что:
- Каждый запрос триггерит хотя бы один tool из `key_tools`
- Нет грубых false positives (обычные factual запросы не триггерят analytics)

Можно написать скрипт:
```python
import json
kw = json.load(open("datasets/tool_keywords.json"))
dataset = json.load(open("datasets/eval_golden_v1.json"))
for q in dataset:
    query = q["query"].lower()
    matched = [t for t, e in kw.items() if t != "_meta" and any(k in query for k in e["keywords"])]
    expected = q.get("key_tools", [])
    # Проверить что expected ∩ matched ≠ ∅
```
