# SPEC-RAG-10: Grounding & Citation Quality

> **Версия:** 1.0
> **Дата:** 2026-03-17
> **Статус:** Draft
> **Цель:** Устранить галлюцинации LLM при генерации ответа. Заставить модель использовать
> ТОЛЬКО факты из retrieved контекста и обязательно цитировать источники.
> **Источники:** R-07 Block 4 (citation-forced generation, -60% hallucinations),
> SPEC-RAG-08 (function calling, SYSTEM_PROMPT, final_answer tool schema)
> **Зависит от:** SPEC-RAG-08 (function calling), SPEC-RAG-09 (reranker в pipeline)

---

## 0. Мотивация

Текущая проблема: LLM получает контекст из Telegram-каналов, но в ответе **придумывает факты**
из training data (GPT-4, медицина, и т.д.) вместо использования реальных данных из контекста.

Per R-07: citation-forced generation **снижает галлюцинации на 60%**. Подход:
- Каждый документ в контексте пронумерован `[1]`, `[2]`, ...
- LLM обязан цитировать источники inline: "DeepSeek выпустит V4 [2]"
- `final_answer` schema требует поле `sources: [1, 2, 3]`
- System prompt жёстко запрещает генерацию фактов без цитат

---

## 1. Изменяемые файлы

| Файл | Характер |
|------|----------|
| `src/services/agent_service.py` | Обновить SYSTEM_PROMPT, observation format для compose_context |
| `src/services/tools/compose_context.py` | Улучшить формат prompt с numbered citations |
| `src/services/tools/final_answer.py` | Валидация sources, постобработка ответа |
| `docs/ai/agent_technical_spec.md` | Grounding rules |
| `docs/ai/modules/src/services/tools/compose_context.py.md` | Обновить |

### Что НЕ менять

- AGENT_TOOLS schema — `final_answer` уже имеет `answer` + `sources` (SPEC-RAG-08)
- SSE events — не менять
- Coverage metric — не менять
- Tool runner, search, rerank — не менять

---

## 2. SYSTEM_PROMPT — усиление grounding

Текущий SYSTEM_PROMPT (SPEC-RAG-08) уже содержит базовые grounding instructions.
Усилить секцию ПРАВИЛА:

```python
SYSTEM_PROMPT = """Ты — RAG-агент для поиска информации в базе новостей из Telegram-каналов.

ПОРЯДОК РАБОТЫ:
1. query_plan — декомпозируй запрос на подзапросы
2. search — найди документы по подзапросам
3. rerank — переранжируй документы по исходному запросу
4. compose_context — собери контекст из лучших документов
5. final_answer — дай итоговый ответ строго на основе контекста

ПРАВИЛА ОТВЕТА (КРИТИЧЕСКИ ВАЖНО):
- Отвечай ТОЛЬКО на русском языке
- Используй ИСКЛЮЧИТЕЛЬНО факты из предоставленного контекста
- КАЖДОЕ фактическое утверждение ОБЯЗАТЕЛЬНО подкрепляй ссылкой [1], [2] и т.д.
- НЕ ДОБАВЛЯЙ информацию из своих знаний — только из контекста
- Если в контексте нет ответа на вопрос, прямо скажи: "В доступных источниках не найдено информации по этому вопросу"
- Не упоминай модели, продукты или события, которых нет в контексте
- В final_answer ОБЯЗАТЕЛЬНО заполни поле sources номерами использованных источников
- После compose_context переходи к final_answer, не ищи повторно
"""
```

---

## 3. compose_context — improved citation format

### 3.1 Текущий формат prompt

```
[1] Текст документа 1...

[2] Текст документа 2...
```

### 3.2 Улучшенный формат

Добавить метаданные к каждому документу (channel, date) чтобы LLM мог упоминать источник:

```
[1] (data_secrets, 2026-01-27): "В декабре 2025 возможности агентных LLM пересекли некий порог..."

[2] (ai_machinelearning_big_data, 2026-01-10): "DeepSeek выпустит V4 в феврале..."
```

Изменить в `compose_context.py` формирование chunks:

```python
# Было:
chunks.append(f"[{idx}] {cut}")

# Стало:
meta = d.get("metadata", {})
channel = meta.get("channel", "")
date = meta.get("date", "")[:10]  # YYYY-MM-DD
if channel and date:
    chunks.append(f"[{idx}] ({channel}, {date}): {cut}")
elif channel:
    chunks.append(f"[{idx}] ({channel}): {cut}")
else:
    chunks.append(f"[{idx}] {cut}")
```

---

## 4. final_answer tool — валидация и постобработка

### 4.1 Валидация sources

В `src/services/tools/final_answer.py` добавить проверку:
- Если `sources` пустой или отсутствует — добавить warning в ответ
- Если в `answer` есть `[N]` ссылки, но `sources` не содержит N — добавить N в sources

```python
def final_answer(
    answer: str,
    sources: Optional[List[int]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Формирует финальный ответ с валидацией citations."""
    # Извлекаем цитаты из текста ответа
    cited_in_text = set(int(m) for m in re.findall(r'\[(\d+)\]', answer))

    # Merge: sources из LLM + найденные в тексте
    all_sources = sorted(set(sources or []) | cited_in_text)

    if not all_sources and answer:
        # LLM не указал ни одного источника — добавляем warning
        answer = (
            answer + "\n\n⚠️ Источники не указаны. "
            "Информация может быть неточной."
        )

    return {
        "answer": answer,
        "sources": all_sources,
    }
```

### 4.2 Постобработка: strip hallucination markers

Иногда LLM генерирует "по данным GPT-4" или "согласно последним исследованиям" без цитат.
Добавить лёгкую постобработку — если предложение содержит фактическое утверждение
без `[N]` ссылки, **не удалять** (это слишком агрессивно), но логировать для мониторинга:

```python
# Подсчёт uncited assertions для observability
sentences = re.split(r'[.!?]\s+', answer)
uncited = [s for s in sentences if s.strip() and not re.search(r'\[\d+\]', s)]
if uncited:
    logger.info(
        "final_answer: %d/%d предложений без цитат",
        len(uncited), len(sentences)
    )
```

---

## 5. Coverage threshold recalibration

После SPEC-RAG-09 (новый embedding + reranker + chunking) dense_score distribution изменится.
Вернуть `coverage_threshold` к **0.65** (DEC-0019 target):

```python
# settings.py
self.coverage_threshold: float = float(os.getenv("COVERAGE_THRESHOLD", "0.65"))
```

Комментарий "калибровано под ~1K points" убрать — теперь с chunking и лучшим embedding
distribution будет ближе к проектной.

---

## 6. Документация

- `docs/ai/agent_technical_spec.md` — добавить секцию "Grounding & Citation Rules"
- `docs/ai/modules/src/services/tools/compose_context.py.md` — обновить формат prompt
- `docs/ai/modules/src/services/tools/final_answer.py.md` — обновить (если существует, или создать)

---

## 7. Тесты

### `src/tests/test_final_answer.py` (новый или обновить)

- `test_sources_extracted_from_text` — `[1]` в ответе → sources содержит 1
- `test_empty_sources_adds_warning` — нет sources → warning в тексте
- `test_sources_merged` — sources=[1] + текст с [2] → all_sources=[1,2]

### `src/tests/test_compose_context.py` (обновить)

- `test_citation_format_includes_metadata` — проверить что prompt содержит "(channel, date): text"

Не запускать pytest.

---

## 8. Чеклист

- [ ] SYSTEM_PROMPT обновлён с жёсткими grounding rules
- [ ] compose_context.py — citation format с metadata (channel, date)
- [ ] final_answer.py — валидация sources, merge с текстовыми цитатами
- [ ] final_answer.py — observability: логирование uncited assertions
- [ ] coverage_threshold вернуть на 0.65
- [ ] Документация обновлена
- [ ] Тесты созданы, не запускались
