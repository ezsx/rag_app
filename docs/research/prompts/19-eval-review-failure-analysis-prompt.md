# Задача: независимая оценка eval результатов + анализ failure cases

## Контекст

Прогнали полный eval (30 вопросов) после реализации SPEC-RAG-15 (entity_tracker + arxiv_tracker).
Claude Opus 4.6 уже сделал manual judge. Нужен независимый review + root cause analysis по failures.

## Часть 1: Независимая оценка ответов

### Файлы для чтения

1. `results/raw/eval_results_20260325-192924.json` — **raw результаты** (30 вопросов, полные ответы агента + metadata)
2. `results/reports/eval_judge_20260325_spec15.md` — **Claude Opus judge** (per-question таблица, scores)
3. `datasets/eval_golden_v1.json` — **golden dataset** (expected answers, key_tools, source_post_ids)

### Что нужно сделать

Для каждого из 30 вопросов оцени:
- **Factual** (0-2): 0=wrong/hallucinated, 1=partially correct, 2=fully correct
- **Useful** (0-2): 0=useless, 1=partially useful, 2=fully answers the question

Сверяй ответ агента с `expected_answer` из golden dataset.
Для проверки фактов можешь обращаться к Qdrant напрямую (см. раздел ниже).

В конце — сводная таблица и сравнение с оценками Claude.

## Часть 2: Анализ 5 failure cases

### Failures

| ID | Query | Failure type | Что произошло |
|----|-------|-------------|---------------|
| q01 | Кого Financial Times назвала человеком года в 2025? | tool_selected_wrong | LLM отказал без вызова search. Информация есть в базе |
| q03 | За сколько Meta купила Manus AI? | tool_selected_wrong | LLM отказал без вызова search. q13 находит ту же информацию через cross_channel_compare |
| q19 | Существует ли модель GPT-7? | refusal quality | Сказал "не найдена", но предложил альтернативы GPT-5/GPT-6. Нарушает refusal policy |
| q22 | Как менялось обсуждение OpenAI за 3 месяца? | tool_execution_failed | entity_tracker выбран правильно (KTA=1), но llama-server вернул 400 |
| q25 | Подходы к LLM в production в llm_under_hood и boris_again | tool_selected_wrong | Ожидался `search`, агент выбрал `channel_search`. Ответ содержательный, но KTA=0 из-за schema mismatch |

### Что нужно для каждого failure

1. **Root cause**: почему именно LLM принял это решение? Посмотри visible_tools, system prompt, query signals
2. **Код**: проверь логику в `src/services/agent_service.py`:
   - Forced search bypass (~строка 659): почему не сработал для q01/q03?
   - Refusal markers: какие markers триггерят bypass?
   - `_get_step_tools()` (~строка 1800): какие tools были видны?
3. **Fix proposal**: конкретные изменения в коде/промпте которые решат проблему

### Ключевые вопросы по q01/q03

Гипотеза: Qwen3 генерирует content с refusal markers ДО вызова tools → forced search видит refusal markers → не форсит search. Проверь:
- Что именно в SSE events? (`visible_tools_history`, `tools_invoked` = [])
- Forced search check: `not tool_calls and search_count == 0 and not navigation_answered and not analytics_done and not is_refusal`
- Какой content генерирует LLM? Содержит ли он "нет в базе" / "отсутству"?

## Доступ к Qdrant (для проверки фактов)

Qdrant доступен на `http://localhost:16333` (Docker Desktop, порт 16333 → контейнер 6333).
Коллекция: `news_colbert_v2`.

### Python примеры

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:16333")
COLLECTION = "news_colbert_v2"

# Проверить что информация есть в базе (для q01)
results, _ = client.scroll(
    collection_name=COLLECTION,
    scroll_filter=models.Filter(must=[
        models.FieldCondition(key="entities", match=models.MatchValue(value="NVIDIA")),
        models.FieldCondition(key="text", match=models.MatchText(text="Financial Times"))
    ]),
    limit=5,
    with_payload=True,
    with_vectors=False,
)
for p in results:
    print(f"[{p.payload['channel']}] {p.payload['date']}: {p.payload['text'][:200]}")

# Facet API (для проверки analytics данных)
result = client.facet(COLLECTION, key="entities", limit=5, exact=True)
for h in result.hits:
    print(f"{h.value}: {h.count}")

# Scroll с text match (для проверки конкретных фактов)
results, _ = client.scroll(
    collection_name=COLLECTION,
    scroll_filter=models.Filter(must=[
        models.FieldCondition(key="text", match=models.MatchText(text="Manus"))
    ]),
    limit=10,
    with_payload=True,
    with_vectors=False,
)
```

### Shell (из WSL2)

```bash
# Health check
curl -s http://localhost:16333/collections/news_colbert_v2 | python3 -m json.tool | head -20

# Scroll с filter
curl -s -X POST http://localhost:16333/collections/news_colbert_v2/points/scroll \
  -H "Content-Type: application/json" \
  -d '{
    "filter": {"must": [{"key": "text", "match": {"text": "Financial Times"}}]},
    "limit": 5,
    "with_payload": true,
    "with_vector": false
  }' | python3 -c "import json,sys; [print(f'{p[\"payload\"][\"channel\"]}: {p[\"payload\"][\"text\"][:150]}') for p in json.load(sys.stdin)['result']['points']]"
```

## Формат ответа

### Часть 1: Per-question scores table
```
| ID | Factual (Codex) | Useful (Codex) | Factual (Claude) | Useful (Claude) | Agreement |
```

### Часть 2: Per-failure analysis
```
### q01: [root cause]
**Что произошло**: ...
**Root cause**: ...
**Fix**: ...
```

### Часть 3: Overall verdict
Общая оценка качества агента, сравнение с Claude, рекомендации.
