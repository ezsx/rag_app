# compose_context - Инструмент сборки контекста с composite coverage metric

## Обзор

`compose_context` собирает prompt-контекст из документов, формирует цитаты и считает `citation_coverage` не по наивному ratio, а по composite metric из пяти сигналов.

## Параметры

- `docs` (`List[Dict]`): документы формата `[{id, text, metadata, dense_score?}]`
- `query` (`str`): исходный запрос пользователя для term coverage
- `max_tokens_ctx` (`int`): лимит контекста
- `citation_format` (`str`): формат цитирования
- `enable_lost_in_middle_mitigation` (`bool`): включение перестановки документов

## Внутренние helper-функции

- `_query_term_coverage(query, docs)`:
  - берёт значимые термины запроса
  - отбрасывает стоп-слова и токены короче 3 символов
  - считает долю терминов, встречающихся в текстах документов
- `_compute_coverage(query, docs)`:
  - использует `dense_score` как основной сигнал cosine similarity
  - fallback: `score`, затем `0.0`
  - считает 5 сигналов:
    - `max_sim`
    - `mean_top_k`
    - `term_coverage`
    - `doc_count_adequacy`
    - `score_gap`
    - `above_threshold_ratio`

## Формула coverage

```text
coverage = min(1.0,
    0.25 * max_sim
  + 0.20 * mean_top_k
  + 0.20 * term_coverage
  + 0.15 * doc_count_adequacy
  + 0.15 * score_gap
  + 0.05 * above_threshold_ratio
)
```

Параметры по умолчанию:
- `relevance_threshold = 0.55`
- `target_k = 5`

## Алгоритм работы

1. Переводит `max_tokens_ctx` в лимит символов
2. Усекает тексты только для prompt
3. При необходимости применяет lost-in-the-middle mitigation
4. Формирует `prompt`, `citations`, `contexts`
5. Считает `citation_coverage` по исходному списку `docs`, а не по усечённому prompt

## Возвращаемые данные

```python
{
    "prompt": str,
    "citations": [{"id": "string", "index": int, "metadata": {...}}],
    "contexts": List[str],
    "citation_coverage": float
}
```


