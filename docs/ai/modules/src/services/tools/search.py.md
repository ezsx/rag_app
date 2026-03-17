# search - Инструмент гибридного поиска

## Обзор

`search` выполняет поиск через `HybridRetriever` и приводит результаты к единому формату tool output.
Phase 1 больше не содержит BM25 fallback внутри этого инструмента: поиск идёт только через текущий гибридный retriever.

## Функциональность

### Основные возможности
- Нормализация и дедупликация одного или нескольких запросов
- Построение `SearchPlan` с фильтрами метаданных
- Вызов `HybridRetriever.search_with_plan(...)`
- Преобразование кандидатов в стандартизованный список `hits`

### Параметры
- `queries` (`List[str] | str`, опционально): список запросов
- `filters` (`Dict`, опционально): фильтры метаданных
- `k` (`int`): число результатов
- `route` (`str`): логический route label
- `hybrid_retriever` (`HybridRetriever`, опционально): retriever для выполнения поиска
- `query` (`str`, опционально): альтернативный способ передать один запрос
- `search_type` (`str`, опционально): дополнительная метка для трассировки

### Возвращаемые данные
```python
{
    "hits": [
        {
            "id": "string",
            "score": float,
            "text": "string",
            "snippet": "string",
            "meta": {
                "channel_id": "string",
                "channel": "string",
                "message_id": "string",
                "date": "YYYY-MM-DDTHH:MM:SSZ",
                "author": "string"
            }
        }
    ],
    "total_found": int,
    "route_used": "hybrid"
}
```

## Алгоритм работы

1. Нормализует `query` и `queries`
2. Строит `MetadataFilters`
3. Создаёт `SearchPlan`
4. Вызывает `HybridRetriever.search_with_plan(...)`
5. Форматирует кандидатов в `hits`

## Обработка ошибок

- Если `HybridRetriever` не передан, возвращает ошибку в tool result
- Если retriever вернул пусто или упал, инструмент возвращает пустой список `hits`
- Тайминг логируется только для гибридного поиска
