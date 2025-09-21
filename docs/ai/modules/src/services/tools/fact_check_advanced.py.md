### Модуль `src/services/tools/fact_check_advanced.py`

Назначение: продвинутая проверка фактов по базе знаний через `Retriever`.

API:
- `fact_check_advanced(claim: str, query?: str, k: int = 6, retriever?: Retriever) -> Dict`
  - Выход: `{ verdict: supported|refuted|insufficient, confidence: float, evidence: [..], took_ms }`

Методика:
- Поиск кандидатов `retriever.search(q, k)`
- Оценка: лексическое пересечение + (1 - distance) → агрегированный `score`
- Эвристика отрицаний: тексты с negation и хорошим overlap усиливают refute
