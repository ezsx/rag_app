### Модуль `src/services/tools/temporal_normalize.py`

Назначение: нормализация временных выражений (даты/периоды) в ISO‑формат.

API:
- `temporal_normalize(text: str) -> Dict`
  - Выход: `{ found_dates: [{original, normalized, position}], normalized_text, dateparser_available }`

Особенности:
- Использует `dateparser` при наличии; иначе — относительные/простые паттерны.
