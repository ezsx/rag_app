### Модуль `src/services/tools/extract_entities.py`

Назначение: извлечение сущностей (паттерны + известные списки + capitalized‑фразы).

API:
- `extract_entities(text: str, entity_types?: List[str]) -> Dict`
  - Выход: `{ entities: {..}, total_count, entity_types: [...], summary }`

Категории:
- Паттерны: email, phone, url, ip, date, time, money, percentage, code
- Списки: tech_companies, programming_languages, databases, cloud_services, frameworks
- Эвристика: фразы с заглавных букв (имена/организации)
