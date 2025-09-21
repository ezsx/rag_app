### Модуль `src/services/tools/content_filter.py`

Назначение: базовая модерация контента (PII, URL, токсичность/брань, хейт, сексуальный контент).

API:
- `content_filter(text: str, categories?: List[str]) -> Dict`
  - Выход: `{ allowed: bool, violations: {..}, suggested_action: allow|sanitize|reject, sanitized_preview }`

Правила (MVP):
- Хейт или много сексуальных маркеров → `reject`
- Любые флаги PII/URL/брань → `sanitize`
- Иначе → `allow`
