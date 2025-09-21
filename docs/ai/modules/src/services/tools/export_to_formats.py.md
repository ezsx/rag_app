### Модуль `src/services/tools/export_to_formats.py`

Назначение: экспорт контента в лёгкие форматы.

API:
- `export_to_formats(content: str, fmt: str, filename_base?: str, metadata?: Dict) -> Dict`
  - Поддержка: `md`, `txt`, `json`; статусы: `not_supported` (pdf/docx), `unknown_format`.

Выход:
- Для поддерживаемых форматов — `{ filename, mime_type, data, encoding, meta }`.
