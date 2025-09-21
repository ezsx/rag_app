### Модуль `src/services/tools/translate.py`

Назначение: перевод текста между языками. При наличии фабрики LLM использует LLM для перевода; иначе — наивный RU↔EN словарный фолбэк.

Ключевые функции:
- `translate(text: str, target_lang: str, source_lang?: str, max_length: int = 1000, llm_factory?) -> Dict`
  - Вход: текст, целевой язык, опционально исходный, лимит длины, фабрика LLM
  - Выход: `{ translated_text, source_lang, target_lang, mode: (llm|naive), took_ms }`

Особенности:
- Детект языка по наличию кириллицы
- LLM‑профиль: temperature=0.2, top_p=0.9, top_k=40, repeat_penalty=1.2, seed=42
- Стоп‑последовательность: `\n\n` (минимизация «объяснений»)
