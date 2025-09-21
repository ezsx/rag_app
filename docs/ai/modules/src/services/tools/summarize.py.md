### Модуль `src/services/tools/summarize.py`

Назначение: экстрактивное резюмирование текста с выделением ключевых моментов.

API:
- `summarize(text: str, max_sentences: int = 5, min_length: int = 100, mode: str = "extractive"|"bullets") -> Dict`
  - Выход: `{ summary, original_length, summary_length, compression_ratio, key_points, keywords, mode, sentence_count }`

Реализация:
- Токенизация предложений, TF‑нормированные веса слов (без стоп‑слов), скоринг предложений и отбор топ‑N.
