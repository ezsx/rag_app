## src/cli/download_models.py — CLI скачивания моделей

### Назначение
- Утилита для скачивания рекомендованных LLM и embedding‑моделей в локальные директории (`MODELS_DIR`, `TRANSFORMERS_CACHE`).

### Основные опции
- `--llm <key>` — ключ из `RECOMMENDED_MODELS["llm"]` (по умолчанию `gpt-oss-20b`).
- `--embedding <key>` — ключ из `RECOMMENDED_MODELS["embedding"]` (по умолчанию `multilingual-e5-large`).
- `--models-dir`, `--cache-dir` — пути хранения моделей и кэша.
- `--list` — показать доступные модели и файлы.
- `--llm-only` / `--embedding-only` — выборочно скачивать.

### Поведение
- Добавляет `src/` в `sys.path`, использует `utils.model_downloader.auto_download_models`.
- Печатает подробный прогресс и итоговые пути/успехи.


