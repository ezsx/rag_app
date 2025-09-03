### Модуль: `src/utils/model_downloader.py`

Назначение: скачивание моделей (LLM GGUF, embedding, reranker) с прогрессом и автоподбором.

#### Функции
- `download_file_with_progress(url, local_path)` — потоковая загрузка с tqdm.
- `download_llm_model_from_hf(model_repo, filename, local_dir, cache_dir)` — скачивание конкретного файла через `hf_hub_download`, при неудаче — `snapshot_download` и выбор подходящего `.gguf`.
- `download_embedding_model(model_name, cache_dir)` — загрузка репозитория embedding‑модели в кэш.
- `auto_download_models(llm_model_key, embedding_model_key, models_dir, cache_dir)` — оркестрация загрузок по ключам из `RECOMMENDED_MODELS`.
- `download_reranker_model(model_name, cache_dir)` — прогрев кэша CrossEncoder.

#### Константы
- `RECOMMENDED_MODELS` — реестр рекомендуемых LLM/embedding/reranker моделей и их метаданных.





