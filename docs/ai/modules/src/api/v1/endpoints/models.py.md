### Модуль: `src/api/v1/endpoints/models.py`

Назначение: управление активными моделями (LLM, embedding).

#### Эндпоинты
- `GET /v1/models` → список доступных моделей из `RECOMMENDED_MODELS` + текущие ключи.
- `POST /v1/models/select` → горячая смена LLM/embedding (сброс кешей через методы `Settings`).
- `GET /v1/models/{model_type}/current` → информация о текущей модели.

#### Зависимости
- `get_settings`, `utils.model_downloader.RECOMMENDED_MODELS`.





