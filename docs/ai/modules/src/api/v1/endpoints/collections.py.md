### Модуль: `src/api/v1/endpoints/collections.py`

Назначение: управление коллекциями ChromaDB.

#### Эндпоинты
- `GET /v1/collections` → список коллекций, их счётчики и текущая активная коллекция.
- `POST /v1/collections/select` → выбор активной коллекции, обновляет `Settings.current_collection`.
- `GET /v1/collections/{collection}/info` → count и метаданные конкретной коллекции.

#### Зависимости
- `get_chroma_client`, `get_settings`.





