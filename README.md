# Telegram RAG Application - система вопросов и ответов

Система **Retrieval-Augmented Generation (RAG)** для ответов на вопросы на основе сообщений из каналов telegram.

## ✨ Возможности

- 🇷🇺 **Русскоязычная модель** - Vikhr-7B-instruct для качественных ответов
- 🚀 **GPU ускорение** - Быстрая обработка на NVIDIA GPU
- 📊 **Векторная база** - ChromaDB для поиска релевантных документов  
- 📱 **REST API** - Простой HTTP интерфейс
- 📨 **Telegram инgest** - Загрузка сообщений из Telegram каналов
- 🔄 **Автоскачивание моделей** - Настройка одной командой
- 🐳 **Docker готовность** - Простой запуск в контейнерах

## 🛠 Технологии

- **Backend**: FastAPI, Python 3.11+
- **Векторная БД**: ChromaDB 1.0.13
- **LLM**: Vikhr-7B-instruct (GGUF Q4_K_M)
- **Embeddings**: intfloat/multilingual-e5-large
- **ML**: PyTorch 2.2, Sentence-Transformers
- **Деплой**: Docker, NVIDIA Container Runtime

## 📋 Требования

### Минимальные
- Docker + Docker Compose
- 8 GB RAM
- 10 GB свободного места

### Рекомендуемые (для GPU)
- NVIDIA GPU с 8+ GB VRAM
- NVIDIA Container Runtime
- CUDA 12.x

## 🚀 Быстрый старт

### 1. Запуск системы
```bash
# Первый запуск (скачает и настроит модели)
docker compose --profile api up

# Ожидаем сообщение: "✅ Инициализация завершена успешно"
# API доступен по адресу: http://localhost:8000
```

### 2. Проверка работы
```bash
# Проверяем статус
curl http://localhost:8000/health

# Пробуем задать вопрос
curl -X POST "http://localhost:8000/v1/qa" \
  -H "Content-Type: application/json" \
  -d '{"query": "Расскажи о системе", "include_context": false}'
```

## 📊 Загрузка данных из Telegram

### Подготовка
1. Получите API ключи Telegram:
   - Зайдите на https://my.telegram.org
   - Создайте приложение и получите `api_id` и `api_hash`

2. Настройте `.env`:
```bash
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
```

### Загрузка сообщений
```bash

# CPU \ GPU версия (быстро, требует NVIDIA GPU)
docker compose run --rm ingest --channel durov --since 2025-07-01 \
 --until 2025-08-01 --collection news_demo4
```

**Пример производительности:**
- CPU: ~10,000 сообщений за 30 минут
- GPU: ~10,000 сообщений за 3-5 минут

## 🔧 Конфигурация (.env)

[.env.example](.env.example)

## 📡 API Endpoints

### GET /health
Проверка статуса системы
```bash
curl http://localhost:8000/health
# Ответ: {"status": "healthy", "timestamp": "..."}
```

### POST /v1/qa
Основной endpoint для вопросов и ответов

**Без контекста:**
```bash
curl -X POST "http://localhost:8000/v1/qa" \
  -H "Content-Type: application/json" \
  -d '{"query": "Что такое искусственный интеллект?"}'
```

**С контекстом:**
```bash
curl -X POST "http://localhost:8000/v1/qa" \
  -H "Content-Type: application/json" \
  -d '{"query": "О чём говорят в новостях?", "include_context": true}'
```

**Ответ с контекстом:**
```json
{
  "answer": "На основе найденных документов...",
  "query": "О чём говорят в новостях?",
  "context": [
    {
      "document": "Текст документа...",
      "metadata": {"source": "telegram", "date": "2024-01-01"},
      "distance": 0.15
    }
  ],
  "total_documents": 1
}
```


## 💾 Структура проекта

```
rag_app/
├── 📁 src/                          # Исходный код приложения
│   ├── 📁 adapters/chroma/          # Адаптер для ChromaDB
│   ├── 📁 core/                     # Основная логика (DI, конфигурация)
│   ├── 📁 schemas/                  # Pydantic схемы для API
│   ├── 📁 services/                 # Бизнес-логика (QA сервис)
│   ├── 📁 utils/                    # Утилиты (промпты, загрузка моделей)
│   ├── 📁 cli/                      # CLI команды
│   ├── 📁 tests/                    # Unit тесты
│   └── 📄 main.py                   # FastAPI приложение
├── 📁 scripts/                      # Скрипты инестинга
│   └── 📄 ingest_telegram.py        # Загрузка из Telegram
├── 📁 chroma-data/                  # Данные ChromaDB (персистентные)
├── 📁 models/                       # Кэш моделей
├── 🐳 docker-compose.yml            # Оркестрация контейнеров
├── 🐳 Dockerfile.api                # Образ для API
├── 🐳 Dockerfile.ingest             # Образ для ingest
├── 🐳 Dockerfile.chroma             # Образ для ChromaDB
├── 📄 requirements_api.txt          # Зависимости API
├── 📄 requirements_ingest.txt       # Зависимости ingest
└── 📄 .env.example                  # Пример конфигурации
```

## 📝 Лицензия

MIT License - см. файл LICENSE для деталей

---
**Версия**: MVP 1.0  
