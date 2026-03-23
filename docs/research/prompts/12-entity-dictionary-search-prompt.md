# Поиск готовых AI/ML entity dictionaries

## Задача

Найди готовые открытые словари/списки/датасеты AI/ML сущностей которые можно скачать и использовать для NER в русскоязычном корпусе Telegram-каналов про AI/ML.

## Что нужно

Словарь entity → aliases, желательно с:
- Каноническое имя (OpenAI, GPT-4o, PyTorch, ...)
- Aliases и вариации написания (включая русские: "дипсик", "нвидиа", ...)
- Категория (company, model, framework, technique, conference)
- Минимум 200-500 entities

## Типы сущностей

1. **Компании**: OpenAI, Google, Anthropic, NVIDIA, Meta, Microsoft, DeepSeek, Яндекс, Сбер, ...
2. **Модели**: GPT-5, Claude, Gemini, Llama, Qwen, Mistral, FLUX, Sora, SAM, ...
3. **Фреймворки**: PyTorch, TensorFlow, LangChain, vLLM, Ollama, ...
4. **Методы/техники**: RAG, LoRA, RLHF, MoE, Transformer, CoT, ...
5. **Конференции**: NeurIPS, ICML, ICLR, GTC, ...
6. **Датасеты/бенчмарки**: MMLU, HellaSwag, HumanEval, ...

## Где искать

- GitHub (awesome-lists, NER datasets, AI taxonomies)
- HuggingFace Datasets
- Papers with Code API / taxonomy
- Wikidata SPARQL queries (AI entities)
- Kaggle datasets
- Русскоязычные NLP ресурсы (DeepPavlov, Natasha entity lists, ru-NER datasets)
- Готовые gazetteers для NER систем

## Критерии

- **Открытая лицензия** (MIT, Apache, CC)
- **Актуальность** — содержит модели 2024-2026 (GPT-5, Claude 3.5/4, Gemini 2, Qwen3, ...)
- **Русские aliases** — огромный плюс если есть
- **Формат** — JSON, CSV, TSV — легко парсить

## Что НЕ нужно

- Обученные NER модели (GLiNER, spaCy) — это tier-2, отдельная задача
- Общие NER датасеты (CoNLL, OntoNotes) — слишком generic
- Платные API (Google Knowledge Graph, etc.)

## Формат ответа

Для каждого найденного ресурса:
1. Название и URL
2. Что содержит (количество entities, категории)
3. Формат данных
4. Есть ли русские aliases
5. Насколько актуален (последнее обновление)
6. Как использовать для нашей задачи

Если ничего идеального нет — предложи лучший способ автоматически сгенерировать словарь из открытых источников (Papers with Code API, HuggingFace API, Wikidata SPARQL).
