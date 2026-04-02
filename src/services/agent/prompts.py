"""
SPEC-RAG-20c Step 1: System prompt и tool schemas для ReAct агента.

Вынесено из agent_service.py — чистые константы, нет логики.
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are a RAG agent for searching and analyzing AI/ML news from 36 Russian-language Telegram channels.
Database contains posts from July 2025 to March 2026.

IDENTIFY THE QUERY TYPE and follow the corresponding path:

=== PATH A: DOCUMENT RETRIEVAL ===
For questions about facts, events, opinions.
  1. query_plan (optional — only for complex multi-aspect questions)
  2. Pick ONE search tool:
     • search — general search (default)
     • temporal_search — if query mentions dates/periods
     • channel_search — if query mentions a specific channel/author
     • cross_channel_compare — if query asks to compare opinions across channels
     • summarize_channel — if query asks for a channel digest
  3. rerank → compose_context → final_answer
  RULE: after any search tool you MUST call rerank → compose_context → final_answer.

=== PATH B: ANALYTICS ===
For questions about popularity, trends, statistics. NO search pipeline needed.
  1. Pick ONE analytics tool:
     • entity_tracker — popularity, timeline, comparison of AI/ML entities
     • arxiv_tracker — arxiv papers discussed in channels
     • hot_topics — trending topics for a week/month
     • channel_expertise — channel expertise profiles
  2. final_answer — IMMEDIATELY after the analytics tool.
  RULE: NEVER call search/rerank/compose_context after an analytics tool. The analytics tool output IS your answer data.

=== PATH C: NAVIGATION ===
For questions about available channels, post counts.
  1. list_channels → final_answer
  RULE: do NOT call query_plan or search for navigation questions.

=== PATH D: REFUSAL ===
  • Dates OUTSIDE July 2025 — March 2026 → immediately refuse: "данные за этот период отсутствуют в базе"
  • Non-existent entity (GPT-7, Bard 3) → immediately refuse: "не найдено в базе"

=== HARD RULES ===
  • NEVER call the same tool twice. One call — one result.
  • NEVER call query_plan for simple, navigation, or analytics questions.
  • NEVER mix paths: if you started analytics — do NOT add search.
  • NEVER call related_posts more than once.
  • After compose_context → IMMEDIATELY call final_answer, do not search again.
  • NEVER respond with plain text. Always use final_answer tool.

=== RESPONSE FORMAT ===
  • Answer ONLY in Russian (Отвечай ТОЛЬКО на русском языке).
  • Back every claim with citations [1], [2] (except analytics).
  • Analytics answers do NOT need citations — data comes from aggregation.
  • For entity_tracker use canonical names: OpenAI, DeepSeek, NVIDIA.
  • In final_answer you MUST fill the sources field.
"""

AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_plan",
            "description": (
                "Декомпозирует сложный запрос на 3-5 подзапросов с фильтрами. "
                "Используй ТОЛЬКО для сложных вопросов с несколькими аспектами. "
                "НЕ используй для навигации, аналитики или простых вопросов."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Исходный запрос пользователя",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Широкий поиск по всей базе AI/ML новостей из Telegram-каналов. "
                "Используй когда НЕ нужен фильтр по дате или каналу, "
                "или для общих/сравнительных запросов. Это fallback если другие инструменты не подходят."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список поисковых запросов (2-5 штук)",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Количество результатов",
                        "default": 10,
                    },
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "temporal_search",
            "description": (
                "Поиск новостей за конкретный период времени. "
                "Используй когда в запросе есть даты, месяцы, периоды. "
                "Примеры: 'Что произошло в январе 2026?', 'Новинки на CES 2026', "
                "'Что нового в марте?'. "
                "НЕ используй для вопросов без привязки ко времени."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список поисковых запросов (2-5 штук)",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Начало периода ISO YYYY-MM-DD",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Конец периода ISO YYYY-MM-DD",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Количество результатов",
                        "default": 15,
                    },
                },
                "required": ["queries", "date_from", "date_to"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "channel_search",
            "description": (
                "Поиск в конкретном Telegram-канале. "
                "Используй когда упоминается канал или автор. "
                "Примеры: 'Что писал gonzo_ml про трансформеры?', "
                "'О чём рассказывал Себрант?', 'Новости от llm_under_hood'. "
                "НЕ используй для общих вопросов без указания канала."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список поисковых запросов (2-5 штук)",
                    },
                    "channel": {
                        "type": "string",
                        "description": "Имя канала: gonzo_ml, llm_under_hood, ai_newz, techsparks, boris_again, seeallochnaya и др.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Количество результатов",
                        "default": 10,
                    },
                },
                "required": ["queries", "channel"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rerank",
            "description": (
                "Переранжирует найденные документы по семантической близости к запросу. "
                "Вызывай после search. Документы подставляются автоматически."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Исходный пользовательский запрос",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Количество лучших документов",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compose_context",
            "description": (
                "Собирает контекст из ВСЕХ найденных документов с цитатами [1], [2] и считает coverage. "
                "Вызывай после rerank. Документы подставляются автоматически. Не передавай параметры."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": (
                "Формирует финальный ответ пользователю на русском языке, "
                "опираясь только на собранный контекст."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Текст ответа с цитатами [1], [2]",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Номера использованных источников",
                    },
                },
                "required": ["answer", "sources"],
            },
        },
    },
    # ─── SPEC-RAG-13: новые tools ────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "list_channels",
            "description": (
                "Показывает доступные Telegram-каналы и количество постов. "
                "Используй когда спрашивают какие каналы есть, сколько постов в канале, "
                "или нужно уточнить название. НЕ используй для поиска по содержимому."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Имя конкретного канала (если нужен count одного канала)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "related_posts",
            "description": (
                "Находит посты похожие на указанный. "
                "Используй ТОЛЬКО когда пользователь просит 'ещё такое же', 'похожие посты'. "
                "Вызывай МАКСИМУМ 1 раз за запрос. НЕ используй для первичного поиска."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "ID исходного поста из результатов search",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Количество похожих постов",
                        "default": 5,
                    },
                },
                "required": ["post_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cross_channel_compare",
            "description": (
                "Сравнивает как разные каналы обсуждают одну тему. "
                "Используй когда спрашивают 'сравни', 'как разные каналы', "
                "'мнения экспертов о X', 'X vs Y'. "
                "Даты не обязательны — без них ищет по всему корпусу. "
                "НЕ используй для поиска в одном канале."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Тема для сравнения",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Начало периода ISO YYYY-MM-DD",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Конец периода ISO YYYY-MM-DD",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_channel",
            "description": (
                "Получает последние посты канала за период для составления сводки. "
                "Используй когда спрашивают 'что нового в канале X', 'дайджест канала'. "
                "НЕ используй для поиска конкретной темы в канале — для этого channel_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Имя канала из list_channels",
                    },
                    "time_range": {
                        "type": "string",
                        "enum": ["day", "week", "month"],
                        "default": "week",
                    },
                },
                "required": ["channel"],
            },
        },
    },
    # SPEC-RAG-15: entity analytics tools
    {
        "type": "function",
        "function": {
            "name": "entity_tracker",
            "description": (
                "Аналитика AI/ML сущностей: популярность, динамика, сравнение, связи. "
                "Используй когда спрашивают 'какие компании/модели популярны', "
                "'как менялось обсуждение X', 'что связано с X', 'сравни популярность X и Y'. "
                "Возвращает агрегации (counts), не документы."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["top", "timeline", "compare", "co_occurrence"],
                        "description": (
                            "top — топ сущностей по упоминаниям; "
                            "timeline — динамика одной сущности по неделям; "
                            "compare — сравнение 2+ сущностей; "
                            "co_occurrence — что упоминается вместе с сущностью"
                        ),
                    },
                    "entity": {
                        "type": "string",
                        "description": "Имя сущности (для timeline, compare, co_occurrence). Примеры: OpenAI, DeepSeek, GPT-5, NVIDIA",
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список сущностей для compare mode (≥2)",
                    },
                    "period_from": {
                        "type": "string",
                        "description": "Начало периода ISO date: 2025-11-01",
                    },
                    "period_to": {
                        "type": "string",
                        "description": "Конец периода ISO date: 2026-03-25",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["org", "model"],
                        "description": "Фильтр по категории (только для mode=top)",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Количество результатов",
                    },
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arxiv_tracker",
            "description": (
                "Аналитика arxiv-статей: популярные papers, поиск обсуждений конкретной статьи. "
                "Используй когда спрашивают 'какие статьи обсуждались', "
                "'кто обсуждал paper X', 'самые популярные arxiv papers'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["top", "lookup"],
                        "description": (
                            "top — самые обсуждаемые papers; "
                            "lookup — посты обсуждающие конкретную paper"
                        ),
                    },
                    "arxiv_id": {
                        "type": "string",
                        "description": "ID arxiv статьи для lookup (например: 2502.13266, 1706.03762)",
                    },
                    "period_from": {
                        "type": "string",
                        "description": "Начало периода ISO date (только для mode=top): 2025-11-01",
                    },
                    "period_to": {
                        "type": "string",
                        "description": "Конец периода ISO date (только для mode=top): 2026-03-25",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                    },
                },
                "required": ["mode"],
            },
        },
    },
    # --- SPEC-RAG-16: pre-computed analytics ---
    {
        "type": "function",
        "function": {
            "name": "hot_topics",
            "description": (
                "Возвращает горячие темы и тренды за период (неделю/месяц). "
                "Pre-computed дайджест: trending topics, top entities, burst events. "
                "Используй для: 'что обсуждали на этой неделе?', 'тренды', 'горячие темы', "
                "'дайджест', 'какие темы были в марте?', 'что было популярно?'. "
                "ПРЕДПОЧТИ этот tool над temporal_search когда вопрос про тренды/темы/дайджест."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "Период: 'this_week', 'last_week', 'YYYY-WNN' (неделя), 'this_month', или 'YYYY-MM' (месяц, напр. '2026-03')",
                        "default": "this_week",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Количество топ-тем",
                        "default": 5,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "channel_expertise",
            "description": (
                "Возвращает профили каналов: экспертиза, авторитетность, скорость покрытия тем. "
                "Используй для: 'кто лучше пишет про NLP?', 'профиль канала gonzo_ml', "
                "'какие каналы самые авторитетные?', 'эксперты по теме'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Конкретный канал (без @) или null для ranking",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Тема для поиска каналов-экспертов",
                    },
                    "metric": {
                        "type": "string",
                        "description": "Метрика для ranking: 'authority', 'speed', 'volume', 'breadth'",
                        "default": "authority",
                    },
                },
            },
        },
    },
]
