# ADR: RAG Agent API — Security Hardening, Research Alignment, and Tooling Expansion (2025‑09‑21)

## Контекст

Проект RAG Agent API требовал перехода к production‑ready состоянию согласно внутреннему плейбуку и исследованию. Ключевые требования:
- ReAct‑агент с инструментами и SSE
- Гибридный поиск (Chroma + BM25), MMR/RRF, опциональный CPU‑reranker
- Строгие форматы (GBNF/Schema) на критических участках
- Безопасность: аутентификация, rate limiting, санитизация, защита от prompt‑injection, безопасное логирование
- Инструменты согласно research: multi_query_rewrite, temporal normalization, web_search, summarize и др.

## Решение

1) Безопасность и платформа
- Добавлен JWT‑auth и роли (`src/core/auth.py`).
- Добавлен RateLimit middleware с экспоненциальным бэкоффом (`src/core/rate_limit.py`), заголовки экспонируются в CORS.
- Реализован `SecurityManager` для валидации/санитизации (`src/core/security.py`).
- Безопасное логирование (`sanitize_for_logging`) в `src/main.py`, `src/services/agent_service.py`.
- Обновлён порядок middleware, CORS и global exception handler.

2) Выравнивание с research
- Единые decoding‑параметры для llama.cpp: temperature 0.2–0.4, top_p 0.9, top_k 40, repeat_penalty 1.2, seed=42.
- Лимиты длины/глубины для JSON‑входов инструментов в агенте.
- Улучшена `compose_context`: Lost‑in‑the‑Middle mitigation и `citation_coverage`.

3) Инструменты (новые файлы)
- `src/services/tools/temporal_normalize.py`
- `src/services/tools/summarize.py`
- `src/services/tools/extract_entities.py`
- `src/services/tools/translate.py`
- `src/services/tools/fact_check_advanced.py`
- `src/services/tools/semantic_similarity.py`
- `src/services/tools/content_filter.py`
- `src/services/tools/export_to_formats.py`

Все зарегистрированы в `src/core/deps.py` через `ToolRunner` с таймаутами, описания добавлены в `AgentService.get_available_tools()`.

4) Агент
- Валидация входного prompt и параметров инструментов, ограничение размеров и глубины JSON.
- Обновлены decoding‑профили и stop‑последовательности.
- Исключены чувствительные данные из логов.

## Последствия

Плюсы:
- Усилена безопасность и управляемость (auth, rate‑limit, санитизация).
- Следование research/плейбуку по decoding, LITM‑mitigation, инструментарию.
- Шире покрытие сценариев (перевод, факт‑проверка, модерация, экспорт).

Минусы/риски:
- Рост сложности и необходимость расширенных тестов.
- Часть инструментов — упрощённые (web_search, naive‑translate) и требует улучшения.

## Альтернативы
- Внешние сервисы модерации/перевода (дороже, выше латентность).
- Сократить набор инструментов (хуже UX/покрытие).

## Тест‑план (высокоуровневый)
- Unit: `SecurityManager`, rate‑limit, JWT, каждый инструмент (валидный/невалидный вход, таймауты).
- Интеграция: ReAct‑петля с инструментами, проверка SSE событий.
- Нагрузка: p95 латентность, timeout‑rate, деградации.
- Наблюдаемость: метрики/логирование/трейсинг — в ближайшем инкременте.

## Ссылки
- Агент: `src/services/agent_service.py` (около 420–489)
- DI/регистрация инструментов: `src/core/deps.py` (около 469–554)
- Инструменты: `src/services/tools/*.py`
- Плейбук: `docs/ai/research/react_rag_playbook_v1.md`
