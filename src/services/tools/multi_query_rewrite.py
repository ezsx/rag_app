"""
Инструмент для генерации дополнительных перефразов запроса
"""

import logging
from typing import Dict, Any, List
from utils.gbnf import get_string_array_grammar

logger = logging.getLogger(__name__)


def multi_query_rewrite(
    query: str, existing_queries: List[str], target_count: int = 5, llm_factory=None
) -> Dict[str, Any]:
    """
    Генерирует дополнительные перефразы запроса для улучшения полноты поиска.

    Args:
        query: Исходный запрос пользователя
        existing_queries: Уже существующие подзапросы
        target_count: Целевое количество запросов (по умолчанию 5)
        llm_factory: Фабрика для создания LLM

    Returns:
        Dict с ключами:
        - rewrites: список новых перефразов
        - total_queries: общее количество запросов после добавления
    """
    try:
        # Определяем, сколько нужно догенерировать
        current_count = len(existing_queries)
        need_count = max(0, min(target_count - current_count, 3))  # Максимум 3 за раз

        if need_count <= 0:
            return {"rewrites": [], "total_queries": current_count}

        # Получаем LLM
        if llm_factory is None:
            raise ValueError("LLM factory is required for multi_query_rewrite")

        llm = llm_factory()

        # Подготавливаем промпт
        existing_str = "\n".join(f"- {q}" for q in existing_queries)
        prompt = f"""<s>system
Ты — помощник для перефразирования запросов. Добавь ещё {need_count} альтернативных формулировок запроса.
Ответь строго JSON-массивом строк без пояснений.
Требования:
- Короткие фразы (3-8 слов)
- Разные углы зрения на тему
- Без повторов существующих
- На русском языке
</s>
<s>user
Исходный запрос: {query}

Уже есть формулировки:
{existing_str}

Добавь ещё {need_count} вариантов.
</s>
<s>bot"""

        # Получаем грамматику для массива строк нужной длины
        grammar = get_string_array_grammar(need_count)

        # Генерируем с GBNF
        response = llm(
            prompt,
            grammar=grammar,
            temperature=0.3,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.2,
            max_tokens=192,
            seed=42,
        )

        text = response["choices"][0]["text"].strip()

        # Парсим результат
        import json

        try:
            rewrites = json.loads(text) if text else []
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse multi_query_rewrite response: {text}")
            rewrites = []

        # Фильтруем и нормализуем
        cleaned_rewrites = []
        existing_normalized = {q.lower().strip() for q in existing_queries}

        for rewrite in rewrites:
            if isinstance(rewrite, str):
                normalized = rewrite.strip()
                normalized_lower = normalized.lower()

                # Проверки
                if (
                    normalized
                    and normalized_lower not in existing_normalized
                    and len(normalized.split()) >= 2
                    and len(normalized.split()) <= 12
                ):
                    cleaned_rewrites.append(normalized)
                    existing_normalized.add(normalized_lower)

        logger.info(f"Generated {len(cleaned_rewrites)} new query rewrites")

        return {
            "rewrites": cleaned_rewrites,
            "total_queries": current_count + len(cleaned_rewrites),
        }

    except Exception as e:
        logger.error(f"Error in multi_query_rewrite: {e}")
        return {"rewrites": [], "total_queries": len(existing_queries), "error": str(e)}
