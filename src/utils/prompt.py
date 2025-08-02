"""
Утилиты для работы с промптами для LLM
"""

from typing import List


def build_prompt(query: str, context: List[str], max_context_length: int = 2000) -> str:
    """
    Строит промпт для LLM на основе запроса пользователя и контекста
    Оптимизирован для модели Vikhr (использует формат <s>{role}\n{content}</s>)

    Args:
        query: Вопрос пользователя
        context: Список найденных документов
        max_context_length: Максимальная длина контекста в символах

    Returns:
        Сформированный промпт для LLM
    """
    if not context:
        # Промпт без контекста для модели Vikhr
        return f"""<s>system
Ты — Вихрь, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им. Отвечай грамотно и подробно на русском языке.
</s>
<s>user
{query}
</s>
<s>bot
"""

    # Объединяем контекст, ограничивая длину
    context_text = ""
    current_length = 0
    docs_included = 0

    for i, doc in enumerate(context):
        doc_length = len(doc)
        separator_length = 50  # Длина разделителя

        if current_length + doc_length + separator_length > max_context_length:
            # Если добавление документа превысит лимит, пытаемся обрезать его
            remaining_space = max_context_length - current_length - separator_length
            if remaining_space > 200:  # Минимум 200 символов для значимого текста
                context_text += f"\n\n=== ДОКУМЕНТ {i+1} (обрезан) ===\n"
                context_text += doc[:remaining_space] + "...\n"
                docs_included += 1
            break
        else:
            context_text += f"\n\n=== ДОКУМЕНТ {i+1} ===\n"
            context_text += doc + "\n"
            current_length += doc_length + separator_length
            docs_included += 1

            # Промпт для модели Vikhr с контекстом
        prompt = f"""<s>system
Ты — Вихрь, русскоязычный автоматический ассистент. Ты специализируешься на анализе и обобщении информации. 

КОНТЕКСТ (найдено {docs_included} релевантных документов):{context_text}

ИНСТРУКЦИИ:
1. Отвечай ТОЛЬКО на русском языке
2. Используй ИСКЛЮЧИТЕЛЬНО информацию из предоставленного контекста
3. Если в контексте недостаточно информации для полного ответа, честно об этом скажи
4. Структурируй ответ логично и читабельно
5. Не добавляй информацию, которой нет в контексте
6. Если вопрос не связан с контекстом, скажи об этом прямо

ФОРМАТ ОТВЕТА:
- Краткий, но исчерпывающий ответ
- При необходимости используй нумерованные списки или маркеры
- Ссылайся на конкретные документы при необходимости
</s>
<s>user
{query}
</s>
<s>bot
"""

    return prompt


def build_simple_prompt(query: str, context: List[str]) -> str:
    """
    Упрощенная версия промпта для базовых LLM без поддержки ChatML
    Оптимизирована для русского языка
    """
    if not context:
        return f"""Вопрос: {query}

Ответ на русском языке:"""

    # Ограничиваем до 3 самых релевантных документов
    context_text = "\n\n".join(
        [f"ДОКУМЕНТ {i+1}:\n{doc}" for i, doc in enumerate(context[:3])]
    )

    return f"""КОНТЕКСТ:
{context_text}

ЗАДАЧА: Ответь на вопрос, используя только информацию из контекста выше.

ВОПРОС: {query}

ОТВЕТ (на русском языке):"""
