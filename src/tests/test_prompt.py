"""
Тесты для утилит промптов
"""

import pytest
from utils.prompt import build_prompt, build_simple_prompt


class TestPromptUtils:

    def test_build_prompt_with_context(self):
        """Тест создания промпта с контекстом"""
        query = "Что такое Python?"
        context = [
            "Python - это язык программирования",
            "Python используется для разработки",
        ]

        prompt = build_prompt(query, context)

        assert "Python - это язык программирования" in prompt
        assert "Что такое Python?" in prompt
        assert "<|im_start|>system" in prompt
        assert "<|im_start|>user" in prompt
        assert "<|im_start|>assistant" in prompt

    def test_build_prompt_without_context(self):
        """Тест создания промпта без контекста"""
        query = "Привет!"
        context = []

        prompt = build_prompt(query, context)

        assert "Привет!" in prompt
        assert "КОНТЕКСТ:" not in prompt
        assert "<|im_start|>system" in prompt

    def test_build_prompt_max_context_length(self):
        """Тест ограничения длины контекста"""
        query = "Тест"
        long_doc = "x" * 1500
        context = [long_doc, "короткий документ"]

        prompt = build_prompt(query, context, max_context_length=1000)

        # Проверяем, что длинный документ обрезан
        assert len(prompt) < len(long_doc) + 1000
        assert "..." in prompt  # Должно быть обрезание

    def test_build_simple_prompt(self):
        """Тест простого промпта"""
        query = "Что такое AI?"
        context = ["AI - искусственный интеллект"]

        prompt = build_simple_prompt(query, context)

        assert "Что такое AI?" in prompt
        assert "AI - искусственный интеллект" in prompt
        assert "Вопрос:" in prompt
        assert "Документ 1:" in prompt
