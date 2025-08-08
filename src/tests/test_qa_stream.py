"""
Тесты для стриминга QA через Server-Sent Events
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
import json

try:
    from httpx import AsyncClient
except ImportError:
    # Для случая если httpx не установлен
    AsyncClient = None


@pytest.fixture
def mock_qa_service():
    """Мок QA сервиса для тестирования стриминга"""
    service = MagicMock()
    service.top_k = 5
    service.llm = MagicMock()

    # Мокаем async generator для stream_answer
    async def mock_stream_answer(query: str, include_context: bool = False):
        """Мок стримящего ответа"""
        test_tokens = ["Привет", ", ", "это", " ", "тестовый", " ", "ответ", "!"]
        for token in test_tokens:
            await asyncio.sleep(
                0.01
            )  # Небольшая задержка для имитации реального стрима
            yield token

    service.stream_answer = mock_stream_answer
    return service


@pytest.fixture
def mock_settings():
    """Мок настроек"""
    settings = MagicMock()
    settings.current_collection = "test_collection"
    settings.redis_enabled = False
    return settings


@pytest.fixture
def app_with_mocks(mock_qa_service, mock_settings):
    """FastAPI приложение с моками для тестирования"""
    from fastapi import FastAPI
    from api.v1.endpoints.qa import router
    from core.deps import get_qa_service, get_settings

    app = FastAPI()
    app.include_router(router, prefix="/v1")

    # Переопределяем зависимости
    app.dependency_overrides[get_qa_service] = lambda: mock_qa_service
    app.dependency_overrides[get_settings] = lambda: mock_settings

    return app


class TestQAStream:
    """Тесты для SSE стрима QA"""

    @pytest.mark.asyncio
    async def test_qa_stream_basic(self, app_with_mocks):
        """Тест базового стриминга без контекста"""
        async with AsyncClient(app=app_with_mocks, base_url="http://test") as client:
            response = await client.post(
                "/v1/qa/stream",
                json={"query": "Привет", "include_context": False},
                headers={"Accept": "text/event-stream"},
            )

            assert response.status_code == 200
            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )

            # Парсим SSE события
            events = []
            tokens = []

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Убираем "data: "
                    events.append(data)
                    if data != "[DONE]":
                        tokens.append(data)
                elif line.startswith("event: "):
                    event_type = line[7:]  # Убираем "event: "

            # Проверяем что получили токены
            assert len(tokens) >= 1
            assert "[DONE]" in events

            # Проверяем что последнее событие - это [DONE]
            assert events[-1] == "[DONE]"

    @pytest.mark.asyncio
    async def test_qa_stream_with_context(self, app_with_mocks):
        """Тест стриминга с контекстом"""
        async with AsyncClient(app=app_with_mocks, base_url="http://test") as client:
            response = await client.post(
                "/v1/qa/stream",
                json={"query": "Расскажи о тестах", "include_context": True},
                headers={"Accept": "text/event-stream"},
            )

            assert response.status_code == 200

            # Собираем все события
            events = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    events.append(data)

            # Проверяем завершение стрима
            assert "[DONE]" in events
            assert len([e for e in events if e != "[DONE]"]) >= 1  # Хотя бы один токен

    @pytest.mark.asyncio
    async def test_qa_stream_error_handling(self, mock_settings):
        """Тест обработки ошибок в стриме"""
        # Создаем QA сервис который выбрасывает ошибку
        error_service = MagicMock()
        error_service.top_k = 5
        error_service.llm = MagicMock()

        async def error_stream_answer(query: str, include_context: bool = False):
            yield "Начало"
            raise Exception("Тестовая ошибка")

        error_service.stream_answer = error_stream_answer

        from fastapi import FastAPI
        from api.v1.endpoints.qa import router
        from core.deps import get_qa_service, get_settings

        app = FastAPI()
        app.include_router(router, prefix="/v1")
        app.dependency_overrides[get_qa_service] = lambda: error_service
        app.dependency_overrides[get_settings] = lambda: mock_settings

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/v1/qa/stream",
                json={"query": "Тест ошибки", "include_context": False},
                headers={"Accept": "text/event-stream"},
            )

            assert response.status_code == 200

            # Проверяем что получили сообщение об ошибке и завершение
            events = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    events.append(line[6:])

            # Должны получить начальный токен, ошибку и [DONE]
            assert len(events) >= 2
            assert "[DONE]" in events
            # Проверяем что есть сообщение об ошибке
            error_messages = [e for e in events if "Ошибка" in e]
            assert len(error_messages) >= 1

    @pytest.mark.asyncio
    async def test_qa_stream_validation_error(self, app_with_mocks):
        """Тест валидационных ошибок"""
        async with AsyncClient(app=app_with_mocks, base_url="http://test") as client:
            # Пустой запрос - должен вызвать валидационную ошибку
            response = await client.post(
                "/v1/qa/stream",
                json={"query": "", "include_context": False},
                headers={"Accept": "text/event-stream"},
            )

            # Пустой query должен вызвать 422 ошибку валидации
            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_qa_stream_with_collection(self, mock_qa_service, mock_settings):
        """Тест стриминга с выбором коллекции"""
        # Мокаем get_retriever для переключения коллекций
        mock_retriever = MagicMock()

        from fastapi import FastAPI
        from api.v1.endpoints.qa import router
        from core.deps import get_qa_service, get_settings, get_retriever

        app = FastAPI()
        app.include_router(router, prefix="/v1")
        app.dependency_overrides[get_qa_service] = lambda: mock_qa_service
        app.dependency_overrides[get_settings] = lambda: mock_settings
        app.dependency_overrides[get_retriever] = lambda: mock_retriever

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/v1/qa/stream",
                json={
                    "query": "Тест с коллекцией",
                    "include_context": False,
                    "collection": "other_collection",
                },
                headers={"Accept": "text/event-stream"},
            )

            assert response.status_code == 200

            # Проверяем что получили токены
            token_count = 0
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    token_count += 1

            assert token_count >= 1


# Импорт функций DI для тестов
try:
    from core.deps import get_qa_service, get_settings, get_retriever
except ImportError:
    # Заглушки если импорт не удался
    def get_qa_service():
        pass

    def get_settings():
        pass

    def get_retriever():
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
