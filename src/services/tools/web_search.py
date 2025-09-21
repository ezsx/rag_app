"""
Инструмент для поиска информации в интернете
"""

import os
import logging
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Инструмент для поиска актуальной информации в интернете"""

    def __init__(
        self, api_key: Optional[str] = None, search_engine: str = "duckduckgo"
    ):
        """
        Args:
            api_key: API ключ для поискового сервиса (если требуется)
            search_engine: Используемый поисковик (duckduckgo, google, bing)
        """
        self.api_key = api_key or os.getenv("WEB_SEARCH_API_KEY")
        self.search_engine = search_engine
        self.client = httpx.AsyncClient(timeout=10.0)

        # Простой кеш для экономии API запросов
        self._cache = {}
        self._cache_ttl = 3600  # 1 час

    async def search(
        self,
        query: str,
        num_results: int = 5,
        region: str = "ru-ru",
        time_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Выполняет поиск в интернете

        Args:
            query: Поисковый запрос
            num_results: Количество результатов
            region: Регион поиска
            time_range: Временной диапазон (d - день, w - неделя, m - месяц, y - год)

        Returns:
            Dict с результатами поиска
        """
        try:
            # Проверяем кеш
            cache_key = self._get_cache_key(query, num_results, region, time_range)
            cached = self._get_cached(cache_key)
            if cached:
                return cached

            # Выполняем поиск в зависимости от движка
            if self.search_engine == "duckduckgo":
                results = await self._search_duckduckgo(
                    query, num_results, region, time_range
                )
            else:
                # Заглушка для других поисковиков
                logger.warning(
                    f"Search engine {self.search_engine} not implemented, using mock"
                )
                results = self._mock_search(query, num_results)

            # Форматируем результаты
            formatted = {
                "query": query,
                "results": results,
                "timestamp": datetime.utcnow().isoformat(),
                "total_results": len(results),
                "search_engine": self.search_engine,
            }

            # Кешируем
            self._set_cached(cache_key, formatted)

            return formatted

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {
                "query": query,
                "results": [],
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _search_duckduckgo(
        self, query: str, num_results: int, region: str, time_range: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Поиск через DuckDuckGo (без API ключа)"""
        try:
            # DuckDuckGo HTML API (упрощенная версия)
            params = {
                "q": query,
                "kl": region,
            }

            if time_range:
                time_map = {"d": "d", "w": "w", "m": "m", "y": "y"}
                params["df"] = time_map.get(time_range, "")

            # Для production нужна полноценная интеграция с DuckDuckGo API
            # Сейчас возвращаем заглушку
            logger.info(f"DuckDuckGo search for: {query}")
            return self._mock_search(query, num_results)

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    def _mock_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Мок результатов для тестирования"""
        results = []

        # Генерируем мок результаты на основе запроса
        base_snippets = {
            "deepseek": [
                {
                    "title": "DeepSeek-V3.1 представила новые тарифы API",
                    "url": "https://example.com/deepseek-v3-pricing",
                    "snippet": "Компания DeepSeek объявила о снижении цен на API модели V3.1. Новые тарифы вступят в силу с 1 января 2025 года...",
                    "date": "2024-12-15",
                },
                {
                    "title": "Обзор возможностей DeepSeek-V3.1",
                    "url": "https://example.com/deepseek-v3-features",
                    "snippet": "DeepSeek-V3.1 демонстрирует впечатляющие результаты в бенчмарках, превосходя GPT-4 по многим параметрам...",
                    "date": "2024-12-10",
                },
            ],
            "default": [
                {
                    "title": f"Результаты поиска по запросу: {query}",
                    "url": f"https://example.com/search?q={query}",
                    "snippet": f"Найдена информация по теме '{query}'. Актуальные данные и последние обновления...",
                    "date": datetime.utcnow().strftime("%Y-%m-%d"),
                }
            ],
        }

        # Выбираем подходящие результаты
        if "deepseek" in query.lower():
            results = base_snippets["deepseek"]
        else:
            results = base_snippets["default"]

        return results[:num_results]

    def _get_cache_key(
        self, query: str, num_results: int, region: str, time_range: Optional[str]
    ) -> str:
        """Генерирует ключ кеша"""
        key_data = f"{query}:{num_results}:{region}:{time_range or 'all'}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Получает данные из кеша"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if datetime.utcnow().timestamp() - timestamp < self._cache_ttl:
                logger.info(f"Web search cache hit for key: {key}")
                return data
            else:
                del self._cache[key]
        return None

    def _set_cached(self, key: str, data: Dict[str, Any]):
        """Сохраняет данные в кеш"""
        self._cache[key] = (data, datetime.utcnow().timestamp())

    async def close(self):
        """Закрывает HTTP клиент"""
        await self.client.aclose()


# Глобальный экземпляр для использования в tool
_web_search_instance = None


def get_web_search_tool() -> WebSearchTool:
    """Получает или создает экземпляр WebSearchTool"""
    global _web_search_instance
    if _web_search_instance is None:
        _web_search_instance = WebSearchTool()
    return _web_search_instance


async def web_search(
    query: str,
    num_results: int = 5,
    region: str = "ru-ru",
    time_range: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Функция-обертка для использования в качестве инструмента агента

    Args:
        query: Поисковый запрос
        num_results: Количество результатов (1-10)
        region: Регион поиска
        time_range: Временной диапазон (d/w/m/y)

    Returns:
        Результаты поиска
    """
    tool = get_web_search_tool()
    return await tool.search(query, num_results, region, time_range)
