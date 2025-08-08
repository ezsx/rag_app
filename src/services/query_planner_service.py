import json
import logging
import time
from typing import Optional

from core.settings import Settings, get_settings
from schemas.search import SearchPlan, SearchPlanRequest, MetadataFilters


logger = logging.getLogger(__name__)


class _TTLCache:
    def __init__(self, default_ttl_seconds: int):
        self._store: dict[str, tuple[float, object]] = {}
        self._ttl = default_ttl_seconds

    def get(self, key: str):
        now = time.time()
        value = self._store.get(key)
        if not value:
            return None
        expires_at, payload = value
        if expires_at < now:
            self._store.pop(key, None)
            return None
        return payload

    def set(self, key: str, value: object, ttl: Optional[int] = None):
        expires_at = time.time() + (ttl if ttl is not None else self._ttl)
        self._store[key] = (expires_at, value)


class QueryPlannerService:
    def __init__(self, llm, settings: Optional[Settings] = None):
        self.llm = llm
        self.settings = settings or get_settings()
        # TTL: план — 10 минут, слияние — 5 минут
        self._plan_cache = _TTLCache(default_ttl_seconds=600)
        self._fusion_cache = _TTLCache(default_ttl_seconds=300)

    def _fallback_plan(self, query: str) -> SearchPlan:
        return SearchPlan(
            normalized_queries=[query],
            must_phrases=[],
            should_phrases=[],
            metadata_filters=None,
            k_per_query=self.settings.search_k_per_query_default,
            fusion="rrf",
        )

    def make_plan(self, query: str) -> SearchPlan:
        if not query or not query.strip():
            return self._fallback_plan(query)

        cache_key = f"plan:{hash(query)}"
        if self.settings.enable_cache:
            cached = self._plan_cache.get(cache_key)
            if cached:
                return cached  # type: ignore

        prompt = self._build_prompt(query)

        t0 = time.time()
        try:
            response = self.llm(
                prompt,
                max_tokens=512,
                temperature=0.0,
                top_p=1.0,
                stop=["</s>", "\n\nВопрос:", "\n\nUser:", "<s>"],
                echo=False,
            )
            text = response["choices"][0]["text"].strip()
            # Жесткий парсинг JSON
            raw = json.loads(text)

            # Пост-обработка и валидация
            normalized_queries = raw.get("normalized_queries") or []
            must = raw.get("must_phrases") or []
            should = raw.get("should_phrases") or []
            metadata = raw.get("metadata_filters") or None
            k_per_query = (
                raw.get("k_per_query") or self.settings.search_k_per_query_default
            )
            fusion = (raw.get("fusion") or self.settings.fusion_strategy).lower()

            # Ограничение числа под-запросов
            if len(normalized_queries) > self.settings.max_plan_subqueries:
                normalized_queries = normalized_queries[
                    : self.settings.max_plan_subqueries
                ]

            plan = SearchPlan(
                normalized_queries=normalized_queries,
                must_phrases=must,
                should_phrases=should,
                metadata_filters=MetadataFilters(**metadata) if metadata else None,
                k_per_query=int(k_per_query),
                fusion="mmr" if fusion == "mmr" else "rrf",
            )

        except Exception as e:
            logger.warning(
                f"Ошибка планировщика (парсинг/LLM): {e}. Используем fallback"
            )
            plan = self._fallback_plan(query)

        t1 = time.time()
        logger.info(
            f"QueryPlanner: план за {int((t1 - t0)*1000)} мс, под-запросов: {len(plan.normalized_queries)}"
        )

        if self.settings.enable_cache:
            self._plan_cache.set(cache_key, plan, ttl=600)

        return plan

    def _build_prompt(self, query: str) -> str:
        return f"""<s>system
Ты — помощник-планировщик для семантического поиска по Telegram данным. Задача — построить ПОИСКОВЫЙ ПЛАН строго в формате JSON на русском языке.

ТРЕБОВАНИЯ:
- Только JSON, без комментариев и текста вокруг
- 1-5 нормализованных под-запросов на русском языке
- Если в запросе есть явные/неявные даты — выставь date_from/date_to (ISO)
- must_phrases/should_phrases — ключевые слова/фразы
- metadata_filters может содержать: channel_usernames, channel_ids, date_from, date_to, min_views, reply_to
- Если данные неизвестны — ставь null или пустые массивы
- k_per_query разумное значение (по умолчанию {self.settings.search_k_per_query_default})
- fusion — "rrf" или "mmr" (по умолчанию {self.settings.fusion_strategy})

ФОРМАТ ОТВЕТА (ТОЛЬКО JSON!):
{{
  "normalized_queries": ["..."],
  "must_phrases": ["..."],
  "should_phrases": ["..."],
  "metadata_filters": {{
      "channel_usernames": ["@..."],
      "channel_ids": [123],
      "date_from": "YYYY-MM-DD",
      "date_to": "YYYY-MM-DD",
      "min_views": 0,
      "reply_to": 123
  }},
  "k_per_query": {self.settings.search_k_per_query_default},
  "fusion": "{self.settings.fusion_strategy}"
}}
</s>
<s>user
Построй поисковый план для запроса (на русском): {query}
</s>
<s>bot
"""

    # Публичный доступ к кешу результатов фьюжна
    def get_cached_fusion(self, key: str):
        if not self.settings.enable_cache:
            return None
        return self._fusion_cache.get(key)

    def set_cached_fusion(self, key: str, value, ttl: Optional[int] = 300):
        if not self.settings.enable_cache:
            return
        self._fusion_cache.set(key, value, ttl=ttl)
