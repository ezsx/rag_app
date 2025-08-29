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
            # JSON Schema для строгого ответа (используется в chat json_schema)
            import json as pyjson

            # Строковые поля без regex/длин — минимальная грамматика (валидация позже кодом)
            safe_string = {"type": "string"}

            schema = {
                "type": "object",
                "required": [
                    "normalized_queries",
                    "must_phrases",
                    "should_phrases",
                    "metadata_filters",
                    "k_per_query",
                    "fusion",
                ],
                "properties": {
                    "normalized_queries": {
                        "type": "array",
                        "items": safe_string,
                        "minItems": 1,
                        "maxItems": max(1, self.settings.max_plan_subqueries),
                    },
                    "must_phrases": {
                        "type": "array",
                        "items": safe_string,
                    },
                    "should_phrases": {
                        "type": "array",
                        "items": safe_string,
                    },
                    "metadata_filters": {
                        "type": ["object", "null"],
                        "properties": {
                            "channel_usernames": {
                                "type": "array",
                                "items": safe_string,
                            },
                            "channel_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                            "date_from": {
                                "type": "string",
                                "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
                            },
                            "date_to": {
                                "type": "string",
                                "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
                            },
                            "min_views": {"type": "integer"},
                            "reply_to": {"type": "integer"},
                        },
                        "additionalProperties": False,
                    },
                    "k_per_query": {"type": "integer", "minimum": 1, "maximum": 50},
                    "fusion": {"enum": ["rrf", "mmr"]},
                },
                "additionalProperties": False,
            }

            # Попытка 1: chat-completions с json_schema (без грамматики). Затем free completion с жёсткими stop.
            def _try_parse_json(s: str):
                try:
                    return json.loads(s)
                except Exception:
                    # Попробуем вытащить первую JSON-скобку
                    start = s.find("{")
                    end = s.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        return json.loads(s[start : end + 1])
                    raise

            raw = None
            # chat json_schema
            try:
                response = self.llm.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": "Ты — помощник-планировщик. Отвечай строго JSON.",
                        },
                        {
                            "role": "user",
                            "content": f"Построй поисковый план (на русском): {query}",
                        },
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=320,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "SearchPlan",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                )
                chat_text = (
                    response["choices"][0].get("message", {}).get("content") or ""
                ).strip()
                if not chat_text:
                    chat_text = (response["choices"][0].get("text") or "").strip()
                if chat_text:
                    raw = _try_parse_json(chat_text)
            except Exception:
                raw = None

            # free completion
            if raw is None:
                comp = self.llm(
                    f"{prompt}\nОтвечай строго валидным JSON без пояснений и без Markdown.",
                    max_tokens=320,
                    temperature=0.0,
                    top_p=1.0,
                    echo=False,
                    stop=["}\n", "\n\n", "</s>"],
                )
                free_text = (comp["choices"][0].get("text") or "").strip()
                raw = _try_parse_json(free_text)

            # Пост-обработка и валидация
            def _as_list_of_str(v):
                if v is None:
                    return []
                if isinstance(v, str):
                    v = [v]
                if isinstance(v, list):
                    return [
                        str(x).strip() for x in v if x is not None and str(x).strip()
                    ]
                return []

            normalized_queries = _as_list_of_str(raw.get("normalized_queries"))
            if not normalized_queries:
                normalized_queries = [query]
            must = _as_list_of_str(raw.get("must_phrases"))
            should = _as_list_of_str(raw.get("should_phrases"))
            metadata = raw.get("metadata_filters") or None
            if isinstance(metadata, dict):
                # подрежем даты до YYYY-MM-DD
                for k in ("date_from", "date_to"):
                    v = metadata.get(k)
                    if isinstance(v, str) and len(v) >= 10:
                        metadata[k] = v[:10]
                # channel_ids → int
                if isinstance(metadata.get("channel_ids"), list):
                    metadata["channel_ids"] = [
                        int(x) for x in metadata["channel_ids"] if str(x).isdigit()
                    ]
                # очищаем пустые списки
                for k in list(metadata.keys()):
                    if metadata[k] in (None, [], ""):
                        del metadata[k]
                if not metadata:
                    metadata = None
            k_per_query = raw.get("k_per_query")
            try:
                k_per_query = int(k_per_query)
            except Exception:
                k_per_query = self.settings.search_k_per_query_default
            k_per_query = max(1, min(50, k_per_query))
            fusion = str(raw.get("fusion") or self.settings.fusion_strategy).lower()

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
Ты — помощник-планировщик для семантического поиска по Telegram данным. Верни строго JSON, без пояснений снаружи.

ПРАВИЛА:
- normalized_queries: 2–5 коротких подзапросов (ключевые слова/словосочетания), без глаголов и лишних слов, 3–8 слов каждый.
- Фокус: сущности (модели, версии), предмет (что ищем), уточнения (API, тарифы, дата), без фраз типа «вытяни/скажи/если нет данных».
- must_phrases: обязательные точные маркеры (например, названия моделей: DeepSeek-V3.1, API, тарифы).
- should_phrases: полезные, но необязательные уточнения (цены, стоимость, прайс, дата вступления и т.п.).
- Если видны даты в запросе — metadata_filters.date_from/date_to (ISO YYYY-MM-DD), иначе null.
- Остальные фильтры (channel_usernames, channel_ids, min_views, reply_to) заполняй только если явно просили.
- k_per_query разумное значение (по умолчанию {self.settings.search_k_per_query_default}).
- fusion — "rrf" или "mmr" (по умолчанию {self.settings.fusion_strategy}).

ПРИМЕР:
Запрос: "Найди новости про DeepSeek-V3.1: новые цены API, когда вступают" ->
{{
  "normalized_queries": [
    "DeepSeek-V3.1 API цены",
    "изменение тарифов DeepSeek API",
    "дата вступления новых тарифов"
  ],
  "must_phrases": ["DeepSeek-V3.1", "API", "тарифы"],
  "should_phrases": ["цены", "стоимость", "дата вступления"],
  "metadata_filters": null,
  "k_per_query": {self.settings.search_k_per_query_default},
  "fusion": "{self.settings.fusion_strategy}"
}}

ФОРМАТ ОТВЕТА (ТОЛЬКО JSON):
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
