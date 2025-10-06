import json
import logging
import re
import time
from datetime import datetime
from typing import Optional, Any, Dict, List

from core.settings import Settings, get_settings
from utils.gbnf import get_searchplan_grammar, get_string_array_grammar, gbnf_selfcheck
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

        # Фиксированный seed для стабильности генераций планировщика
        self._seed: int = 42

        # GBNF грамматика (инициализируется один раз)
        self._grammar = None
        if getattr(self.settings, "use_gbnf_planner", False):
            try:
                try:
                    gbnf_selfcheck()
                    logger.info("QueryPlanner: GBNF self-check passed")
                except Exception as _e:
                    logger.warning(f"QueryPlanner: GBNF self-check failed: {_e}")
                self._grammar = get_searchplan_grammar()
                logger.info("QueryPlanner: GBNF grammar initialized")
            except Exception as e:
                logger.warning(
                    f"QueryPlanner: GBNF init failed: {e}. Will use fallback"
                )

    # Строгая JSON-схема ответа планировщика (для llama.cpp json_schema)
    JSON_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "normalized_queries": {
                "type": "array",
                "minItems": 3,
                "maxItems": 6,
                "items": {"type": "string", "minLength": 1},
            },
            "must_phrases": {"type": "array", "items": {"type": "string"}},
            "should_phrases": {"type": "array", "items": {"type": "string"}},
            "metadata_filters": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "channel_usernames": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "channel_ids": {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 0},
                            },
                            "date_from": {
                                "type": "string",
                                "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
                            },
                            "date_to": {
                                "type": "string",
                                "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
                            },
                            "min_views": {"type": "integer", "minimum": 0},
                            "reply_to": {"type": "integer", "minimum": 0},
                        },
                    },
                ]
            },
            "k_per_query": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
            },
            "fusion": {"type": "string", "enum": ["rrf", "mmr"]},
        },
        "required": [
            "normalized_queries",
            "must_phrases",
            "should_phrases",
            "metadata_filters",
            "k_per_query",
            "fusion",
        ],
    }

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        s = (text or "").strip().lower()
        # Убираем завершающую пунктуацию/кавычки
        s = re.sub(r"[\s\u2026\.!?,;:]+$", "", s)
        s = s.strip().strip("\"'\u00ab\u00bb")
        return s

    @staticmethod
    def _is_sql_like_or_imperative(text: str) -> bool:
        """Фильтруем SQL/DSL и императивные команды, бесполезные для dense/BM25."""
        t = (text or "").lower()
        sql_patterns = [
            r"\bselect\b",
            r"\bfrom\b",
            r"\bwhere\b",
            r"\border\s+by\b",
            r"\bgroup\s+by\b",
            r"\binsert\b",
            r"\bupdate\b",
            r"\bdelete\b",
            r"\bjoin\b",
        ]
        imperative_patterns = [
            r"\bвытяни\b",
            r"\bнайди\b",
            r"\bскажи\b",
            r"\bпокажи\b",
            r"\bизвлеки\b",
            r"\bсделай\b",
        ]
        for p in sql_patterns + imperative_patterns:
            if re.search(p, t):
                return True
        return False

    @staticmethod
    def _process_phrase(text: str) -> str:
        """Санитизация фразы: убираем императивы/SQL-слова, сжимаем до 12 слов."""
        s = QueryPlannerService._normalize_phrase(text)
        s = re.sub(r"\b(вытяни|найди|скажи|покажи|извлеки|сделай)\b", "", s)
        s = re.sub(
            r"\b(select|from|where|order|group|by|insert|update|delete|join)\b", "", s
        )
        s = re.sub(r"\b(если|пожалуйста|прошу|что|когда|как)\b", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        words = [w for w in s.split(" ") if w]
        if len(words) > 12:
            words = words[:12]
        s = " ".join(words)
        return s

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        seen: set[str] = set()
        result: List[str] = []
        for it in items:
            key = re.sub(r"\s+", " ", it)
            if key not in seen:
                seen.add(key)
                result.append(it)
        return result

    @staticmethod
    def _trim_date_yyyy_mm_dd(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        s = value.strip()
        if len(s) >= 10:
            s = s[:10]
        # Простая проверка формата YYYY-MM-DD
        try:
            datetime.strptime(s, "%Y-%m-%d")
            return s
        except Exception:
            return None

    @classmethod
    def post_validate(
        cls, raw: Dict[str, Any], user_query: str, settings: Settings
    ) -> SearchPlan:
        def as_list_of_str(val: Any) -> List[str]:
            if val is None:
                return []
            if isinstance(val, str):
                val = [val]
            if isinstance(val, list):
                return [
                    cls._normalize_phrase(str(x))
                    for x in val
                    if x is not None and str(x).strip()
                ]
            return []

        normalized_queries = as_list_of_str(raw.get("normalized_queries"))
        if not normalized_queries:
            normalized_queries = [cls._normalize_phrase(user_query)]
        # Фильтрация и сжатие
        processed: List[str] = []
        for q in normalized_queries:
            qq = cls._process_phrase(q)
            if not qq or cls._is_sql_like_or_imperative(qq):
                continue
            # Отсеиваем слишком короткие однословные
            parts = [w for w in re.split(r"\s+", qq) if w]
            if len(parts) == 1 and len(parts[0]) < 3:
                continue
            processed.append(" ".join(parts))
        normalized_queries = processed
        if not normalized_queries:
            # Последний шанс: из user_query сделать ключевую фразу
            safe = cls._process_phrase(user_query)
            if safe:
                normalized_queries = [safe]
        normalized_queries = cls._dedupe_preserve_order(normalized_queries)
        # Усечение до 6 (верхняя граница схемы)
        if len(normalized_queries) > 6:
            normalized_queries = normalized_queries[:6]

        must_phrases = cls._dedupe_preserve_order(
            as_list_of_str(raw.get("must_phrases"))
        )
        should_phrases = cls._dedupe_preserve_order(
            as_list_of_str(raw.get("should_phrases"))
        )

        metadata = raw.get("metadata_filters") or None
        meta_obj: Optional[Dict[str, Any]] = None
        if isinstance(metadata, dict):
            meta_obj = {}
            if isinstance(metadata.get("channel_usernames"), list):
                usernames = [
                    cls._normalize_phrase(str(x))
                    for x in metadata["channel_usernames"]
                    if str(x).strip()
                ]
                if usernames:
                    meta_obj["channel_usernames"] = cls._dedupe_preserve_order(
                        usernames
                    )

            if isinstance(metadata.get("channel_ids"), list):
                chan_ids: List[int] = []
                for x in metadata["channel_ids"]:
                    try:
                        xi = int(x)
                        if xi >= 0:
                            chan_ids.append(xi)
                    except Exception:
                        continue
                if chan_ids:
                    meta_obj["channel_ids"] = chan_ids

            date_from = cls._trim_date_yyyy_mm_dd(metadata.get("date_from"))
            date_to = cls._trim_date_yyyy_mm_dd(metadata.get("date_to"))
            if date_from:
                meta_obj["date_from"] = date_from
            if date_to:
                meta_obj["date_to"] = date_to
            # swap при инверсии
            if date_from and date_to and date_from > date_to:
                meta_obj["date_from"], meta_obj["date_to"] = date_to, date_from

            try:
                mv = metadata.get("min_views")
                if mv is not None:
                    mvi = int(mv)
                    if mvi >= 0:
                        meta_obj["min_views"] = mvi
            except Exception:
                pass

            try:
                rt = metadata.get("reply_to")
                if rt is not None:
                    rti = int(rt)
                    if rti >= 0:
                        meta_obj["reply_to"] = rti
            except Exception:
                pass

            if not meta_obj:
                meta_obj = None

        # k_per_query
        try:
            k_per_query = int(
                raw.get("k_per_query", settings.search_k_per_query_default)
            )
        except Exception:
            k_per_query = settings.search_k_per_query_default
        k_per_query = max(1, min(50, k_per_query))

        fusion = str(raw.get("fusion") or settings.fusion_strategy).lower()
        fusion = "mmr" if fusion == "mmr" else "rrf"

        # Ограничение общесистемное (оставим максимум из настроек тоже)
        if len(normalized_queries) > settings.max_plan_subqueries:
            normalized_queries = normalized_queries[: settings.max_plan_subqueries]

        return SearchPlan(
            normalized_queries=normalized_queries,
            must_phrases=must_phrases,
            should_phrases=should_phrases,
            metadata_filters=MetadataFilters(**meta_obj) if meta_obj else None,
            k_per_query=k_per_query,
            fusion=fusion,
        )

    def _generate_plan(self, query: str) -> SearchPlan:
        """Генерирует план поиска через LLM с GBNF грамматикой."""
        prompt = self._build_prompt(query)

        try:
            # Вызов LLM с GBNF грамматикой (если включено)
            if self._grammar:
                response = self.llm(
                    prompt,
                    max_tokens=512,
                    temperature=0.1,
                    top_p=0.95,
                    grammar=self._grammar,
                    stop=["</s>", "\n\n"],
                )
            else:
                response = self.llm(
                    prompt,
                    max_tokens=512,
                    temperature=0.1,
                    top_p=0.95,
                    stop=["</s>", "\n\n"],
                )

            raw_text = response["choices"][0]["text"].strip()

            # Извлекаем JSON блок
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in LLM response, using fallback")
                return self._fallback_plan(query)

            raw_json = json.loads(json_match.group(0))

            # Применяем post_validate для санитизации
            plan = self.post_validate(raw_json, query, self.settings)

            # Кешируем результат
            if self.settings.enable_cache:
                cache_key = f"plan:{hash(query)}"
                self._plan_cache.set(cache_key, plan)

            return plan

        except Exception as e:
            logger.error(f"_generate_plan failed for query='{query[:80]}': {e}")
            return self._fallback_plan(query)

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

        logger.info(
            "QueryPlannerService.make_plan | query=%s | enable_hybrid=%s | enforce_router=%s",
            query[:80],
            self.settings.enable_hybrid_retriever,
            self.settings.enforce_router_route,
        )
        start_ts = time.perf_counter()
        try:
            plan = self._generate_plan(query)
        except Exception as exc:
            logger.error("QueryPlannerService.make_plan failed: %s", exc, exc_info=True)
            raise
        took_ms = int((time.perf_counter() - start_ts) * 1000)
        logger.debug(
            "QueryPlannerService.make_plan success | took_ms=%s | normalized=%s | must=%s | should=%s",
            took_ms,
            plan.normalized_queries,
            plan.must_phrases,
            plan.should_phrases,
        )
        return plan

    def _build_prompt(self, query: str) -> str:
        return f"""<s>system
Ты — помощник-планировщик для семантического поиска по Telegram данным. Верни строго JSON, без пояснений снаружи.

ПРАВИЛА:
- normalized_queries: 3–6 коротких подзапросов (ключевые слова/словосочетания), без глаголов и лишних слов, 3–8 слов каждый.
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

    def _generate_additional_queries(
        self, user_query: str, have: List[str], need: int
    ) -> List[str]:
        """Догенерация недостающих подзапросов через микро-GBNF массив строк.

        Возвращает новые подзапросы (без дублей и шума).
        """
        if need <= 0:
            return []
        try:
            prompt = (
                f"<s>system\n"
                f"Ты — планировщик. Добавь ещё {need} коротких нормализованных подзапросов по теме пользовательского запроса.\n"
                f"Ответь строго JSON-массивом строк без пояснений.\n"
                f"</s>\n<s>user\nЗапрос: {user_query}\nУже есть: {have}\n</s>\n<s>bot"
            )
            grammar = get_string_array_grammar(need)
            res = self.llm(
                prompt,
                grammar=grammar,
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.2,
                max_tokens=128,
                seed=self._seed,
            )
            text = (res["choices"][0].get("text") or "").strip()
            items = json.loads(text) if text else []
            have_set = {self._normalize_phrase(x) for x in have}
            result: List[str] = []
            for it in items:
                s = self._normalize_phrase(str(it))
                if not s or self._is_sql_like_or_imperative(s):
                    continue
                words = [w for w in re.split(r"\s+", s) if w]
                if len(words) > 12:
                    words = words[:12]
                if len(words) == 1 and len(words[0]) < 3:
                    continue
                norm = " ".join(words)
                if norm and norm not in have_set:
                    result.append(norm)
            return result
        except Exception as e:
            logger.warning(f"QueryPlanner[dogen] failed: {e}")
            return []

    # Публичный доступ к кешу результатов фьюжна
    def get_cached_fusion(self, key: str):
        if not self.settings.enable_cache:
            return None
        return self._fusion_cache.get(key)

    def set_cached_fusion(self, key: str, value, ttl: Optional[int] = 300):
        if not self.settings.enable_cache:
            return
        self._fusion_cache.set(key, value, ttl=ttl)
