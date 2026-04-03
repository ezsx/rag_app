import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any

from core.observability import observe_span
from core.settings import Settings, get_settings
from schemas.search import MetadataFilters, SearchPlan
from services.planner_prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
)

logger = logging.getLogger(__name__)


# ── Stopword dictionaries (loaded once) ──────────────────────────
def _load_stopwords() -> dict[str, list[str]]:
    """Load stopword dictionaries from datasets/planner_stopwords.json."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "datasets" / "planner_stopwords.json",
        Path("datasets/planner_stopwords.json"),
    ]
    for p in candidates:
        if p.is_file():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    logger.warning("planner_stopwords.json not found, using empty dictionaries")
    return {}


_STOPWORDS = _load_stopwords()
_SQL_KEYWORDS: list[str] = _STOPWORDS.get("sql_keywords", [])
_IMPERATIVE_RU: list[str] = _STOPWORDS.get("imperative_verbs_ru", [])
_IMPERATIVE_EN: list[str] = _STOPWORDS.get("imperative_verbs_en", [])
_FILLER_RU: list[str] = _STOPWORDS.get("filler_words_ru", [])

# Compiled regex для фильтрации (word boundary + case insensitive)
_SQL_PATTERNS = [re.compile(rf"\b{w}\b", re.IGNORECASE) for w in _SQL_KEYWORDS]
_IMPERATIVE_PATTERNS = [
    re.compile(rf"\b{w}\b", re.IGNORECASE) for w in _IMPERATIVE_RU + _IMPERATIVE_EN
]
# Единый regex для удаления стоп-слов из фраз
_STRIP_WORDS = _IMPERATIVE_RU + _IMPERATIVE_EN + _SQL_KEYWORDS + _FILLER_RU
_STRIP_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in _STRIP_WORDS) + r")\b", re.IGNORECASE) if _STRIP_WORDS else None


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

    def set(self, key: str, value: object, ttl: int | None = None):
        expires_at = time.time() + (ttl if ttl is not None else self._ttl)
        self._store[key] = (expires_at, value)


class QueryPlannerService:
    def __init__(self, llm, settings: Settings | None = None):
        self.llm = llm
        self.settings = settings or get_settings()
        # TTL: план — 10 минут, слияние — 5 минут
        self._plan_cache = _TTLCache(default_ttl_seconds=600)
        self._fusion_cache = _TTLCache(default_ttl_seconds=300)

        self._seed: int = 42

    # Строгая JSON-схема ответа планировщика (для llama.cpp json_schema)
    JSON_SCHEMA: dict[str, Any] = {
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
            "strategy": {
                "type": "string",
                "enum": ["broad", "temporal", "channel", "entity"],
            },
        },
        "required": [
            "normalized_queries",
            "must_phrases",
            "should_phrases",
            "metadata_filters",
            "k_per_query",
            "fusion",
            "strategy",
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
        """Filter SQL/DSL and imperative commands useless for dense/BM25."""
        t = (text or "").lower()
        return any(p.search(t) for p in _SQL_PATTERNS + _IMPERATIVE_PATTERNS)

    @staticmethod
    def _process_phrase(text: str) -> str:
        """Sanitize phrase: strip imperatives/SQL/filler words, cap at 12 words."""
        s = QueryPlannerService._normalize_phrase(text)
        if _STRIP_RE:
            s = _STRIP_RE.sub("", s)
        s = re.sub(r"\s+", " ", s).strip()
        words = [w for w in s.split() if w]
        if len(words) > 12:
            words = words[:12]
        return " ".join(words)

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for it in items:
            key = re.sub(r"\s+", " ", it)
            if key not in seen:
                seen.add(key)
                result.append(it)
        return result

    @staticmethod
    def _trim_date_yyyy_mm_dd(value: str | None) -> str | None:
        if not value:
            return None
        s = value.strip()
        if len(s) >= 10:
            s = s[:10]
        # Простая проверка формата YYYY-MM-DD
        try:
            datetime.strptime(s, "%Y-%m-%d")
            return s
        except (ValueError, TypeError):
            return None

    @classmethod
    def post_validate(
        cls, raw: dict[str, Any], user_query: str, settings: Settings
    ) -> SearchPlan:
        def as_list_of_str(val: Any) -> list[str]:
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
        processed: list[str] = []
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
        meta_obj: dict[str, Any] | None = None
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
                chan_ids: list[int] = []
                for x in metadata["channel_ids"]:
                    try:
                        xi = int(x)
                        if xi >= 0:
                            chan_ids.append(xi)
                    except (ValueError, TypeError):
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
            except (ValueError, TypeError):
                pass

            try:
                rt = metadata.get("reply_to")
                if rt is not None:
                    rti = int(rt)
                    if rti >= 0:
                        meta_obj["reply_to"] = rti
            except (ValueError, TypeError):
                pass

            if not meta_obj:
                meta_obj = None

        # k_per_query
        try:
            k_per_query = int(
                raw.get("k_per_query", settings.search_k_per_query_default)
            )
        except (ValueError, TypeError):
            k_per_query = settings.search_k_per_query_default
        k_per_query = max(1, min(50, k_per_query))

        fusion = str(raw.get("fusion") or settings.fusion_strategy).lower()
        fusion = "mmr" if fusion == "mmr" else "rrf"

        # Ограничение общесистемное (оставим максимум из настроек тоже)
        if len(normalized_queries) > settings.max_plan_subqueries:
            normalized_queries = normalized_queries[: settings.max_plan_subqueries]

        # Strategy: из LLM или rule override
        strategy = str(raw.get("strategy") or "broad").lower()
        if strategy not in ("broad", "temporal", "channel", "entity"):
            strategy = "broad"

        # Rule-based override: если signals с высокой confidence — доверяем regex
        from services.query_signals import extract_query_signals
        signals = extract_query_signals(user_query)
        if signals.strategy_hint and signals.confidence >= 0.8:
            strategy = signals.strategy_hint
            logger.debug(
                "QueryPlanner: rule override strategy=%s (confidence=%.2f)",
                strategy, signals.confidence,
            )
            # Дополняем metadata из signals если LLM не заполнил
            if not meta_obj:
                meta_obj = {}
            if signals.date_from and not meta_obj.get("date_from"):
                meta_obj["date_from"] = signals.date_from
            if signals.date_to and not meta_obj.get("date_to"):
                meta_obj["date_to"] = signals.date_to
            if signals.channels and not meta_obj.get("channel_usernames"):
                meta_obj["channel_usernames"] = signals.channels

        return SearchPlan(
            normalized_queries=normalized_queries,
            must_phrases=must_phrases,
            should_phrases=should_phrases,
            metadata_filters=MetadataFilters(**meta_obj) if meta_obj else None,
            k_per_query=k_per_query,
            fusion=fusion,
            strategy=strategy,
        )

    def _call_planner_llm(self, prompt: str) -> dict[str, Any]:
        """Call planner LLM with application-level timeout."""

        kwargs: dict[str, Any] = {
            "max_tokens": self.settings.planner_token_budget,
            "temperature": self.settings.planner_temp,
            "top_p": self.settings.planner_top_p,
            "top_k": getattr(self.settings, "planner_top_k", 40),
            "repeat_penalty": self.settings.planner_repeat_penalty,
            "stop": self.settings.planner_stop_list,
        }

        timeout_sec = max(float(self.settings.planner_timeout), 0.1)

        import contextvars
        ctx = contextvars.copy_context()

        # Используем chat_completion вместо completions (__call__)
        # чтобы не сбрасывать KV cache Qwen3.5 Gated Delta Networks
        # при переключении между endpoints (DEC-0039 follow-up)
        use_chat = hasattr(self.llm, "chat_completion")

        def _invoke_llm():
            if use_chat:
                system_msg, user_msg = self._split_prompt_to_messages(prompt)
                resp = self.llm.chat_completion(
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=kwargs.get("max_tokens", 512),
                    temperature=kwargs.get("temperature", 0.3),
                    top_p=kwargs.get("top_p", 0.9),
                    top_k=kwargs.get("top_k", 40),
                    seed=42,
                    response_format={"type": "json_object"},
                )
                # Адаптируем формат ответа к completions-compatible
                content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {"choices": [{"text": content}], "usage": resp.get("usage", {})}
            else:
                return self.llm(prompt, **kwargs)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(ctx.run, _invoke_llm)
            try:
                return future.result(timeout=timeout_sec)
            except FuturesTimeoutError:
                future.cancel()
                raise TimeoutError(
                    f"Planner LLM timeout after {int(timeout_sec * 1000)}ms"
                )

    def _generate_plan(self, query: str) -> SearchPlan:
        """Generate search plan via LLM (chat_completion + json_object mode)."""
        prompt = self._build_prompt(query)

        try:
            response = self._call_planner_llm(prompt)

            raw_text = response["choices"][0]["text"].strip()

            # Извлекаем первый валидный JSON объект из ответа LLM.
            # Qwen3.5 иногда добавляет лишний текст после JSON.
            raw_json = None
            for match in re.finditer(r"\{", raw_text):
                try:
                    candidate = json.loads(raw_text[match.start():])
                    raw_json = candidate
                    break
                except json.JSONDecodeError:
                    # json.loads с лишним текстом после JSON — пробуем decoder
                    decoder = json.JSONDecoder()
                    try:
                        raw_json, _ = decoder.raw_decode(raw_text, match.start())
                        break
                    except json.JSONDecodeError:
                        continue
            if raw_json is None:
                logger.warning("No valid JSON found in LLM response, using fallback")
                return self._fallback_plan(query)

            # Применяем post_validate для санитизации
            plan = self.post_validate(raw_json, query, self.settings)

            # Кешируем результат
            if self.settings.enable_cache:
                cache_key = f"plan:{hash(query)}"
                self._plan_cache.set(cache_key, plan)

            return plan

        except TimeoutError as timeout_exc:
            logger.error(
                "Planner LLM timeout for query='%s': %s",
                query[:80],
                timeout_exc,
            )
            return self._fallback_plan(query)
        except Exception as e:  # broad: LLM + JSON parsing fallback
            logger.error("_generate_plan failed for query='%s': %s", query[:80], e)
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
        with observe_span(
            "query_planner", input={"query": query[:200]},
        ) as span:
            start_ts = time.perf_counter()
            try:
                plan = self._generate_plan(query)
            except Exception as exc:  # broad: LLM + JSON parsing fallback
                logger.error("QueryPlannerService.make_plan failed: %s", exc, exc_info=True)
                raise
            took_ms = int((time.perf_counter() - start_ts) * 1000)
            if span:
                span.update(output={
                    "num_subqueries": len(plan.normalized_queries),
                    "strategy": getattr(plan, "strategy", None),
                    "took_ms": took_ms,
                })
            logger.debug(
                "QueryPlannerService.make_plan success | took_ms=%s | normalized=%s | must=%s | should=%s",
                took_ms,
                plan.normalized_queries,
                plan.must_phrases,
                plan.should_phrases,
            )
            return plan

    @staticmethod
    def _split_prompt_to_messages(prompt: str):
        """Split raw completions prompt into system + user messages."""
        # Извлекаем system content между <s>system и <s>user
        import re
        system_match = re.search(r"<s>system\s*\n(.*?)<s>user", prompt, re.DOTALL)
        user_match = re.search(r"<s>user\s*\n(.*?)(?:<s>bot|$)", prompt, re.DOTALL)
        system = system_match.group(1).strip() if system_match else ""
        user = user_match.group(1).strip() if user_match else prompt
        return system, user

    def _build_prompt(self, query: str) -> str:
        """Build prompt from templates in planner_prompts.py."""
        system = PLANNER_SYSTEM_PROMPT.format(
            k_default=self.settings.search_k_per_query_default,
            fusion_default=self.settings.fusion_strategy,
        )
        user = PLANNER_USER_PROMPT.format(query=query)
        return f"<s>system\n{system}\n</s>\n<s>user\n{user}\n</s>\n<s>bot\n"

    # Публичный доступ к кешу результатов фьюжна
    def get_cached_fusion(self, key: str):
        if not self.settings.enable_cache:
            return None
        return self._fusion_cache.get(key)

    def set_cached_fusion(self, key: str, value, ttl: int | None = 300):
        if not self.settings.enable_cache:
            return
        self._fusion_cache.set(key, value, ttl=ttl)
