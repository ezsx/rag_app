"""
Translate tool.
Перевод текста между языками. При наличии llm_factory использует LLM,
иначе применяет простой эвристический перевод RU↔EN для базовых слов.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, Optional


_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


def _detect_lang(text: str) -> str:
    if _CYRILLIC_RE.search(text):
        return "ru"
    return "en"


def _naive_ru_en_translate(text: str, src: str, tgt: str) -> str:
    # Небольшой словарь для базовой поддержки без внешних зависимостей
    ru_en = {
        "привет": "hello",
        "мир": "world",
        "как": "how",
        "дела": "are you",
        "спасибо": "thank you",
        "пожалуйста": "please",
        "вопрос": "question",
        "ответ": "answer",
        "документ": "document",
        "поиск": "search",
        "проверка": "verification",
        "время": "time",
        "дата": "date",
    }
    en_ru = {v: k for k, v in ru_en.items()}

    words = re.findall(r"\w+|\W+", text)
    out: list[str] = []
    if src == "ru" and tgt == "en":
        for w in words:
            lw = w.lower()
            if lw in ru_en:
                repl = ru_en[lw]
                repl = repl.capitalize() if w[:1].isupper() else repl
                out.append(repl)
            else:
                out.append(w)
    elif src == "en" and tgt == "ru":
        for w in words:
            lw = w.lower()
            if lw in en_ru:
                repl = en_ru[lw]
                repl = repl.capitalize() if w[:1].isupper() else repl
                out.append(repl)
            else:
                out.append(w)
    else:
        # Неизвестная пара — возвращаем исходный текст
        return text

    return "".join(out)


def translate(
    text: str,
    target_lang: str,
    source_lang: Optional[str] = None,
    max_length: int = 1000,
    llm_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    start = time.perf_counter()

    if not text:
        return {
            "translated_text": "",
            "source_lang": source_lang or "unknown",
            "target_lang": target_lang,
            "mode": "na",
            "took_ms": int((time.perf_counter() - start) * 1000),
        }

    src = (source_lang or _detect_lang(text)).lower()
    tgt = (target_lang or "en").lower()

    # Ограничиваем длину
    truncated = text[: max(1, int(max_length))]

    # Пытаемся использовать LLM при наличии
    if llm_factory is not None:
        try:
            llm = llm_factory()
            prompt = (
                f"Translate the following text from {src} to {tgt}. "
                f"Output only the translation, no explanations.\n\n" + truncated
            )
            resp = llm.create_completion(
                prompt=prompt,
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                max_tokens=512,
                repeat_penalty=1.2,
                stop=["\n\n"],
                seed=42,
            )
            text_out = resp.get("choices", [{}])[0].get("text", "").strip()
            return {
                "translated_text": text_out,
                "source_lang": src,
                "target_lang": tgt,
                "mode": "llm",
                "took_ms": int((time.perf_counter() - start) * 1000),
            }
        except Exception:
            # Фолбэк на наивный перевод
            pass

    naive = _naive_ru_en_translate(truncated, src, tgt)
    return {
        "translated_text": naive,
        "source_lang": src,
        "target_lang": tgt,
        "mode": "naive",
        "took_ms": int((time.perf_counter() - start) * 1000),
    }
