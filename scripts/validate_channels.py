"""Проверка доступности Telegram-каналов через публичный preview (t.me/s/username).

Использование:
    python scripts/validate_channels.py
"""

import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Все 32 канала из research report (без отклонённых dlinnlp, machinelearning_ru, ml_world)
CHANNELS = [
    # 1 · LLM новости и релизы
    "ai_newz",
    "neurohive",
    "denissexy",
    # 2 · Научные статьи / research
    "gonzo_ml",
    "seeallochnaya",
    "dendi_math_ai",
    "complete_ai",
    # 3 · Applied ML / production
    "llm_under_hood",
    "varim_ml",
    "boris_again",
    "cryptovalerii",
    # 4 · Open-source
    "scientific_opensource",
    "ruadaptnaya",
    # 5 · AI индустрия
    "techsparks",
    "addmeto",
    "aioftheday",
    "singularityfm",
    "oulenspiegel_channel",
    # 6 · NLP
    "rybolos_channel",
    "stuffynlp",
    # 7 · MLOps
    "MLunderhood",
    # 8 · Computer Vision
    "deep_school",
    "CVML_team",
    # 9 · Data Science
    "smalldatascience",
    "inforetriever",
    # 10 · AI этика
    "theworldisnoteasy",
    # Кросс-категорийные
    "aihappens",
    "AIgobrr",
    "toBeAnMLspecialist",
    "ml_product",
    "techno_yandex",
    "atmyre_channell",
]

# Уже есть в коллекции
EXISTING = [
    "protechietich",
    "data_secrets",
    "ai_machinelearning_big_data",
    "data_easy",
    "xor_journal",
]


def check_channel(username: str) -> dict:
    """Проверяет канал через публичный preview t.me/s/username."""
    url = f"https://t.me/s/{username}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ChannelValidator/1.0)"}
    result = {"username": username, "url": url, "status": "unknown", "posts": 0, "title": ""}

    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
            result["status"] = "ok" if resp.status == 200 else f"http_{resp.status}"

            # Извлекаем название канала
            title_match = re.search(
                r'<meta property="og:title" content="([^"]+)"', html
            )
            if title_match:
                result["title"] = title_match.group(1)

            # Считаем посты на странице preview
            post_count = len(re.findall(r'class="tgme_widget_message_wrap', html))
            result["posts"] = post_count

            # Проверяем нет ли редиректа на "канал не найден"
            if "tgme_page_icon" not in html and post_count == 0:
                if "If you have <strong>Telegram</strong>" in html:
                    result["status"] = "private_or_empty"

    except HTTPError as e:
        result["status"] = f"http_{e.code}"
    except URLError as e:
        result["status"] = f"url_error: {e.reason}"
    except Exception as e:
        result["status"] = f"error: {e}"

    return result


def main():
    all_channels = EXISTING + CHANNELS
    print(f"Проверяю {len(all_channels)} каналов ({len(EXISTING)} existing + {len(CHANNELS)} new)...\n")

    ok, problems = [], []

    for i, username in enumerate(all_channels, 1):
        result = check_channel(username)
        _marker = "EXISTS" if username in EXISTING else "NEW"
        status_icon = "OK" if result["status"] == "ok" and result["posts"] > 0 else "XX"

        print(
            f"  [{i:2d}/{len(all_channels)}] {status_icon} @{username:<25s} "
            f"| {result['status']:<20s} | posts={result['posts']:>2d} "
            f"| {result['title'][:40]}"
        )

        if result["status"] == "ok" and result["posts"] > 0:
            ok.append(result)
        else:
            problems.append(result)

        # Задержка чтоб не забанили
        time.sleep(0.5)

    print(f"\n{'='*70}")
    print(f"Доступны: {len(ok)}/{len(all_channels)}")

    if problems:
        print(f"\nПроблемные каналы ({len(problems)}):")
        for p in problems:
            print(f"  XX @{p['username']:<25s} — {p['status']}")

    # Формируем список для ingest
    valid_new = [r["username"] for r in ok if r["username"] not in EXISTING]
    valid_existing = [r["username"] for r in ok if r["username"] in EXISTING]

    print(f"\n{'='*70}")
    print(f"Валидных для ingest (new): {len(valid_new)}")
    print(f"Валидных existing: {len(valid_existing)}")

    if valid_new:
        channels_str = ",".join(f"@{c}" for c in valid_new)
        print("\nIngest команда (new channels):")
        print(
            f"  docker compose -f deploy/compose/compose.dev.yml run --rm ingest "
            f'--channels "{channels_str}" '
            f"--since 2025-07-01 --until 2026-03-18"
        )

    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
