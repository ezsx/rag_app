"""
Parity check — проверяет что production config совпадает с experiments/baseline.yaml.

Запускается ПЕРЕД каждым eval прогоном (автоматически или вручную).
Exit code 0 = parity OK, exit code 1 = drift detected.

Использование:
    python scripts/parity_check.py
    python scripts/parity_check.py --fix-endpoints  # override Docker URLs to localhost
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

# Bootstrap src/ для import production settings
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def load_baseline() -> dict:
    """Загрузить experiments/baseline.yaml."""
    baseline_path = Path(__file__).resolve().parent.parent / "experiments" / "baseline.yaml"
    if not baseline_path.exists():
        print(f"FAIL: baseline.yaml не найден: {baseline_path}", file=sys.stderr)
        sys.exit(1)
    with open(baseline_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_production_config(fix_endpoints: bool = False) -> dict:
    """Получить текущий production config из settings.py."""
    if fix_endpoints:
        os.environ["QDRANT_URL"] = "http://localhost:16333"
        os.environ["EMBEDDING_TEI_URL"] = "http://localhost:8082"
        os.environ["RERANKER_TEI_URL"] = "http://localhost:8082"
        os.environ["LLM_BASE_URL"] = "http://localhost:8080"

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    if fix_endpoints:
        os.environ["QDRANT_URL"] = "http://localhost:16333"
        os.environ["EMBEDDING_TEI_URL"] = "http://localhost:8082"
        os.environ["RERANKER_TEI_URL"] = "http://localhost:8082"
        os.environ["LLM_BASE_URL"] = "http://localhost:8080"

    from core.settings import Settings
    settings = Settings()

    return {
        "retrieval": {
            "embedding_query_instruction": settings.embedding_query_instruction,
            "search_k_per_query_default": settings.search_k_per_query_default,
            "hybrid_enabled": settings.hybrid_enabled,
            "qdrant_collection": settings.qdrant_collection,
        },
        "fusion": {
            "fusion_strategy": settings.fusion_strategy,
        },
        "reranker": {
            "enable_reranker": settings.enable_reranker,
            "reranker_top_n": settings.reranker_top_n,
        },
        "endpoints": {
            "qdrant_url": settings.qdrant_url,
            "embedding_tei_url": settings.embedding_tei_url,
            "reranker_tei_url": settings.reranker_tei_url,
            "llm_base_url": settings.llm_base_url,
        },
    }


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, timeout=5,
        ).strip()
    except Exception:
        return "unknown"


def compare(baseline: dict, production: dict, allowed_changes: list[str] | None = None) -> list[str]:
    """Сравнить baseline и production config. Вернуть список расхождений."""
    allowed = set(allowed_changes or [])
    diffs = []

    for section in ["retrieval", "fusion", "reranker", "endpoints"]:
        b_section = baseline.get(section, {})
        p_section = production.get(section, {})

        for key, b_val in b_section.items():
            p_val = p_section.get(key)
            if p_val is None:
                continue

            # Нормализуем для сравнения
            if isinstance(b_val, str) and isinstance(p_val, str):
                match = b_val.strip() == p_val.strip()
            else:
                match = b_val == p_val

            if not match:
                param_path = f"{section}.{key}"
                if param_path in allowed:
                    diffs.append(f"  ALLOWED  {param_path}: {b_val!r} → {p_val!r}")
                else:
                    diffs.append(f"  DRIFT    {param_path}: baseline={b_val!r}, production={p_val!r}")

    return diffs


def main():
    parser = argparse.ArgumentParser(description="Config parity check: baseline vs production")
    parser.add_argument("--fix-endpoints", action="store_true",
                        help="Override Docker URLs to localhost (for standalone eval)")
    parser.add_argument("--allow", nargs="*", default=[],
                        help="Allowed changes, e.g. --allow retrieval.search_k_per_query_default")
    args = parser.parse_args()

    baseline = load_baseline()
    production = get_production_config(fix_endpoints=args.fix_endpoints)

    print(f"Parity check: baseline (git {baseline.get('git_sha', '?')}) vs production (git {get_git_sha()})")
    print()

    diffs = compare(baseline, production, allowed_changes=args.allow)

    if not diffs:
        print("PARITY OK — production config matches baseline")
        return 0

    has_drift = any("DRIFT" in d for d in diffs)
    for d in diffs:
        print(d, file=sys.stderr if "DRIFT" in d else sys.stdout)

    if has_drift:
        print(f"\nPARITY FAIL — {sum(1 for d in diffs if 'DRIFT' in d)} unexpected drift(s)")
        print("Если это намеренное изменение, укажите в spec.yaml what_changes и --allow", file=sys.stderr)
        return 1
    else:
        print(f"\nPARITY OK — {len(diffs)} allowed change(s)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
