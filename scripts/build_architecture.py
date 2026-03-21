#!/usr/bin/env python3
"""
Сборка монолитного Architecture.md из модульных файлов.

Использование:
    python scripts/build_architecture.py
    python scripts/build_architecture.py --out docs/Architecture_ru.md
    python scripts/build_architecture.py --check  # проверка без записи

Конфигурация: docs/architecture/build_order.json
"""

import json
import sys
from pathlib import Path


def build(root: Path, check_only: bool = False, out_override: str | None = None) -> bool:
    """Собрать Architecture.md из модулей по build_order.json."""
    config_path = root / "docs" / "architecture" / "build_order.json"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        return False

    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_path = root / (out_override or config["output"])
    header_path = root / config["header"]
    module_paths = [root / m for m in config["modules"]]

    # Проверяем что все файлы существуют
    missing = []
    if not header_path.exists():
        missing.append(str(header_path))
    for p in module_paths:
        if not p.exists():
            missing.append(str(p))

    if missing:
        print(f"ERROR: missing {len(missing)} file(s):")
        for m in missing:
            print(f"  - {m}")
        return False

    # Собираем
    parts = []
    parts.append("<!-- GENERATED FILE: do not edit by hand. Edit sources in docs/architecture/ and re-run build. -->\n")
    parts.append(header_path.read_text(encoding="utf-8").rstrip())

    for p in module_paths:
        content = p.read_text(encoding="utf-8").rstrip()
        parts.append(f"\n\n---\n\n{content}")

    result = "\n".join(parts) + "\n"

    if check_only:
        if output_path.exists():
            existing = output_path.read_text(encoding="utf-8")
            if existing == result:
                print(f"OK: {output_path.name} is up to date")
                return True
            else:
                print(f"STALE: {output_path.name} differs from sources. Run without --check to rebuild.")
                return False
        else:
            print(f"MISSING: {output_path.name} does not exist. Run without --check to build.")
            return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result, encoding="utf-8")

    total_modules = len(module_paths)
    total_lines = result.count("\n")
    print(f"Built {output_path.relative_to(root)} ({total_modules} modules, {total_lines} lines)")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build Architecture.md from modular sources")
    parser.add_argument("--out", help="Override output path")
    parser.add_argument("--check", action="store_true", help="Check if output is up to date (no write)")
    parser.add_argument("--root", default=".", help="Project root directory")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    ok = build(root, check_only=args.check, out_override=args.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
