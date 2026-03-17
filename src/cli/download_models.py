#!/usr/bin/env python3
"""
CLI-справка по моделям для Phase 1.
"""
import argparse
import sys
from pathlib import Path

# Добавляем src в путь для импортов
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import RECOMMENDED_MODELS
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    parser = argparse.ArgumentParser(
        description="Информация о моделях для RAG системы",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Показать поддерживаемые ключи моделей
  python download_models.py

  # Показать справку по моделям
  python download_models.py --list
        """,
    )

    parser.add_argument(
        "--llm",
        choices=list(RECOMMENDED_MODELS["llm"].keys()),
        default="gpt-oss-20b",
        help="Ключ LLM модели",
    )

    parser.add_argument(
        "--embedding",
        choices=list(RECOMMENDED_MODELS["embedding"].keys()),
        default="multilingual-e5-large",
        help="Ключ embedding модели",
    )

    parser.add_argument(
        "--list", action="store_true", help="Показать список доступных моделей"
    )

    parser.add_argument(
        "--llm-only", action="store_true", help="Показать только LLM модель"
    )

    parser.add_argument(
        "--embedding-only", action="store_true", help="Показать только embedding модель"
    )

    args = parser.parse_args()

    # Показать список моделей
    if args.list:
        print("🤖 Доступные LLM модели:")
        for key, config in RECOMMENDED_MODELS["llm"].items():
            print(f"  {key}: {config['description']}")
            print(f"    Файл: {config['filename']}")
            print()

        print("📊 Доступные Embedding модели:")
        for key, config in RECOMMENDED_MODELS["embedding"].items():
            print(f"  {key}: {config['description']}")
            print(f"    Модель: {config['name']}")
            print()
        return

    print("Локальное скачивание моделей удалено в Phase 1.")
    print("LLM запускается отдельно через llama-server на Windows Host.")
    print("Embedding и reranker управляются TEI сервисами вне Docker.")
    print()

    llm_key = args.llm if not args.embedding_only else ""
    embedding_key = args.embedding if not args.llm_only else ""

    if llm_key:
        print(f"🧠 LLM модель: {RECOMMENDED_MODELS['llm'][llm_key]['description']}")
    if embedding_key:
        print(
            f"📊 Embedding модель: {RECOMMENDED_MODELS['embedding'][embedding_key]['description']}"
        )
    print()
    print("Запускать локальное скачивание больше не требуется.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
