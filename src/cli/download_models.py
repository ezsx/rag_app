#!/usr/bin/env python3
"""
CLI скрипт для скачивания моделей
"""
import argparse
import os
import sys
from pathlib import Path

# Добавляем src в путь для импортов
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_downloader import auto_download_models, RECOMMENDED_MODELS
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    parser = argparse.ArgumentParser(
        description="Скачивание моделей для RAG системы",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Скачать рекомендуемые модели для русского языка
  python download_models.py

  # Скачать конкретные модели
      python download_models.py --llm vikhr-7b-instruct --embedding multilingual-e5-large

  # Показать доступные модели
  python download_models.py --list

  # Скачать в определенную папку
  python download_models.py --models-dir /path/to/models
        """,
    )

    parser.add_argument(
        "--llm",
        choices=list(RECOMMENDED_MODELS["llm"].keys()),
        default="vikhr-7b-instruct",
        help="LLM модель для скачивания",
    )

    parser.add_argument(
        "--embedding",
        choices=list(RECOMMENDED_MODELS["embedding"].keys()),
        default="multilingual-e5-large",
        help="Embedding модель для скачивания",
    )

    parser.add_argument(
        "--models-dir", default="/models", help="Директория для сохранения моделей"
    )

    parser.add_argument(
        "--cache-dir", default="/models/.cache", help="Директория кэша HuggingFace"
    )

    parser.add_argument(
        "--list", action="store_true", help="Показать список доступных моделей"
    )

    parser.add_argument(
        "--llm-only", action="store_true", help="Скачать только LLM модель"
    )

    parser.add_argument(
        "--embedding-only", action="store_true", help="Скачать только embedding модель"
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

    # Создаем директории
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    print("🚀 Запуск скачивания моделей...")
    print(f"📁 Директория моделей: {args.models_dir}")
    print(f"💾 Кэш: {args.cache_dir}")
    print()

    # Определяем что скачивать
    llm_key = args.llm if not args.embedding_only else ""
    embedding_key = args.embedding if not args.llm_only else ""

    if llm_key:
        print(f"🧠 LLM модель: {RECOMMENDED_MODELS['llm'][llm_key]['description']}")
    if embedding_key:
        print(
            f"📊 Embedding модель: {RECOMMENDED_MODELS['embedding'][embedding_key]['description']}"
        )
    print()

    # Скачиваем модели
    llm_path, embedding_success = auto_download_models(
        llm_model_key=llm_key,
        embedding_model_key=embedding_key,
        models_dir=args.models_dir,
        cache_dir=args.cache_dir,
    )

    # Результаты
    print("\n" + "=" * 50)
    print("📊 РЕЗУЛЬТАТЫ СКАЧИВАНИЯ:")

    if llm_key:
        if llm_path:
            print(f"✅ LLM модель: {llm_path}")
        else:
            print("❌ LLM модель: не удалось скачать")

    if embedding_key:
        if embedding_success:
            print(f"✅ Embedding модель: успешно скачана")
        else:
            print("❌ Embedding модель: не удалось скачать")

    if (not llm_key or llm_path) and (not embedding_key or embedding_success):
        print("\n🎉 Все модели успешно скачаны!")
        print("\nТеперь можете запустить API:")
        print("docker compose --profile api up")
        return 0
    else:
        print("\n⚠️ Некоторые модели не удалось скачать")
        return 1


if __name__ == "__main__":
    sys.exit(main())
