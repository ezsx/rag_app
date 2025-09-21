"""
Entity extraction tool.
Извлекает именованные сущности из текста.
"""

import re
from typing import Dict, List, Set
from collections import defaultdict


class EntityExtractor:
    """Извлекает именованные сущности из текста."""

    def __init__(self):
        # Паттерны для различных типов сущностей
        self.patterns = {
            # Email
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            # Phone
            "phone": re.compile(
                r"(?:\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
            ),
            # URL
            "url": re.compile(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
            ),
            # IP Address
            "ip": re.compile(
                r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
            ),
            # Date patterns
            "date": re.compile(
                r"\b(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})\b"
            ),
            # Time patterns
            "time": re.compile(
                r"\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?\s*(?:AM|PM|am|pm)?\b"
            ),
            # Money patterns
            "money": re.compile(
                r"(?:[$€£¥₽]\s*\d+(?:[,\.]\d+)*(?:[kKmMbB])?|\d+(?:[,\.]\d+)*\s*(?:руб|рублей|долларов?|евро|USD|EUR|RUB))"
            ),
            # Percentage
            "percentage": re.compile(r"\b\d+(?:[,\.]\d+)?\s*%"),
            # Code/ID patterns (like API keys, UUIDs, etc)
            "code": re.compile(
                r"\b[A-Z0-9]{8,}-?[A-Z0-9]{4,}-?[A-Z0-9]{4,}-?[A-Z0-9]{4,}-?[A-Z0-9]{12,}\b"
            ),
        }

        # Списки известных сущностей
        self.known_entities = {
            "tech_companies": {
                "google",
                "apple",
                "microsoft",
                "amazon",
                "facebook",
                "meta",
                "tesla",
                "nvidia",
                "intel",
                "amd",
                "oracle",
                "ibm",
                "adobe",
                "salesforce",
                "netflix",
                "uber",
                "airbnb",
                "twitter",
                "x",
            },
            "programming_languages": {
                "python",
                "javascript",
                "java",
                "c++",
                "c#",
                "ruby",
                "go",
                "rust",
                "swift",
                "kotlin",
                "typescript",
                "php",
                "r",
                "scala",
                "perl",
                "haskell",
                "erlang",
                "elixir",
                "clojure",
                "julia",
            },
            "databases": {
                "mysql",
                "postgresql",
                "mongodb",
                "redis",
                "elasticsearch",
                "cassandra",
                "sqlite",
                "oracle",
                "mssql",
                "dynamodb",
                "neo4j",
                "influxdb",
                "clickhouse",
                "mariadb",
            },
            "cloud_services": {
                "aws",
                "azure",
                "gcp",
                "google cloud",
                "digitalocean",
                "heroku",
                "vercel",
                "netlify",
                "cloudflare",
            },
            "frameworks": {
                "react",
                "angular",
                "vue",
                "django",
                "flask",
                "fastapi",
                "spring",
                "express",
                "rails",
                "laravel",
                "nextjs",
                "nuxtjs",
                "tensorflow",
                "pytorch",
                "keras",
                "scikit-learn",
            },
        }

    def extract_pattern_entities(self, text: str) -> Dict[str, List[str]]:
        """Извлекает сущности по паттернам."""
        results = defaultdict(list)

        for entity_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                # Убираем дубликаты, сохраняя порядок
                seen = set()
                unique_matches = []
                for match in matches:
                    if match not in seen:
                        seen.add(match)
                        unique_matches.append(match)
                results[entity_type] = unique_matches

        return dict(results)

    def extract_known_entities(self, text: str) -> Dict[str, List[str]]:
        """Извлекает известные сущности из текста."""
        results = defaultdict(list)
        text_lower = text.lower()

        # Токенизация - простое разбиение по не-буквенным символам
        tokens = re.findall(r"\b\w+\b", text_lower)
        token_positions = {}

        # Сохраняем позиции токенов для восстановления оригинального регистра
        for match in re.finditer(r"\b\w+\b", text):
            token_lower = match.group().lower()
            if token_lower not in token_positions:
                token_positions[token_lower] = []
            token_positions[token_lower].append(match.group())

        for category, entities in self.known_entities.items():
            found = []
            for entity in entities:
                if entity in text_lower:
                    # Восстанавливаем оригинальный регистр
                    if entity in token_positions:
                        found.extend(token_positions[entity])
                    else:
                        # Если не нашли в токенах, ищем в тексте напрямую
                        pattern = re.compile(re.escape(entity), re.IGNORECASE)
                        matches = pattern.findall(text)
                        found.extend(matches)

            if found:
                # Убираем дубликаты
                results[category] = list(dict.fromkeys(found))

        return dict(results)

    def extract_capitalized_phrases(self, text: str) -> List[str]:
        """Извлекает фразы с заглавными буквами (потенциальные имена/названия)."""
        # Паттерн для слов с заглавной буквы
        pattern = r"\b[A-ZА-Я][a-zа-я]+(?:\s+[A-ZА-Я][a-zа-я]+)*\b"
        matches = re.findall(pattern, text)

        # Фильтруем слишком короткие и слишком длинные
        filtered = []
        for match in matches:
            words = match.split()
            if 1 <= len(words) <= 4 and len(match) >= 3:
                filtered.append(match)

        # Убираем дубликаты
        return list(dict.fromkeys(filtered))

    def extract(
        self, text: str, entity_types: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Извлекает сущности из текста.

        Args:
            text: Текст для анализа
            entity_types: Список типов сущностей для извлечения (если None - все)

        Returns:
            Dict с найденными сущностями
        """
        if not text:
            return {"entities": {}, "total_count": 0, "entity_types": []}

        all_entities = {}

        # Извлекаем сущности по паттернам
        pattern_entities = self.extract_pattern_entities(text)

        # Извлекаем известные сущности
        known_entities = self.extract_known_entities(text)

        # Извлекаем потенциальные имена/названия
        capitalized = self.extract_capitalized_phrases(text)
        if capitalized:
            all_entities["names_organizations"] = capitalized

        # Объединяем результаты
        all_entities.update(pattern_entities)
        all_entities.update(known_entities)

        # Фильтруем по запрошенным типам
        if entity_types:
            filtered_entities = {}
            for entity_type in entity_types:
                if entity_type in all_entities:
                    filtered_entities[entity_type] = all_entities[entity_type]
            all_entities = filtered_entities

        # Подсчитываем общее количество
        total_count = sum(len(entities) for entities in all_entities.values())

        # Создаем сводку
        summary = []
        for entity_type, entities in all_entities.items():
            if len(entities) <= 3:
                summary.append(f"{entity_type}: {', '.join(entities)}")
            else:
                summary.append(
                    f"{entity_type}: {', '.join(entities[:3])} и еще {len(entities)-3}"
                )

        return {
            "entities": all_entities,
            "total_count": total_count,
            "entity_types": list(all_entities.keys()),
            "summary": "; ".join(summary) if summary else "Сущности не найдены",
        }


def extract_entities(
    text: str, entity_types: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Извлекает именованные сущности из текста.

    Args:
        text: Текст для анализа
        entity_types: Типы сущностей для извлечения

    Returns:
        Dict с найденными сущностями
    """
    extractor = EntityExtractor()
    return extractor.extract(text, entity_types)
