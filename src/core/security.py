"""
Модуль безопасности для защиты от различных атак
"""

import re
import logging
import hashlib
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class SecurityManager:
    """Менеджер безопасности для защиты от различных типов атак"""

    # Паттерны для обнаружения потенциально опасного контента
    SQL_PATTERNS = [
        r"\bSELECT\b.*\bFROM\b",
        r"\bINSERT\s+INTO\b",
        r"\bUPDATE\b.*\bSET\b",
        r"\bDELETE\s+FROM\b",
        r"\bDROP\s+(TABLE|DATABASE)\b",
        r"\bUNION\s+(SELECT|ALL)\b",
        r"\bEXEC(UTE)?\s*\(",
        r";\s*(SELECT|INSERT|UPDATE|DELETE|DROP)",
        r"--\s*$",  # SQL comments
        r"/\*.*\*/",  # Multi-line comments
        r"\bOR\s+1\s*=\s*1\b",
        r"\bAND\s+1\s*=\s*0\b",
    ]

    PROMPT_INJECTION_PATTERNS = [
        r"ignore\s+(previous|all)\s+(instructions?|prompts?)",
        r"disregard\s+(previous|all)\s+(instructions?|prompts?)",
        r"forget\s+(everything|all)",
        r"new\s+instructions?:",
        r"system\s*:\s*",
        r"assistant\s*:\s*",
        r"\[INST\]|\[/INST\]",  # Common instruction markers
        r"<\|im_start\|>|<\|im_end\|>",  # ChatML markers
        r"###\s*(Human|Assistant|System)",
        r"(you\s+are|act\s+as)\s+now",
        r"roleplay\s+as",
        r"pretend\s+(to\s+be|you\s+are)",
        r"bypass\s+(safety|security|restrictions?)",
        r"jailbreak",
        r"DAN\s+mode",  # "Do Anything Now"
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # Event handlers
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"<link[^>]*>",
        r"eval\s*\(",
        r"expression\s*\(",
        r"vbscript:",
        r"data:text/html",
    ]

    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./|\.\.\\\|",
        r"/etc/passwd",
        r"/etc/shadow",
        r"c:\\windows\\system32",
        r"/proc/self",
        r"\\\\[^\\]+\\[^\\]+",  # UNC paths
    ]

    def __init__(
        self,
        enable_sql_check: bool = True,
        enable_prompt_injection_check: bool = True,
        enable_xss_check: bool = True,
        enable_path_traversal_check: bool = True,
        max_input_length: int = 60000,
    ):
        self.enable_sql_check = enable_sql_check
        self.enable_prompt_injection_check = enable_prompt_injection_check
        self.enable_xss_check = enable_xss_check
        self.enable_path_traversal_check = enable_path_traversal_check
        self.max_input_length = max_input_length

    def sanitize_input(self, input_text: str, context: str = "general") -> str:
        """
        Санитизирует входной текст, удаляя потенциально опасные элементы

        Args:
            input_text: Входной текст для санитизации
            context: Контекст использования (general, query, filename, etc.)

        Returns:
            Санитизированный текст
        """
        if not input_text:
            return input_text

        # Проверка длины
        if len(input_text) > self.max_input_length:
            logger.warning(
                f"Input truncated from {len(input_text)} to {self.max_input_length}"
            )
            input_text = input_text[: self.max_input_length]

        # Удаление управляющих символов
        input_text = self._remove_control_characters(input_text)

        # Контекст-специфичная санитизация
        if context == "filename":
            input_text = self._sanitize_filename(input_text)
        elif context == "query":
            input_text = self._sanitize_query(input_text)

        # Экранирование HTML entities
        input_text = self._escape_html(input_text)

        return input_text.strip()

    def check_prompt_injection(self, text: str) -> List[str]:
        """
        Проверяет текст на наличие попыток prompt injection

        Returns:
            Список обнаруженных подозрительных паттернов
        """
        if not self.enable_prompt_injection_check:
            return []

        violations = []
        text_lower = text.lower()

        for pattern in self.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                violations.append(f"Prompt injection pattern: {pattern}")

        # Проверка на множественные роли/персоны
        role_markers = ["system:", "assistant:", "user:", "human:"]
        role_count = sum(1 for marker in role_markers if marker in text_lower)
        if role_count > 1:
            violations.append("Multiple role markers detected")

        # Проверка на попытки изменить системный промпт
        if any(
            phrase in text_lower
            for phrase in [
                "you are",
                "you must",
                "your purpose",
                "your goal",
                "forget previous",
                "ignore above",
                "disregard prior",
            ]
        ):
            violations.append("Attempt to modify system behavior")

        return violations

    def check_sql_injection(self, text: str) -> List[str]:
        """Проверяет текст на SQL injection паттерны"""
        if not self.enable_sql_check:
            return []

        violations = []

        for pattern in self.SQL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"SQL pattern detected: {pattern}")

        # Дополнительные проверки
        if text.count("'") % 2 != 0:  # Непарные кавычки
            violations.append("Unpaired quotes detected")

        if text.count(";") > 2:  # Множественные точки с запятой
            violations.append("Multiple semicolons detected")

        return violations

    def check_xss(self, text: str) -> List[str]:
        """Проверяет текст на XSS паттерны"""
        if not self.enable_xss_check:
            return []

        violations = []

        for pattern in self.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                violations.append(f"XSS pattern detected: {pattern}")

        # Проверка на HTML теги
        if re.search(r"<[^>]+>", text):
            violations.append("HTML tags detected")

        return violations

    def check_path_traversal(self, text: str) -> List[str]:
        """Проверяет текст на path traversal паттерны"""
        if not self.enable_path_traversal_check:
            return []

        violations = []

        for pattern in self.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"Path traversal pattern: {pattern}")

        return violations

    def validate_input(
        self, text: str, context: str = "general"
    ) -> tuple[bool, List[str]]:
        """
        Полная валидация входного текста

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        # Базовые проверки
        if not text:
            return True, []

        if len(text) > self.max_input_length:
            violations.append(f"Input too long: {len(text)} > {self.max_input_length}")

        # Проверки безопасности
        violations.extend(self.check_sql_injection(text))
        violations.extend(self.check_xss(text))
        violations.extend(self.check_path_traversal(text))

        if context == "prompt":
            violations.extend(self.check_prompt_injection(text))

        is_valid = len(violations) == 0

        if not is_valid:
            logger.warning(f"Input validation failed: {violations[:3]}")  # Log first 3

        return is_valid, violations

    def _remove_control_characters(self, text: str) -> str:
        """Удаляет управляющие символы кроме пробельных"""
        # Удаляем все символы с кодами 0-31 кроме \t, \n, \r
        return "".join(char for char in text if ord(char) >= 32 or char in "\t\n\r")

    def _sanitize_filename(self, filename: str) -> str:
        """Санитизирует имя файла"""
        # Удаляем опасные символы
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", filename)
        # Удаляем точки в начале
        filename = filename.lstrip(".")
        # Ограничиваем длину
        if len(filename) > 255:
            name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
            filename = name[:250] + ("." + ext if ext else "")
        return filename or "unnamed"

    def _sanitize_query(self, query: str) -> str:
        """Санитизирует поисковый запрос"""
        # Удаляем SQL-like конструкции
        query = re.sub(
            r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|EXEC)\b",
            "",
            query,
            flags=re.IGNORECASE,
        )
        # Удаляем специальные символы кроме базовых
        query = re.sub(r"[;`${}]", "", query)
        return query

    def _escape_html(self, text: str) -> str:
        """Экранирует HTML entities"""
        replacements = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#x27;",
            "/": "&#x2F;",
        }
        for char, entity in replacements.items():
            text = text.replace(char, entity)
        return text

    def hash_sensitive_data(self, data: str, salt: str = "") -> str:
        """Хеширует чувствительные данные для логирования"""
        return hashlib.sha256(f"{data}{salt}".encode()).hexdigest()[:16]

    def redact_sensitive_info(self, text: str) -> str:
        """Редактирует потенциально чувствительную информацию"""
        # Email addresses
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text
        )
        # Phone numbers
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)
        # Credit card-like numbers
        text = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD]", text)
        # IP addresses
        text = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP]", text)
        return text


# Глобальный экземпляр для использования в приложении
security_manager = SecurityManager()


def sanitize_for_logging(data: Any, max_length: int = 100) -> str:
    """Санитизирует данные для безопасного логирования"""
    if isinstance(data, dict):
        # Рекурсивно обрабатываем словарь
        sanitized = {}
        for key, value in data.items():
            if key.lower() in ["password", "token", "secret", "api_key"]:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = sanitize_for_logging(value, max_length)
        return str(sanitized)

    elif isinstance(data, (list, tuple)):
        return str([sanitize_for_logging(item, max_length) for item in data[:5]])

    elif isinstance(data, str):
        # Редактируем чувствительную информацию
        data = security_manager.redact_sensitive_info(data)
        # Обрезаем длинные строки
        if len(data) > max_length:
            return f"{data[:max_length]}... (truncated)"
        return data

    else:
        return str(data)[:max_length]
