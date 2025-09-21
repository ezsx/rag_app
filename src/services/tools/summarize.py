"""
Summarization tool.
Резюмирует длинные документы, выделяя ключевые моменты.
"""

from typing import Dict, List, Optional
import re
from collections import Counter


class Summarizer:
    """Резюмирует текст, выделяя ключевые моменты."""

    def __init__(self):
        # Стоп-слова для русского и английского языков
        self.stop_words_ru = set(
            [
                "и",
                "в",
                "во",
                "не",
                "что",
                "он",
                "на",
                "я",
                "с",
                "со",
                "как",
                "а",
                "то",
                "все",
                "она",
                "так",
                "его",
                "но",
                "да",
                "ты",
                "к",
                "у",
                "же",
                "вы",
                "за",
                "бы",
                "по",
                "только",
                "ее",
                "мне",
                "было",
                "вот",
                "от",
                "меня",
                "еще",
                "нет",
                "о",
                "из",
                "ему",
                "теперь",
                "когда",
                "даже",
                "ну",
                "вдруг",
                "ли",
                "если",
                "уже",
                "или",
                "ни",
                "быть",
                "был",
                "него",
                "до",
                "вас",
                "нибудь",
                "опять",
                "уж",
                "вам",
                "ведь",
                "там",
                "потом",
                "себя",
                "ничего",
                "ей",
                "может",
                "они",
                "тут",
                "где",
                "есть",
                "надо",
                "ней",
                "для",
                "мы",
                "тебя",
                "их",
                "чем",
                "была",
                "сам",
                "чтоб",
                "без",
                "будто",
                "чего",
                "раз",
                "тоже",
                "себе",
                "под",
                "будет",
                "ж",
                "тогда",
                "кто",
                "этот",
                "того",
                "потому",
                "этого",
                "какой",
                "ним",
                "здесь",
                "этом",
                "один",
                "почти",
                "мой",
                "тем",
                "чтобы",
                "нее",
                "сейчас",
                "были",
                "куда",
                "зачем",
                "всех",
                "никогда",
                "можно",
                "при",
                "наконец",
                "два",
                "об",
                "другой",
                "хоть",
                "после",
                "над",
                "больше",
                "тот",
                "через",
                "эти",
                "нас",
                "про",
                "всего",
                "них",
                "какая",
                "много",
                "разве",
                "три",
                "эту",
                "моя",
                "впрочем",
                "хорошо",
                "свою",
                "этой",
                "перед",
                "иногда",
                "лучше",
                "чуть",
                "том",
                "нельзя",
                "такой",
                "им",
                "более",
                "всегда",
                "конечно",
                "всю",
                "между",
            ]
        )

        self.stop_words_en = set(
            [
                "the",
                "be",
                "to",
                "of",
                "and",
                "a",
                "in",
                "that",
                "have",
                "i",
                "it",
                "for",
                "not",
                "on",
                "with",
                "he",
                "as",
                "you",
                "do",
                "at",
                "this",
                "but",
                "his",
                "by",
                "from",
                "they",
                "we",
                "say",
                "her",
                "she",
                "or",
                "an",
                "will",
                "my",
                "one",
                "all",
                "would",
                "there",
                "their",
                "what",
                "so",
                "up",
                "out",
                "if",
                "about",
                "who",
                "get",
                "which",
                "go",
                "me",
                "when",
                "make",
                "can",
                "like",
                "time",
                "no",
                "just",
                "him",
                "know",
                "take",
                "person",
                "into",
                "year",
                "your",
                "good",
                "some",
                "could",
                "them",
                "see",
                "other",
                "than",
                "then",
                "now",
                "look",
                "only",
                "come",
                "its",
                "over",
                "think",
                "also",
                "back",
                "after",
                "use",
                "two",
                "how",
                "our",
                "work",
                "first",
                "well",
                "way",
                "even",
                "new",
                "want",
                "because",
                "any",
                "these",
                "give",
                "day",
                "most",
                "us",
            ]
        )

        self.stop_words = self.stop_words_ru | self.stop_words_en

    def extract_sentences(self, text: str) -> List[str]:
        """Извлекает предложения из текста."""
        # Простое разбиение по точкам с учетом сокращений
        sentences = re.split(r"(?<=[.!?])\s+", text)
        # Фильтруем короткие предложения
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def calculate_word_scores(self, text: str) -> Dict[str, float]:
        """Вычисляет веса слов на основе частоты."""
        # Токенизация
        words = re.findall(r"\b\w+\b", text.lower())

        # Фильтрация стоп-слов
        words = [w for w in words if w not in self.stop_words and len(w) > 2]

        # Подсчет частоты
        word_freq = Counter(words)

        # Нормализация частот
        max_freq = max(word_freq.values()) if word_freq else 1
        word_scores = {word: freq / max_freq for word, freq in word_freq.items()}

        return word_scores

    def score_sentences(
        self, sentences: List[str], word_scores: Dict[str, float]
    ) -> List[tuple]:
        """Оценивает важность предложений."""
        sentence_scores = []

        for sentence in sentences:
            words = re.findall(r"\b\w+\b", sentence.lower())
            words = [w for w in words if w not in self.stop_words and len(w) > 2]

            if not words:
                continue

            # Суммарный вес предложения
            score = sum(word_scores.get(word, 0) for word in words)
            # Нормализация по длине
            avg_score = score / len(words)

            sentence_scores.append((sentence, avg_score))

        return sentence_scores

    def summarize(
        self,
        text: str,
        max_sentences: int = 5,
        min_length: int = 100,
        mode: str = "extractive",
    ) -> Dict[str, any]:
        """
        Резюмирует текст.

        Args:
            text: Исходный текст
            max_sentences: Максимальное количество предложений в резюме
            min_length: Минимальная длина резюме в символах
            mode: Режим резюмирования (extractive/bullets)

        Returns:
            Dict с резюме и метаданными
        """
        if not text or len(text) < 50:
            return {
                "summary": text,
                "original_length": len(text),
                "summary_length": len(text),
                "compression_ratio": 1.0,
                "key_points": [],
                "mode": mode,
            }

        # Извлекаем предложения
        sentences = self.extract_sentences(text)
        if not sentences:
            return {
                "summary": text[:200] + "..." if len(text) > 200 else text,
                "original_length": len(text),
                "summary_length": min(len(text), 200),
                "compression_ratio": min(len(text), 200) / len(text),
                "key_points": [],
                "mode": mode,
            }

        # Вычисляем веса слов
        word_scores = self.calculate_word_scores(text)

        # Оцениваем предложения
        scored_sentences = self.score_sentences(sentences, word_scores)

        # Сортируем по важности
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # Выбираем топ предложений
        selected_sentences = []
        current_length = 0

        for sentence, score in scored_sentences:
            if (
                len(selected_sentences) >= max_sentences
                and current_length >= min_length
            ):
                break
            selected_sentences.append(sentence)
            current_length += len(sentence)

        # Восстанавливаем исходный порядок
        selected_sentences = [s for s in sentences if s in selected_sentences]

        # Формируем резюме в зависимости от режима
        if mode == "bullets":
            summary = "\n• " + "\n• ".join(selected_sentences)
            key_points = selected_sentences
        else:  # extractive
            summary = " ".join(selected_sentences)
            # Выделяем ключевые моменты (первые 3 предложения по важности)
            key_points = [s[0] for s in scored_sentences[:3]]

        # Извлекаем ключевые слова
        top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        keywords = [word for word, score in top_words]

        return {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / len(text),
            "key_points": key_points,
            "keywords": keywords,
            "mode": mode,
            "sentence_count": len(selected_sentences),
        }


def summarize(
    text: str,
    max_sentences: Optional[int] = 5,
    min_length: Optional[int] = 100,
    mode: Optional[str] = "extractive",
) -> Dict[str, any]:
    """
    Резюмирует текст, выделяя ключевые моменты.

    Args:
        text: Текст для резюмирования
        max_sentences: Максимальное количество предложений
        min_length: Минимальная длина резюме
        mode: Режим (extractive/bullets)

    Returns:
        Dict с резюме и метаданными
    """
    summarizer = Summarizer()
    return summarizer.summarize(text, max_sentences, min_length, mode)
