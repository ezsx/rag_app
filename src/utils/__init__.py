"""Общие утилиты и статическая мета-конфигурация моделей."""

from .prompt import build_prompt
from .ranking import mmr_select, rrf_merge

RECOMMENDED_MODELS = {
    "llm": {
        "gpt-oss-20b": {
            "repo": "unsloth/gpt-oss-20b-GGUF",
            "filename": "gpt-oss-20b-Q6_K.gguf",
            "description": "OpenAI gpt-oss-20b (GGUF, Q6_K) от Unsloth",
        },
        "vikhr-7b-instruct": {
            "repo": "oblivious/Vikhr-7B-instruct-GGUF",
            "filename": "Vikhr-7B-instruct-Q4_K_M.gguf",
            "description": "Vikhr 7B - русскоязычная модель от Vikhrmodels",
        },
        "qwen2.5-7b-instruct": {
            "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "filename": "Qwen2.5-7B-Instruct-Q8_0.gguf",
            "description": "Qwen2.5 7B Instruct (GGUF, Q8_0)",
        },
        "qwen2.5-3b-instruct": {
            "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
            "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
            "description": "Qwen2.5 3B Instruct (GGUF, Q4_K_M)",
        },
        "saiga-mistral-7b": {
            "repo": "IlyaGusev/saiga_mistral_7b_gguf",
            "filename": "model-q4_K.gguf",
            "description": "Saiga Mistral 7B - дообученная модель для русского",
        },
        "openchat-3.6-8b": {
            "repo": "openchat/openchat-3.6-8b-20240522-GGUF",
            "filename": "openchat-3.6-8b-20240522-q4_k_m.gguf",
            "description": "OpenChat 3.6 8B - универсальная модель",
        },
    },
    "embedding": {
        "multilingual-e5-large": {
            "name": "intfloat/multilingual-e5-large",
            "description": "Лучшая многоязычная embedding модель",
        },
        "multilingual-mpnet": {
            "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "description": "Быстрая многоязычная модель",
        },
        "bge-m3": {
            "name": "BAAI/bge-m3",
            "description": "BGE M3 - многоязычная embedding модель",
        },
    },
    "reranker": {
        "bge-reranker-v2-m3": {
            "name": "BAAI/bge-reranker-v2-m3",
            "description": "BAAI bge-reranker-v2-m3 (TEI service)",
        }
    },
}

__all__ = [
    "RECOMMENDED_MODELS",
    "build_prompt",
    "mmr_select",
    "rrf_merge",
]
