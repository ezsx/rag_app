"""Общие утилиты и статическая мета-конфигурация моделей."""

from .prompt import build_prompt
from .ranking import rrf_merge

RECOMMENDED_MODELS = {
    "llm": {
        "qwen3.5-35b-a3b": {
            "repo": "unsloth/Qwen3.5-35B-A3B-GGUF",
            "filename": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
            "description": "Qwen3.5 35B A3B MoE (GGUF, Q4_K_M) — current production model",
        },
    },
    "embedding": {
        "pplx-embed-v1": {
            "name": "pplx-embed-v1-0.6B",
            "description": "Perplexity embedding 0.6B (bf16, mean pooling, no prefix)",
        },
    },
    "reranker": {
        "qwen3-reranker": {
            "name": "Qwen/Qwen3-Reranker-0.6B",
            "description": "Qwen3 Reranker 0.6B (seq-cls, chat template, logit scoring)",
        },
    },
}

__all__ = [
    "RECOMMENDED_MODELS",
    "build_prompt",
    "rrf_merge",
]
