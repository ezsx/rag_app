"""
Utils package for RAG system
"""

from .model_downloader import auto_download_models, RECOMMENDED_MODELS
from .prompt import build_prompt
from .ranking import rrf_merge, mmr_select

__all__ = [
    "auto_download_models",
    "RECOMMENDED_MODELS",
    "build_prompt",
    "rrf_merge",
    "mmr_select",
]
