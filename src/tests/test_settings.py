"""Tests for Settings from core/settings.py."""

from __future__ import annotations

import pytest

from core.settings import Settings


@pytest.fixture()
def settings() -> Settings:
    """Fresh Settings instance (not the lru_cache singleton)."""
    return Settings()


class TestSettingsDefaults:
    """Default values load correctly without .env overrides."""

    def test_default_llm_key(self, settings: Settings) -> None:
        assert settings.current_llm_key == "qwen3-30b-a3b"

    def test_default_qdrant_collection(self, settings: Settings) -> None:
        # .env may override; field default is "news_colbert"
        assert isinstance(settings.qdrant_collection, str)
        assert len(settings.qdrant_collection) > 0

    def test_default_coverage_threshold(self, settings: Settings) -> None:
        assert settings.coverage_threshold == 0.75

    def test_default_agent_max_steps(self, settings: Settings) -> None:
        assert settings.agent_max_steps == 15


class TestSettingsEnvOverride:
    """Environment variables override defaults."""

    def test_env_override_collection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QDRANT_COLLECTION", "test_collection")
        s = Settings()
        assert s.qdrant_collection == "test_collection"

    def test_env_override_llm_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_MODEL_KEY", "test-model")
        s = Settings()
        assert s.current_llm_key == "test-model"


class TestSettingsProperties:
    """Computed properties."""

    def test_planner_stop_list_single(self, settings: Settings) -> None:
        # Default is "Observation:" (no ||) -> single-element list
        assert settings.planner_stop_list == ["Observation:"]

    def test_planner_stop_list_splits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PLANNER_STOP", "Observation:||Action:||End:")
        s = Settings()
        assert s.planner_stop_list == ["Observation:", "Action:", "End:"]

    def test_current_collection_alias(self, settings: Settings) -> None:
        assert settings.current_collection == settings.qdrant_collection

    def test_enable_hybrid_retriever_alias(self, settings: Settings) -> None:
        assert settings.enable_hybrid_retriever == settings.hybrid_enabled
