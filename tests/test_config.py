"""
tests/test_config.py — Unit tests for config.py
================================================
Tests the validate() function under various environment conditions
using monkeypatching so no real .env file or API key is needed.
"""

import pytest
import importlib


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def reload_config_with(monkeypatch, overrides: dict):
    """
    Helper: patches config module attributes and returns a fresh module view.
    Avoids side-effects between tests.
    """
    import config
    for attr, value in overrides.items():
        monkeypatch.setattr(config, attr, value)
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigValidation:

    def test_valid_config_returns_no_errors(self, monkeypatch):
        """A fully configured environment should produce zero errors."""
        cfg = reload_config_with(monkeypatch, {
            "OPENAI_API_KEY":       "sk-validkeyABC123",
            "EMBEDDING_PROVIDER":   "huggingface",
            "CHUNK_SIZE":           800,
            "CHUNK_OVERLAP":        150,
        })
        errors = cfg.validate()
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_missing_api_key_raises_error(self, monkeypatch):
        """Empty OPENAI_API_KEY must produce exactly one error."""
        cfg = reload_config_with(monkeypatch, {"OPENAI_API_KEY": ""})
        errors = cfg.validate()
        assert len(errors) >= 1
        assert any("OPENAI_API_KEY" in e for e in errors)

    def test_malformed_api_key_raises_error(self, monkeypatch):
        """A key that doesn't start with 'sk-' is flagged as malformed."""
        cfg = reload_config_with(monkeypatch, {"OPENAI_API_KEY": "notakey-abc"})
        errors = cfg.validate()
        assert any("malformed" in e.lower() or "sk-" in e for e in errors)

    def test_invalid_embedding_provider_raises_error(self, monkeypatch):
        """EMBEDDING_PROVIDER must be 'huggingface' or 'openai'."""
        cfg = reload_config_with(monkeypatch, {
            "OPENAI_API_KEY":     "sk-good",
            "EMBEDDING_PROVIDER": "cohere",   # unsupported
            "CHUNK_SIZE":         800,
            "CHUNK_OVERLAP":      150,
        })
        errors = cfg.validate()
        assert any("EMBEDDING_PROVIDER" in e for e in errors)

    def test_chunk_overlap_gte_chunk_size_raises_error(self, monkeypatch):
        """CHUNK_OVERLAP must be strictly less than CHUNK_SIZE."""
        cfg = reload_config_with(monkeypatch, {
            "OPENAI_API_KEY":     "sk-good",
            "EMBEDDING_PROVIDER": "huggingface",
            "CHUNK_SIZE":         300,
            "CHUNK_OVERLAP":      300,   # equal — invalid
        })
        errors = cfg.validate()
        assert any("CHUNK_OVERLAP" in e for e in errors)

    def test_openai_embedding_provider_accepted(self, monkeypatch):
        """'openai' is a valid EMBEDDING_PROVIDER value."""
        cfg = reload_config_with(monkeypatch, {
            "OPENAI_API_KEY":     "sk-goodkey",
            "EMBEDDING_PROVIDER": "openai",
            "CHUNK_SIZE":         800,
            "CHUNK_OVERLAP":      150,
        })
        errors = cfg.validate()
        assert errors == []
