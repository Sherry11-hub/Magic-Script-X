"""
tests/conftest.py — Pytest Configuration & Shared Fixtures
===========================================================
This file is auto-loaded by pytest before any test module runs.

Responsibilities:
  1. Add the project root to sys.path so `import config`, `from core.xxx`
     etc. resolve correctly when pytest is run from any directory.
  2. Provide session-scoped and function-scoped fixtures reused across
     all test modules.
  3. Patch heavy I/O (HuggingFace model download, ChromaDB disk writes)
     at the session level so the test suite runs fast and offline.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# ── Make the project root importable ─────────────────────────────────────────
# pytest may be invoked from the `tests/` subdirectory or the project root.
# This ensures `import config`, `from core.xxx import yyy` always work.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Environment Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_env_vars(monkeypatch):
    """
    AUTO-USED: Injects safe dummy environment variables for every test.

    This prevents tests from accidentally reading a real .env file and
    making live API calls, and ensures config.validate() has a baseline
    to work against.

    Tests that need specific values override them with their own
    monkeypatch.setattr(config, ...) calls.
    """
    import config
    monkeypatch.setattr(config, "OPENAI_API_KEY",              "sk-test-dummy-key-for-unit-tests")
    monkeypatch.setattr(config, "LLM_MODEL",                   "gpt-3.5-turbo")
    monkeypatch.setattr(config, "LLM_TEMPERATURE",             0.3)
    monkeypatch.setattr(config, "MAX_TOKENS",                  1000)
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER",          "huggingface")
    monkeypatch.setattr(config, "HUGGINGFACE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR",          "/tmp/test_chroma_db")
    monkeypatch.setattr(config, "CHROMA_COLLECTION_NAME",      "test_collection")
    monkeypatch.setattr(config, "CHUNK_SIZE",                  800)
    monkeypatch.setattr(config, "CHUNK_OVERLAP",               150)
    monkeypatch.setattr(config, "RETRIEVAL_K",                 5)


@pytest.fixture(autouse=True)
def reset_embedding_singleton():
    """
    AUTO-USED: Resets the module-level embedding model singleton before
    each test to prevent state leaking between tests.

    The singleton in document_processor.py is an optimisation for
    production; in tests it can cause unexpected cross-test coupling.

    Uses a lazy, guarded import so this fixture is a no-op when the
    full langchain stack is not installed (e.g. when only running
    config tests in CI without all dependencies).
    """
    try:
        import core.document_processor as dp
        original = dp._embedding_model
        dp._embedding_model = None      # reset before test
        yield
        dp._embedding_model = original  # restore after test
    except ImportError:
        # langchain not installed — skip singleton management silently
        yield


# ─────────────────────────────────────────────────────────────────────────────
# Fake File Fixtures
# ─────────────────────────────────────────────────────────────────────────────

class _FakeUploadedFile:
    """Minimal stand-in for streamlit.runtime.uploaded_file_manager.UploadedFile."""
    def __init__(self, name: str, content: bytes):
        self.name    = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


@pytest.fixture()
def txt_upload():
    """A realistic .txt event schedule as a fake Streamlit upload."""
    content = (
        "TECHWAVE 2025 SCHEDULE\n\n"
        "09:00 AM | Opening Ceremony         | Grand Auditorium | Sarah Chen\n"
        "10:00 AM | Keynote: The Decade of AI | Grand Auditorium | Dr. Marcus Rivera\n"
        "11:00 AM | Coffee Break              | Atrium\n"
        "11:30 AM | Panel: AI Ethics          | Grand Auditorium | Prof. Aisha Okonkwo\n"
        "01:00 PM | Networking Lunch          | Terrace Pavilion\n"
        "02:30 PM | Workshop: Building RAG    | Grand Auditorium | Dr. Priya Nair\n"
        "04:20 PM | Keynote: Web3             | Grand Auditorium | Tom Blackwell\n"
        "06:00 PM | Cocktail Reception        | Rooftop Lounge\n"
    )
    return _FakeUploadedFile("techwave_schedule.txt", content.encode("utf-8"))


@pytest.fixture()
def csv_upload():
    """A realistic .csv speaker list as a fake Streamlit upload."""
    content = (
        "name,title,organization,session,time\n"
        "Dr. Marcus Rivera,CAIO,NeuralPath Inc.,The Decade of AI,10:00 AM\n"
        "Prof. Aisha Okonkwo,Associate Professor,Stanford,AI Ethics Panel,11:30 AM\n"
        "Dr. Priya Nair,Principal Engineer,Anthropic,Building RAG Systems,02:30 PM\n"
        "Tom Blackwell,Co-Founder,ChainLink Ventures,Web3 & Open Internet,04:20 PM\n"
    )
    return _FakeUploadedFile("speakers.csv", content.encode("utf-8"))


@pytest.fixture()
def invalid_upload():
    """A .docx file — unsupported format, should be gracefully rejected."""
    return _FakeUploadedFile("report.docx", b"PK\x03\x04binary-docx-content")


@pytest.fixture()
def multi_upload(txt_upload, csv_upload, invalid_upload):
    """Mixed upload: 2 valid + 1 invalid file."""
    return [txt_upload, csv_upload, invalid_upload]


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB Mock Fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_vector_store():
    """
    Returns a pre-configured MagicMock that behaves like a populated
    Chroma vector store (50 docs, working retriever).

    Use this in tests that exercise retrieval logic without a real DB.
    """
    store = MagicMock()
    store._collection.count.return_value = 50

    # as_retriever() returns a retriever that returns 3 fake docs
    from langchain_core.documents import Document
    fake_docs = [
        Document(
            page_content=f"10:0{i} AM | Session {i} | Hall A | Speaker {i}",
            metadata={"source_file": "schedule.txt", "page": i},
        )
        for i in range(3)
    ]
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = fake_docs
    store.as_retriever.return_value = mock_retriever

    return store


@pytest.fixture()
def mock_empty_vector_store():
    """A Chroma mock with zero documents — simulates a fresh install."""
    store = MagicMock()
    store._collection.count.return_value = 0
    return store


# ─────────────────────────────────────────────────────────────────────────────
# LLM Mock Fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_llm_stream():
    """
    Returns a callable that produces a fake streaming generator.
    Simulates what ChatOpenAI(...).stream() yields in production.

    Usage in tests:
        with patch("core.rag_engine.ChatOpenAI") as mock_openai:
            mock_openai.return_value = mock_llm_stream(["Hello ", "world"])
    """
    def _factory(tokens: list[str]):
        llm = MagicMock()
        llm.__or__ = lambda self, other: llm    # support pipe operator
        llm.stream.return_value = iter(tokens)
        return llm
    return _factory


# ─────────────────────────────────────────────────────────────────────────────
# pytest markers
# ─────────────────────────────────────────────────────────────────────────────
# Register custom marks so pytest doesn't warn about unknown markers.

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring live API keys and services "
        "(skipped by default; set RUN_INTEGRATION=1 to enable)",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow-running (model download, large doc processing)",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration tests unless RUN_INTEGRATION env var is set."""
    if os.getenv("RUN_INTEGRATION"):
        return   # don't skip — user opted in

    skip_integration = pytest.mark.skip(
        reason="Integration test skipped. Set RUN_INTEGRATION=1 to run."
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
