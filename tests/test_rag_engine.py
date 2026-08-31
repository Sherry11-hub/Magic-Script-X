"""
tests/test_rag_engine.py
========================
Unit tests for core/rag_engine.py.

All LLM calls and ChromaDB queries are mocked — no API keys or
running services are needed.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_docs(n=3, source="schedule.txt") -> list[Document]:
    """Returns n fake retrieved documents."""
    return [
        Document(
            page_content=f"09:{i:02d} AM | Session {i} | Hall A | Speaker {i}",
            metadata={"source_file": source, "page": i},
        )
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Test: _format_docs (context formatter)
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatDocs:

    def _call(self, docs):
        from core.rag_engine import _format_docs
        return _format_docs(docs)

    def test_empty_docs_returns_no_context_message(self):
        result = self._call([])
        assert "No relevant context" in result

    def test_each_chunk_header_contains_source_name(self):
        docs = _make_docs(n=2, source="my_schedule.txt")
        result = self._call(docs)
        assert result.count("my_schedule.txt") == 2

    def test_page_number_displayed_as_1_indexed(self):
        """Page metadata stored as 0-indexed must be displayed as 1-indexed."""
        docs = [Document(
            page_content="content",
            metadata={"source_file": "deck.pdf", "page": 0}
        )]
        result = self._call(docs)
        assert "Page 1" in result

    def test_chunk_index_labels_present(self):
        docs = _make_docs(n=3)
        result = self._call(docs)
        assert "Chunk 1" in result
        assert "Chunk 2" in result
        assert "Chunk 3" in result

    def test_page_content_preserved_in_output(self):
        docs = [Document(
            page_content="Opening Ceremony 09:00 AM Grand Auditorium",
            metadata={"source_file": "schedule.txt"}
        )]
        result = self._call(docs)
        assert "Opening Ceremony 09:00 AM Grand Auditorium" in result


# ─────────────────────────────────────────────────────────────────────────────
# Test: _extract_sources
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractSources:

    def _call(self, docs):
        from core.rag_engine import _extract_sources
        return _extract_sources(docs)

    def test_extracts_unique_sources(self):
        docs = [
            Document(page_content="a", metadata={"source_file": "sched.txt"}),
            Document(page_content="b", metadata={"source_file": "bios.txt"}),
            Document(page_content="c", metadata={"source_file": "sched.txt"}),  # duplicate
        ]
        sources = self._call(docs)
        assert sorted(sources) == ["bios.txt", "sched.txt"]

    def test_missing_metadata_uses_fallback(self):
        docs = [Document(page_content="x", metadata={})]
        sources = self._call(docs)
        assert sources == ["Unknown Source"]

    def test_empty_docs_returns_empty_list(self):
        assert self._call([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# Test: _get_retriever guard (empty KB)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRetriever:

    @patch("core.rag_engine.get_or_create_vector_store")
    @patch("core.rag_engine.get_embedding_function")
    def test_raises_runtime_error_when_kb_is_empty(
        self, mock_embed, mock_store
    ):
        """_get_retriever must raise RuntimeError if no docs are indexed."""
        mock_embed.return_value = MagicMock()
        mock_vs = MagicMock()
        mock_vs._collection.count.return_value = 0   # empty KB
        mock_store.return_value = mock_vs

        from core.rag_engine import _get_retriever
        with pytest.raises(RuntimeError, match="empty"):
            _get_retriever(k=5)

    @patch("core.rag_engine.get_or_create_vector_store")
    @patch("core.rag_engine.get_embedding_function")
    def test_k_is_clamped_to_doc_count(self, mock_embed, mock_store):
        """
        If the user requests k=10 but only 3 docs exist,
        effective_k must be 3 to avoid ChromaDB errors.
        """
        mock_embed.return_value = MagicMock()
        mock_vs = MagicMock()
        mock_vs._collection.count.return_value = 3
        mock_store.return_value = mock_vs
        mock_vs.as_retriever.return_value = MagicMock()

        from core.rag_engine import _get_retriever
        _get_retriever(k=10)

        # Check that as_retriever was called with k ≤ 3
        call_kwargs = mock_vs.as_retriever.call_args[1]
        effective_k = call_kwargs["search_kwargs"]["k"]
        assert effective_k <= 3


# ─────────────────────────────────────────────────────────────────────────────
# Test: _get_llm
# ─────────────────────────────────────────────────────────────────────────────

class TestGetLLM:

    def test_raises_value_error_when_api_key_missing(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "OPENAI_API_KEY", "")

        from core.rag_engine import _get_llm
        with pytest.raises(ValueError, match="API key"):
            _get_llm(temperature=0.3)

    @patch("core.rag_engine.ChatOpenAI")
    def test_creates_llm_with_streaming_enabled(self, mock_openai, monkeypatch):
        import config
        monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-testkey")
        monkeypatch.setattr(config, "LLM_MODEL", "gpt-3.5-turbo")
        monkeypatch.setattr(config, "MAX_TOKENS", 1000)

        from core.rag_engine import _get_llm
        _get_llm(temperature=0.5)

        _, kwargs = mock_openai.call_args
        assert kwargs.get("streaming") is True, "streaming must be True for Streamlit"

    @patch("core.rag_engine.ChatOpenAI")
    def test_temperature_is_passed_through(self, mock_openai, monkeypatch):
        import config
        monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-testkey")
        monkeypatch.setattr(config, "LLM_MODEL", "gpt-3.5-turbo")
        monkeypatch.setattr(config, "MAX_TOKENS", 1000)

        from core.rag_engine import _get_llm
        _get_llm(temperature=0.1)

        _, kwargs = mock_openai.call_args
        assert kwargs.get("temperature") == 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Test: run_script_copilot (end-to-end with full mocking)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunScriptCopilot:

    def _setup_mocks(self, monkeypatch):
        """Returns (mock_retriever, mock_chain) pre-wired for a happy path."""
        import config
        monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-testkey")
        monkeypatch.setattr(config, "LLM_MODEL", "gpt-3.5-turbo")
        monkeypatch.setattr(config, "MAX_TOKENS", 1000)
        monkeypatch.setattr(config, "RETRIEVAL_K", 5)

        mock_docs = _make_docs(n=3, source="bios.txt")
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_docs

        return mock_retriever, mock_docs

    @patch("core.rag_engine.StrOutputParser")
    @patch("core.rag_engine.ChatOpenAI")
    @patch("core.rag_engine._get_retriever")
    def test_returns_tuple_of_stream_and_sources(
        self, mock_get_retriever, mock_openai, mock_parser, monkeypatch
    ):
        """run_script_copilot must return (generator, list[str])."""
        import config
        monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(config, "LLM_MODEL", "gpt-3.5-turbo")
        monkeypatch.setattr(config, "MAX_TOKENS", 500)
        monkeypatch.setattr(config, "RETRIEVAL_K", 3)

        mock_docs = _make_docs(n=2, source="bios.txt")
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_docs
        mock_get_retriever.return_value = mock_retriever

        # Chain mock: pipe operator returns a chainable mock
        chain_mock = MagicMock()
        chain_mock.__or__ = lambda self, other: chain_mock
        chain_mock.stream.return_value = iter(["Hello ", "world"])
        mock_openai.return_value.__or__ = lambda self, other: chain_mock
        mock_parser.return_value.__ror__ = lambda self, other: chain_mock

        from core.rag_engine import run_script_copilot
        result = run_script_copilot("Write an intro for the keynote speaker")

        assert isinstance(result, tuple)
        assert len(result) == 2
        stream, sources = result
        assert isinstance(sources, list)

    @patch("core.rag_engine._get_retriever")
    def test_propagates_runtime_error_from_empty_kb(
        self, mock_get_retriever
    ):
        """If KB is empty, RuntimeError from _get_retriever must bubble up."""
        mock_get_retriever.side_effect = RuntimeError("Knowledge base is empty.")

        from core.rag_engine import run_script_copilot
        with pytest.raises(RuntimeError, match="empty"):
            run_script_copilot("Write me a script")


# ─────────────────────────────────────────────────────────────────────────────
# Test: Anti-hallucination prompt content
# ─────────────────────────────────────────────────────────────────────────────

class TestAntiHallucinationPrompts:
    """
    Validates that the critical safety instructions are present
    in both prompt templates — no code path should bypass them.
    """

    def test_script_copilot_prompt_contains_only_instruction(self):
        from core.rag_engine import SCRIPT_COPILOT_SYSTEM
        assert "ONLY" in SCRIPT_COPILOT_SYSTEM
        assert "context" in SCRIPT_COPILOT_SYSTEM.lower()
        assert "do not have enough information" in SCRIPT_COPILOT_SYSTEM.lower()

    def test_schedule_adapter_prompt_contains_only_instruction(self):
        from core.rag_engine import SCHEDULE_ADAPTER_SYSTEM
        assert "EXCLUSIVELY" in SCHEDULE_ADAPTER_SYSTEM or "ONLY" in SCHEDULE_ADAPTER_SYSTEM
        assert "context" in SCHEDULE_ADAPTER_SYSTEM.lower()
        assert "could not find" in SCHEDULE_ADAPTER_SYSTEM.lower()

    def test_script_copilot_prompt_forbids_fabrication(self):
        from core.rag_engine import SCRIPT_COPILOT_SYSTEM
        # The word "fabricate" or "invent" must appear to explicitly bar it
        forbidden_terms = ["fabricate", "invent", "NOT invent", "Do NOT invent"]
        assert any(t.lower() in SCRIPT_COPILOT_SYSTEM.lower() for t in forbidden_terms)

    def test_schedule_adapter_prompt_has_output_format_spec(self):
        """The schedule adapter must specify an output format to be parseable."""
        from core.rag_engine import SCHEDULE_ADAPTER_SYSTEM
        # Check for the numbered list format instruction
        assert "HH:MM" in SCHEDULE_ADAPTER_SYSTEM or "Time" in SCHEDULE_ADAPTER_SYSTEM
