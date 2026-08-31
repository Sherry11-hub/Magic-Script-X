"""
tests/test_document_processor.py
=================================
Unit tests for core/document_processor.py.

Strategy:
  - No real ChromaDB writes or network calls in unit tests.
  - Mocking is done with pytest monkeypatch and unittest.mock.
  - Integration-style tests (marked @pytest.mark.integration) are skipped
    in CI unless the RUN_INTEGRATION env var is set.
"""

import io
import os
import pytest
from unittest.mock import MagicMock, patch, call
from langchain_core.documents import Document

# We import document_processor after patching heavy dependencies
# to avoid downloading ML models during the test run.


# ─────────────────────────────────────────────────────────────────────────────
# Fake UploadedFile helper
# ─────────────────────────────────────────────────────────────────────────────

class FakeUploadedFile:
    """
    Mimics a Streamlit UploadedFile object for testing.
    Streamlit's real class requires a running server context.
    """
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def make_txt_file(name="schedule.txt", text="09:00 AM | Keynote | Hall A") -> FakeUploadedFile:
    return FakeUploadedFile(name, text.encode("utf-8"))

def make_csv_file(name="speakers.csv") -> FakeUploadedFile:
    csv_content = b"name,title,session\nDr. Smith,CEO,Keynote\nProf. Lee,CTO,Workshop"
    return FakeUploadedFile(name, csv_content)


# ─────────────────────────────────────────────────────────────────────────────
# Test: load_document_from_upload
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadDocumentFromUpload:

    def test_txt_file_loads_successfully(self):
        """A valid .txt file should return at least one Document."""
        from core.document_processor import load_document_from_upload
        uf = make_txt_file(text="Welcome to TechWave 2025!\nVenue: Grand Hall.")
        docs = load_document_from_upload(uf)
        assert isinstance(docs, list)
        assert len(docs) >= 1
        assert isinstance(docs[0], Document)

    def test_txt_metadata_includes_source_file(self):
        """Loaded documents must carry the original filename in metadata."""
        from core.document_processor import load_document_from_upload
        uf = make_txt_file(name="my_schedule.txt")
        docs = load_document_from_upload(uf)
        assert all(d.metadata.get("source_file") == "my_schedule.txt" for d in docs)

    def test_txt_metadata_includes_file_type(self):
        from core.document_processor import load_document_from_upload
        uf = make_txt_file(name="event.txt")
        docs = load_document_from_upload(uf)
        assert all(d.metadata.get("file_type") == "txt" for d in docs)

    def test_unsupported_extension_raises_value_error(self):
        """Extensions outside .txt/.pdf/.csv must raise ValueError."""
        from core.document_processor import load_document_from_upload
        uf = FakeUploadedFile("document.docx", b"some bytes")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document_from_upload(uf)

    def test_csv_file_loads_successfully(self):
        """A valid .csv file should produce one Document per data row."""
        from core.document_processor import load_document_from_upload
        uf = make_csv_file()
        docs = load_document_from_upload(uf)
        # CSVLoader creates 1 doc per row (excluding header) = 2 rows
        assert len(docs) == 2

    def test_temp_file_is_cleaned_up_after_load(self, tmp_path):
        """The temporary file must not persist after loading."""
        import tempfile
        original_mktemp = tempfile.NamedTemporaryFile
        created_paths = []

        def tracking_mktemp(**kwargs):
            f = original_mktemp(**kwargs)
            created_paths.append(f.name)
            return f

        from core.document_processor import load_document_from_upload
        uf = make_txt_file()

        with patch("core.document_processor.tempfile.NamedTemporaryFile",
                   side_effect=tracking_mktemp):
            load_document_from_upload(uf)

        # All temp files created during this test should be deleted
        for path in created_paths:
            assert not os.path.exists(path), f"Temp file not cleaned up: {path}"


# ─────────────────────────────────────────────────────────────────────────────
# Test: chunk_documents
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkDocuments:

    def _make_large_doc(self, chars=5000) -> list[Document]:
        """Creates a single Document with a long text body."""
        text = ("This is a test sentence about the TechWave 2025 conference. " * 100)[:chars]
        return [Document(page_content=text, metadata={"source_file": "test.txt"})]

    def test_short_doc_produces_at_least_one_chunk(self):
        from core.document_processor import chunk_documents
        docs = [Document(page_content="Short text.", metadata={})]
        chunks = chunk_documents(docs)
        assert len(chunks) >= 1

    def test_large_doc_is_split_into_multiple_chunks(self):
        from core.document_processor import chunk_documents
        docs = self._make_large_doc(chars=5000)
        chunks = chunk_documents(docs)
        assert len(chunks) > 1, "A 5000-char doc must split into multiple chunks"

    def test_chunks_respect_configured_chunk_size(self):
        """No chunk should exceed CHUNK_SIZE + CHUNK_OVERLAP characters."""
        import config
        from core.document_processor import chunk_documents
        docs = self._make_large_doc(chars=8000)
        chunks = chunk_documents(docs)
        max_allowed = config.CHUNK_SIZE + config.CHUNK_OVERLAP
        oversized = [c for c in chunks if len(c.page_content) > max_allowed]
        assert oversized == [], (
            f"{len(oversized)} chunks exceed {max_allowed} chars: "
            f"{[len(c.page_content) for c in oversized]}"
        )

    def test_metadata_is_preserved_through_chunking(self):
        """Original document metadata must survive the split."""
        from core.document_processor import chunk_documents
        docs = self._make_large_doc(chars=3000)
        docs[0].metadata["source_file"] = "preserved_file.txt"
        chunks = chunk_documents(docs)
        assert all(
            c.metadata.get("source_file") == "preserved_file.txt"
            for c in chunks
        ), "Metadata lost during chunking"

    def test_empty_document_list_returns_empty(self):
        from core.document_processor import chunk_documents
        assert chunk_documents([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# Test: ingest_documents (mocked ChromaDB)
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestDocuments:
    """
    Tests the full ingest pipeline with ChromaDB mocked out
    so no vector DB writes happen during unit tests.
    """

    @patch("core.document_processor.Chroma")
    @patch("core.document_processor.get_embedding_function")
    def test_successful_ingestion_returns_positive_count(
        self, mock_embed_fn, mock_chroma
    ):
        """A valid .txt upload should return (N_chunks > 0, success_msg)."""
        # Arrange
        mock_embed_fn.return_value = MagicMock()
        mock_store = MagicMock()
        mock_chroma.return_value = mock_store

        from core.document_processor import ingest_documents
        uf = make_txt_file(
            text="09:00 AM | Opening Keynote | Grand Hall | Dr. Jane Smith\n" * 20
        )

        # Act
        total_chunks, msg = ingest_documents([uf])

        # Assert
        assert total_chunks > 0, "Expected at least one chunk from a valid document"
        assert "Successfully indexed" in msg
        mock_store.add_documents.assert_called_once()

    @patch("core.document_processor.Chroma")
    @patch("core.document_processor.get_embedding_function")
    def test_empty_file_list_returns_zero(self, mock_embed_fn, mock_chroma):
        """Passing an empty list must return (0, warning_msg) without crashing."""
        from core.document_processor import ingest_documents
        total, msg = ingest_documents([])
        assert total == 0
        assert "No files" in msg
        mock_chroma.assert_not_called()

    @patch("core.document_processor.Chroma")
    @patch("core.document_processor.get_embedding_function")
    def test_unsupported_file_is_skipped_gracefully(
        self, mock_embed_fn, mock_chroma
    ):
        """An unsupported file type should be skipped; other files still process."""
        mock_embed_fn.return_value = MagicMock()
        mock_store = MagicMock()
        mock_chroma.return_value = mock_store

        from core.document_processor import ingest_documents

        valid_file   = make_txt_file(text="Valid schedule content " * 30)
        invalid_file = FakeUploadedFile("report.docx", b"binary content")

        total_chunks, msg = ingest_documents([valid_file, invalid_file])

        # Valid file should still produce chunks
        assert total_chunks > 0
        # Warning about the failed file must appear in message
        assert "report.docx" in msg or "Skipped" in msg or "unsupported" in msg.lower()

    @patch("core.document_processor.Chroma")
    @patch("core.document_processor.get_embedding_function")
    def test_chroma_write_failure_returns_error_message(
        self, mock_embed_fn, mock_chroma
    ):
        """If ChromaDB raises an exception, the function returns (0, error_msg)."""
        mock_embed_fn.return_value = MagicMock()
        mock_store = MagicMock()
        mock_store.add_documents.side_effect = RuntimeError("Disk full")
        mock_chroma.return_value = mock_store

        from core.document_processor import ingest_documents
        uf = make_txt_file(text="Some schedule content " * 20)

        total, msg = ingest_documents([uf])
        assert total == 0
        assert "error" in msg.lower() or "failed" in msg.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Test: get_document_count
# ─────────────────────────────────────────────────────────────────────────────

class TestGetDocumentCount:

    @patch("core.document_processor.get_or_create_vector_store")
    def test_returns_count_from_chroma(self, mock_get_store):
        mock_store = MagicMock()
        mock_store._collection.count.return_value = 42
        mock_get_store.return_value = mock_store

        from core.document_processor import get_document_count
        assert get_document_count() == 42

    @patch("core.document_processor.get_or_create_vector_store")
    def test_returns_zero_on_exception(self, mock_get_store):
        """If ChromaDB isn't available yet, return 0 (not an exception)."""
        mock_get_store.side_effect = Exception("DB not initialized")

        from core.document_processor import get_document_count
        assert get_document_count() == 0
