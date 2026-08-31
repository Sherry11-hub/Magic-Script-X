"""
core/document_processor.py — Document Ingestion Pipeline
=========================================================
Responsibility: Everything related to transforming raw uploaded files
into searchable vector embeddings stored in ChromaDB.

Pipeline:  Upload → Load → Chunk → Embed → Store
"""

import os
import logging
import tempfile
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_chroma import Chroma

import config

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Module-level singleton for the embedding model ───────────────────────────
# Loading sentence-transformers downloads ~90MB on first run.
# We cache it at the module level so it's only loaded once per Python process.
_embedding_model = None


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_embedding_function(force_reload: bool = False):
    """
    Returns the configured embedding model as a singleton.

    On first call it downloads / initializes the model (slow).
    Subsequent calls return the cached instance (fast).

    Args:
        force_reload: If True, discards the cache and re-initializes.

    Returns:
        A LangChain-compatible embedding model object.

    Raises:
        ValueError: If EMBEDDING_PROVIDER is 'openai' but no API key is set.
        ImportError: If the required package is not installed.
    """
    global _embedding_model

    if _embedding_model is not None and not force_reload:
        return _embedding_model

    if config.EMBEDDING_PROVIDER == "openai":
        # ── OpenAI Embeddings ─────────────────────────────────────────────
        if not config.OPENAI_API_KEY:
            raise ValueError(
                "EMBEDDING_PROVIDER is set to 'openai' but OPENAI_API_KEY is missing."
            )
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise ImportError("Run: pip install langchain-openai") from exc

        logger.info("Initializing OpenAI text-embedding-3-small model.")
        _embedding_model = OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY,
            model="text-embedding-3-small",
        )

    else:
        # ── HuggingFace Sentence-Transformers (default) ───────────────────
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Run: pip install langchain-huggingface sentence-transformers"
            ) from exc

        logger.info(
            "Initializing HuggingFace embedding model: %s",
            config.HUGGINGFACE_EMBEDDING_MODEL,
        )
        _embedding_model = HuggingFaceEmbeddings(
            model_name=config.HUGGINGFACE_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            # cache_folder keeps the model on disk between runs
            cache_folder="./.model_cache",
        )

    return _embedding_model


# ─────────────────────────────────────────────────────────────────────────────
# Document Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_document_from_upload(uploaded_file) -> List[Document]:
    """
    Converts a Streamlit UploadedFile into a list of LangChain Documents.

    Strategy:
        - Writes bytes to a NamedTemporaryFile (LangChain loaders need file paths)
        - Dispatches to the correct loader based on file extension
        - Injects the original filename into each document's metadata
        - Guarantees temp file cleanup via try/finally

    Args:
        uploaded_file: A ``streamlit.runtime.uploaded_file_manager.UploadedFile``
                       object (the result of ``st.file_uploader``).

    Returns:
        List[Document]: One Document per logical page/row/section in the file.

    Raises:
        ValueError: For unsupported file types.
        UnicodeDecodeError: If a .txt file isn't UTF-8 encoded.
    """
    file_name = uploaded_file.name
    ext = os.path.splitext(file_name)[1].lower()

    logger.info("Loading: '%s' (type: %s, size: %d bytes)",
                file_name, ext, len(uploaded_file.getvalue()))

    # Write to a temporary file — required because loaders need a real path
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=ext,
        mode="wb",
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # ── Route to appropriate loader ───────────────────────────────────
        if ext == ".pdf":
            # PyPDFLoader splits into one Document per page
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

        elif ext == ".txt":
            # TextLoader returns the whole file as a single Document
            loader = TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()

        elif ext == ".csv":
            # CSVLoader returns one Document per row, using column headers as keys
            loader = CSVLoader(
                file_path=tmp_path,
                csv_args={"delimiter": ","},
            )
            docs = loader.load()

        else:
            raise ValueError(
                f"Unsupported file type: '{ext}'. "
                "Please upload .txt, .pdf, or .csv files only."
            )

        # ── Enrich metadata ───────────────────────────────────────────────
        # Inject the user-visible filename so we can show sources later.
        for doc in docs:
            doc.metadata["source_file"] = file_name
            doc.metadata["file_type"]   = ext.lstrip(".")

        logger.info("Loaded %d page(s)/section(s) from '%s'.", len(docs), file_name)
        return docs

    finally:
        # Always delete the temp file, even if loading raised an exception
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Text Splitter / Chunker
# ─────────────────────────────────────────────────────────────────────────────

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits documents into overlapping chunks using RecursiveCharacterTextSplitter.

    Why recursive?
        It tries to split at semantically meaningful boundaries first (paragraph
        breaks, then sentences, then words) before falling back to hard character
        cuts. This preserves context better than a simple fixed-size split.

    Why overlap?
        If an answer spans a chunk boundary, the overlap ensures both chunks
        contain enough context for the retriever to score them highly.

    Args:
        documents: Raw LangChain Documents as loaded from files.

    Returns:
        List[Document]: Smaller, overlapping chunks with original metadata intact.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        # Tried in order — prefers bigger semantic units first
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        length_function=len,
        # Adds a "chunk_index" field to metadata for ordering
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    logger.info(
        "Chunked %d document(s) into %d chunks "
        "(chunk_size=%d, overlap=%d).",
        len(documents), len(chunks),
        config.CHUNK_SIZE, config.CHUNK_OVERLAP,
    )
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_vector_store(
    embedding_function=None,
) -> Chroma:
    """
    Opens (or creates) the persistent ChromaDB vector store.

    ChromaDB will create the directory at ``CHROMA_PERSIST_DIR`` on first call
    and reuse it on subsequent calls — data survives app restarts.

    Args:
        embedding_function: Override the default embedding model. Useful for tests.

    Returns:
        Chroma: A LangChain-wrapped ChromaDB collection.
    """
    if embedding_function is None:
        embedding_function = get_embedding_function()

    # Ensure the persist directory exists
    os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)

    store = Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )

    return store


# ─────────────────────────────────────────────────────────────────────────────
# Main Ingestion Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def ingest_documents(uploaded_files: list) -> Tuple[int, str]:
    """
    Full ingestion pipeline: Load ➜ Chunk ➜ Embed ➜ Store.

    This is the function called directly by the Streamlit UI when the user
    clicks "Process & Index Documents".

    Design decisions:
        - Uses ``add_documents()`` on the existing store rather than
          ``from_documents()``, so repeated uploads ADD to the KB (not reset it).
        - Continues processing remaining files even if one file fails.
        - Reports exactly which files failed and why.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.

    Returns:
        Tuple[int, str]:
            - int: Total number of chunks successfully added.
            - str: Human-readable status message for display in the UI.
    """
    if not uploaded_files:
        return 0, "⚠️ No files provided for ingestion."

    all_chunks: List[Document] = []
    failed_files: List[str]   = []
    success_files: List[str]  = []

    # ── Step 1 & 2: Load and Chunk each file ─────────────────────────────
    for uf in uploaded_files:
        try:
            raw_docs = load_document_from_upload(uf)
            chunks   = chunk_documents(raw_docs)
            all_chunks.extend(chunks)
            success_files.append(uf.name)

        except ValueError as e:
            # Unsupported file type or malformed content
            logger.warning("Skipping '%s': %s", uf.name, e)
            failed_files.append(f"{uf.name} (unsupported type)")

        except UnicodeDecodeError:
            logger.warning("Skipping '%s': not UTF-8 encoded.", uf.name)
            failed_files.append(f"{uf.name} (encoding error — save as UTF-8)")

        except Exception as e:
            logger.error("Unexpected error on '%s': %s", uf.name, e, exc_info=True)
            failed_files.append(f"{uf.name} ({type(e).__name__})")

    if not all_chunks:
        detail = "; ".join(failed_files) if failed_files else "No content extracted."
        return 0, f"❌ Ingestion failed. No valid content was extracted.\n\nDetails: {detail}"

    # ── Step 3: Embed and store in ChromaDB ──────────────────────────────
    try:
        embedding_fn  = get_embedding_function()
        vector_store  = get_or_create_vector_store(embedding_fn)

        # add_documents() appends to the existing collection
        vector_store.add_documents(all_chunks)

        # Build human-readable status
        lines = [
            f"✅ Successfully indexed **{len(all_chunks)} chunks** from "
            f"**{len(success_files)} file(s)**:",
        ]
        for fname in success_files:
            lines.append(f"  • {fname}")

        if failed_files:
            lines.append("\n⚠️ **Skipped files** (see details above):")
            for fname in failed_files:
                lines.append(f"  • {fname}")

        status_msg = "\n".join(lines)
        logger.info("Ingestion complete. %d chunks added.", len(all_chunks))
        return len(all_chunks), status_msg

    except Exception as e:
        logger.error("Vector store write failed: %s", e, exc_info=True)
        return 0, f"❌ Vector store error: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# Utility Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_document_count() -> int:
    """
    Returns the number of chunks currently in the vector store.
    Returns 0 if the store doesn't exist yet (graceful for fresh installs).
    """
    try:
        store = get_or_create_vector_store()
        # _collection is a chromadb.Collection; .count() is its public API
        return store._collection.count()
    except Exception:
        return 0


def clear_vector_store() -> bool:
    """
    Deletes all documents from the ChromaDB collection.
    Useful for resetting the knowledge base without restarting the app.

    Returns:
        True if successful, False otherwise.
    """
    try:
        store = get_or_create_vector_store()
        store._client.delete_collection(config.CHROMA_COLLECTION_NAME)
        logger.info("Vector store collection '%s' cleared.", config.CHROMA_COLLECTION_NAME)
        return True
    except Exception as e:
        logger.error("Failed to clear vector store: %s", e)
        return False
