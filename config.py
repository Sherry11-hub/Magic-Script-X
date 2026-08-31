"""
config.py — Central Configuration Module
=========================================
Loads all application settings from environment variables (via .env).
All other modules import from here — no raw os.getenv() calls elsewhere.

Usage:
    from config import LLM_MODEL, CHUNK_SIZE  # named imports
    import config; config.OPENAI_API_KEY      # module-level access
"""

import os
from dotenv import load_dotenv

# Load the .env file. `override=False` means existing shell vars take precedence.
# This is called here so every module that imports `config` gets the env vars.
load_dotenv(override=False)


# ── OpenAI Settings ────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ── LLM Configuration ─────────────────────────────────────────────────────────
LLM_MODEL: str       = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS: int      = int(os.getenv("MAX_TOKENS", "1500"))

# ── Embedding Configuration ───────────────────────────────────────────────────
EMBEDDING_PROVIDER: str         = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
HUGGINGFACE_EMBEDDING_MODEL: str = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── ChromaDB Configuration ────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str     = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "event_documents")

# ── Document Chunking ─────────────────────────────────────────────────────────
CHUNK_SIZE: int    = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# ── Retrieval Settings ────────────────────────────────────────────────────────
RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "5"))


def validate() -> list[str]:
    """
    Validates critical config values and returns a list of error messages.
    Called at application startup to give clear feedback to the user.
    
    Returns:
        A list of error strings. Empty list means everything is OK.
    """
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set. Please configure your .env file.")
    elif not OPENAI_API_KEY.startswith("sk-"):
        errors.append("OPENAI_API_KEY appears malformed (should start with 'sk-').")
    
    if EMBEDDING_PROVIDER not in ("huggingface", "openai"):
        errors.append(
            f"Invalid EMBEDDING_PROVIDER='{EMBEDDING_PROVIDER}'. "
            "Must be 'huggingface' or 'openai'."
        )
    
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append(
            f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({CHUNK_SIZE})."
        )
    
    return errors
