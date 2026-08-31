"""
core/rag_engine.py — RAG Pipeline & LLM Orchestration
=======================================================
Responsibility: Everything related to retrieval, prompt construction,
LLM invocation, and streaming output.

Two specialized pipelines are exposed:
    1. run_script_copilot()   — Creative script drafting (temp=0.5)
    2. run_schedule_adapter() — Precise schedule recalculation (temp=0.1)

Both follow the LCEL (LangChain Expression Language) composition pattern:
    Retrieve Docs → Format Context → Build Prompt → LLM → Parse Output
"""

import logging
from typing import Iterator, List, Tuple, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

import config
from core.document_processor import get_embedding_function, get_or_create_vector_store

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: Both templates contain the anti-hallucination instruction.
# The model is explicitly forbidden from using any knowledge outside the context.

SCRIPT_COPILOT_SYSTEM = """\
You are an expert Event Script Copilot for large-scale festivals and conferences.
Your role is to help event organizers craft professional anchoring scripts,
speaker introductions, session transitions, and other event content.

══ CRITICAL INSTRUCTIONS ══════════════════════════════════════════════════════
1. Answer the user's request using ONLY the information provided in the CONTEXT
   section below. Do NOT invent speaker names, titles, venue names, times, or
   any other factual detail.
2. If the required information is NOT present in the context, respond with:
   "I don't have enough information in the uploaded documents to complete this
   request accurately. Please upload the relevant event documents first."
3. Use professional, warm, and energetic language appropriate for live events.
4. When writing scripts, include stage directions in [brackets] where helpful.
═══════════════════════════════════════════════════════════════════════════════

CONTEXT (retrieved from your uploaded event documents):
────────────────────────────────────────────────────────
{context}
────────────────────────────────────────────────────────
"""

SCHEDULE_ADAPTER_SYSTEM = """\
You are an expert Event Schedule Adapter for large-scale conferences and festivals.
Your role is to analyze a reported disruption, locate the relevant schedule in the
context, recalculate all affected timings, and produce a clean revised itinerary.

══ CRITICAL INSTRUCTIONS ══════════════════════════════════════════════════════
1. Base your schedule EXCLUSIVELY on the information in the CONTEXT below.
   Do NOT invent sessions, speakers, or times not found in the context.
2. When shifting time slots, cascade the delay/change to ALL subsequent events
   on the same track/room unless the context indicates a buffer period.
3. If the original schedule is not found in the context, respond with:
   "I could not find schedule information in the uploaded documents.
   Please upload an event schedule document first."
4. Format the revised schedule as a numbered list:
      [##] [HH:MM AM/PM] – [Session Title] | [Location] | [Speaker/Host]
   Add a ⚠️  prefix to any entry that was modified from the original.
5. After the schedule, add a brief "Changes Made" summary section.
═══════════════════════════════════════════════════════════════════════════════

CONTEXT (retrieved from your uploaded schedule documents):
────────────────────────────────────────────────────────
{context}
────────────────────────────────────────────────────────
"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm(temperature: float) -> ChatOpenAI:
    """
    Instantiates a ChatOpenAI client with the given temperature.

    Args:
        temperature: Creativity level (0.0 = deterministic, 1.0 = creative).

    Returns:
        ChatOpenAI instance with streaming enabled.

    Raises:
        ValueError: If OPENAI_API_KEY is not configured.
    """
    if not config.OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API key is missing.\n"
            "Add OPENAI_API_KEY=sk-... to your .env file and restart."
        )

    return ChatOpenAI(
        openai_api_key=config.OPENAI_API_KEY,
        model_name=config.LLM_MODEL,
        temperature=temperature,
        max_tokens=config.MAX_TOKENS,
        streaming=True,   # Enables token-by-token streaming for Streamlit
    )


# ─────────────────────────────────────────────────────────────────────────────
# Retriever Factory
# ─────────────────────────────────────────────────────────────────────────────

def _get_retriever(k: int):
    """
    Creates a retriever from the ChromaDB collection.

    Uses MMR (Maximal Marginal Relevance) search, which balances:
        - Relevance: how similar the chunk is to the query
        - Diversity: how different the chunk is from already-selected chunks
    This prevents returning K nearly-identical chunks (e.g. the same table
    repeated on consecutive pages).

    Args:
        k: Number of chunks to return.

    Returns:
        A LangChain BaseRetriever.

    Raises:
        RuntimeError: If the vector store is empty (no documents indexed yet).
    """
    embedding_fn  = get_embedding_function()
    vector_store  = get_or_create_vector_store(embedding_fn)

    doc_count = vector_store._collection.count()
    if doc_count == 0:
        raise RuntimeError(
            "The knowledge base is empty.\n"
            "Please upload your event documents using the sidebar first."
        )

    # Clamp k to the actual document count to avoid ChromaDB errors
    effective_k = min(k, doc_count)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":       effective_k,
            "fetch_k": effective_k * 3,  # Candidate pool for MMR diversity
            "lambda_mult": 0.6,          # 0=max diversity, 1=max relevance
        },
    )
    return retriever


# ─────────────────────────────────────────────────────────────────────────────
# Context Formatter
# ─────────────────────────────────────────────────────────────────────────────

def _format_docs(docs: List[Document]) -> str:
    """
    Converts retrieved Document objects into a single formatted context string.

    Includes source metadata so the LLM (and user) know where each piece of
    information came from.

    Args:
        docs: List of LangChain Document objects from the retriever.

    Returns:
        A multi-section string with source headers per chunk.
    """
    if not docs:
        return "No relevant context was found in the uploaded documents."

    parts = []
    for i, doc in enumerate(docs, start=1):
        source   = doc.metadata.get("source_file", "Unknown")
        page     = doc.metadata.get("page", None)
        page_str = f" · Page {int(page) + 1}" if page is not None else ""

        parts.append(
            f"[Chunk {i} | Source: {source}{page_str}]\n"
            f"{doc.page_content.strip()}"
        )

    return "\n\n".join(parts)


def _extract_sources(docs: List[Document]) -> List[str]:
    """
    Extracts unique source file names from a list of Documents.

    Returns:
        Sorted list of unique source file names.
    """
    return sorted(set(
        doc.metadata.get("source_file", "Unknown Source")
        for doc in docs
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Public Pipeline Functions
# ─────────────────────────────────────────────────────────────────────────────

def run_script_copilot(
    query: str,
    k: Optional[int] = None,
) -> Tuple[Iterator[str], List[str]]:
    """
    Runs the Script Copilot RAG pipeline and returns a streaming response.

    Temperature 0.5 — balanced between creative and grounded. Slightly higher
    than default RAG to allow natural, flowing script language.

    Args:
        query: The user's request (e.g. "Write an intro for the keynote speaker").
        k:     Number of context chunks to retrieve. Defaults to config.RETRIEVAL_K.

    Returns:
        Tuple:
            - Iterator[str]: A streaming generator of response tokens.
              Pass directly to ``st.write_stream()``.
            - List[str]:     Source file names that were retrieved.

    Raises:
        RuntimeError: Propagated from _get_retriever() if KB is empty.
        ValueError:   Propagated from _get_llm() if API key is missing.
    """
    effective_k = k or config.RETRIEVAL_K

    # ── Retrieve relevant chunks ──────────────────────────────────────────
    retriever = _get_retriever(effective_k)
    docs      = retriever.invoke(query)
    context   = _format_docs(docs)
    sources   = _extract_sources(docs)

    logger.info(
        "Script Copilot: retrieved %d chunks for query='%s...' | sources: %s",
        len(docs), query[:60], sources,
    )

    # ── Build prompt & chain ──────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SCRIPT_COPILOT_SYSTEM),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])

    llm   = _get_llm(temperature=0.5)
    chain = prompt | llm | StrOutputParser()

    # ── Stream the response ───────────────────────────────────────────────
    # We pass context and question as separate variables to the prompt.
    # The chain.stream() returns a generator — the caller uses st.write_stream().
    stream = chain.stream({
        "context":  context,
        "question": query,
    })

    return stream, sources


def run_schedule_adapter(
    disruption: str,
    k: Optional[int] = None,
) -> Tuple[Iterator[str], List[str]]:
    """
    Runs the Schedule Adapter RAG pipeline and returns a streaming response.

    Uses:
        - Higher k (default 8) to capture the full schedule across chunks.
        - Temperature 0.1 — near-deterministic for accurate time arithmetic.

    Args:
        disruption: Description of what changed (e.g. "Keynote delayed 30 min").
        k:          Number of context chunks. Defaults to 2× config.RETRIEVAL_K.

    Returns:
        Tuple:
            - Iterator[str]: Streaming generator for st.write_stream().
            - List[str]:     Source files referenced.

    Raises:
        RuntimeError: Propagated from _get_retriever() if KB is empty.
        ValueError:   Propagated from _get_llm() if API key is missing.
    """
    # Use a larger k to capture the entire schedule (may span many chunks)
    effective_k = k or max(config.RETRIEVAL_K * 2, 8)

    # ── Retrieve schedule chunks ──────────────────────────────────────────
    retriever = _get_retriever(effective_k)
    docs      = retriever.invoke(disruption)
    context   = _format_docs(docs)
    sources   = _extract_sources(docs)

    logger.info(
        "Schedule Adapter: retrieved %d chunks for disruption='%s...' | sources: %s",
        len(docs), disruption[:60], sources,
    )

    # ── Build prompt & chain ──────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SCHEDULE_ADAPTER_SYSTEM),
        HumanMessagePromptTemplate.from_template(
            "Disruption Report:\n{question}\n\n"
            "Please produce the complete revised schedule."
        ),
    ])

    llm   = _get_llm(temperature=0.1)
    chain = prompt | llm | StrOutputParser()

    # ── Stream the response ───────────────────────────────────────────────
    stream = chain.stream({
        "context":  context,
        "question": disruption,
    })

    return stream, sources
