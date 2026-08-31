"""
ui/sidebar.py — Data Ingestion Hub
====================================
Renders the left sidebar containing:
  - Knowledge Base status badge (live document count)
  - Multi-file uploader (.txt, .pdf, .csv)
  - "Process & Index" action button with progress feedback
  - "Clear Knowledge Base" danger action
  - Tips accordion
"""

import streamlit as st
from core.document_processor import ingest_documents, get_document_count, clear_vector_store


def render_sidebar() -> None:
    """
    Renders the full Data Ingestion Hub sidebar.
    Called once from app.py before the tab layout.
    """
    with st.sidebar:

        # ── Branding ──────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #0f3460);
                padding: 1rem 1.25rem;
                border-radius: 10px;
                margin-bottom: 0.5rem;
            ">
                <h2 style="color:#e94560; margin:0; font-size:1.2rem;">
                    🎤 Event Copilot
                </h2>
                <p style="color:#a8b2d8; margin:0.25rem 0 0; font-size:0.75rem;">
                    Powered by RAG + LangChain
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("## 📂 Data Ingestion Hub")
        st.caption("Feed the AI your event documents to unlock all features.")
        st.divider()

        # ── Knowledge Base Status Badge ───────────────────────────────────
        _render_kb_status()
        st.divider()

        # ── File Uploader ─────────────────────────────────────────────────
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            label="Drag & drop or browse files",
            type=["txt", "pdf", "csv"],
            accept_multiple_files=True,
            help=(
                "Accepted formats: .txt · .pdf · .csv\n\n"
                "Suggested content: schedules, speaker bios, venue details, themes."
            ),
            key="sidebar_file_uploader",
            label_visibility="collapsed",
        )

        # Show a preview of what was selected
        if uploaded_files:
            st.caption(f"**{len(uploaded_files)} file(s) selected:**")
            for uf in uploaded_files:
                size_kb = len(uf.getvalue()) / 1024
                icon    = {"pdf": "📄", "txt": "📝", "csv": "📊"}.get(
                    uf.name.rsplit(".", 1)[-1].lower(), "📎"
                )
                st.caption(f"  {icon} {uf.name} ({size_kb:.1f} KB)")

        # ── Ingestion Action Button ───────────────────────────────────────
        st.markdown("")   # visual spacer

        process_disabled = not bool(uploaded_files)
        if st.button(
            "🚀  Process & Index Documents",
            type="primary",
            use_container_width=True,
            disabled=process_disabled,
            help="Chunk, embed, and store the selected files in the vector database.",
            key="process_docs_btn",
        ):
            _run_ingestion(uploaded_files)

        st.divider()

        # ── Clear KB Danger Zone ──────────────────────────────────────────
        _render_clear_kb_section()

        st.divider()

        # ── Tips Accordion ────────────────────────────────────────────────
        _render_tips()


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_kb_status() -> None:
    """Displays a live knowledge-base size badge."""
    count = get_document_count()

    if count > 0:
        st.success(f"✅ Knowledge Base: **{count:,} chunks** indexed")
    else:
        st.warning(
            "⚠️ Knowledge base is **empty**.\n\n"
            "Upload event documents above to get started."
        )


def _run_ingestion(uploaded_files: list) -> None:
    """
    Handles the full ingestion flow with live progress feedback.
    Triggered when the user clicks "Process & Index Documents".
    """
    progress_bar = st.progress(0, text="Starting ingestion...")
    status_box   = st.empty()

    # Phase 1 – Loading
    progress_bar.progress(20, text="📖 Loading documents...")

    # Phase 2 – Chunking + Embedding + Storing (the heavy work)
    progress_bar.progress(40, text="✂️  Chunking & embedding...")

    total_chunks, status_msg = ingest_documents(uploaded_files)

    if total_chunks > 0:
        progress_bar.progress(100, text="✅ Done!")
        status_box.success(status_msg)
        st.balloons()
        # Force a re-render so the KB status badge updates
        st.rerun()
    else:
        progress_bar.progress(100, text="❌ Failed")
        status_box.error(status_msg)


def _render_clear_kb_section() -> None:
    """Renders the (collapsible) danger-zone for clearing the KB."""
    with st.expander("🗑️ Manage Knowledge Base", expanded=False):
        st.warning(
            "**Danger Zone**  \n"
            "Clearing the knowledge base removes all indexed documents. "
            "You will need to re-upload and re-process your files."
        )

        # Two-step confirmation — prevents accidental deletion
        confirm = st.checkbox(
            "I understand this is irreversible",
            key="confirm_clear_kb",
        )
        if st.button(
            "Clear All Documents",
            type="secondary",
            disabled=not confirm,
            use_container_width=True,
        ):
            if clear_vector_store():
                st.success("Knowledge base cleared.")
                st.rerun()
            else:
                st.error("Failed to clear the knowledge base. Check logs.")


def _render_tips() -> None:
    """Renders a tips accordion with guidance on document formatting."""
    with st.expander("💡 Tips for Best Results", expanded=False):
        st.markdown("""
        **Schedules (.txt / .csv)**
        - Include: date, time, session title, location, speaker name
        - Example row: `09:00 AM | Opening Keynote | Grand Hall | Dr. Jane Smith`

        **Speaker Bios (.txt / .pdf)**
        - Include: full name, title, organization, session topic
        - More detail = better introductions

        **Venue Details (.txt)**
        - Hall/room names, capacity, AV equipment, accessibility info

        **Themes & Runbook (.pdf / .txt)**
        - Event name, tagline, brand voice guidelines, key messages

        ---
        *More documents = richer context = better AI responses.*
        """)
