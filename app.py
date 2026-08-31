"""
app.py — Generative AI Event & Script Copilot
==============================================
Main Streamlit application entry point.

Run with:
    streamlit run app.py

Architecture:
    app.py              ← You are here (orchestrator + page config)
    ├── ui/sidebar.py   ← Data Ingestion Hub (sidebar)
    ├── ui/script_copilot.py   ← Script Copilot tab
    ├── ui/schedule_adapter.py ← Schedule Adapter tab
    ├── core/document_processor.py  ← Load / Chunk / Embed / Store
    ├── core/rag_engine.py           ← Retrieve / Prompt / Generate
    └── config.py                    ← All settings from .env
"""

# ── Standard library ─────────────────────────────────────────────────────────
import logging

# ── Third-party: Streamlit MUST be imported before other project modules
#    because st.set_page_config() must be the very first Streamlit call ────────
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION — Must be the FIRST Streamlit command in the script.
# Any st.* call before this will raise a StreamlitAPIException.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Event & Script Copilot",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "## 🎤 Generative AI Event & Script Copilot\n"
            "An intelligent RAG-powered assistant for festival and conference organizers.\n\n"
            "**Stack:** LangChain · ChromaDB · OpenAI · Streamlit"
        ),
    },
)

# ── Project modules (imported AFTER set_page_config) ─────────────────────────
import config
from ui.sidebar import render_sidebar
from ui.script_copilot import render_script_copilot
from ui.schedule_adapter import render_schedule_adapter

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

def _apply_styles() -> None:
    """
    Injects global CSS overrides for the app.
    Called once at the top of main().
    """
    st.markdown(
        """
        <style>
            /* ── Global page tweaks ───────────────────────────────────── */
            .main .block-container {
                padding-top: 1.5rem;
                padding-bottom: 3rem;
                max-width: 1100px;
            }

            /* ── Hero banner ──────────────────────────────────────────── */
            .hero-banner {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                padding: 1.75rem 2rem;
                border-radius: 14px;
                margin-bottom: 1.25rem;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }
            .hero-banner h1 {
                color: #e94560;
                font-size: 2rem;
                font-weight: 800;
                margin: 0;
                letter-spacing: 0.5px;
            }
            .hero-banner p {
                color: #a8b2d8;
                font-size: 0.95rem;
                margin: 0.4rem 0 0;
            }

            /* ── Tab bar ──────────────────────────────────────────────── */
            .stTabs [data-baseweb="tab-list"] {
                gap: 6px;
                border-bottom: 2px solid #e0e0e0;
            }
            .stTabs [data-baseweb="tab"] {
                height: 48px;
                padding: 0 1.25rem;
                border-radius: 8px 8px 0 0;
                font-weight: 600;
                font-size: 0.95rem;
            }

            /* ── Chat messages ────────────────────────────────────────── */
            .stChatMessage {
                border-radius: 10px;
                margin-bottom: 0.5rem;
            }

            /* ── Info / warning / error boxes ────────────────────────── */
            .stAlert {
                border-radius: 8px;
            }

            /* ── Footer config bar ───────────────────────────────────── */
            .config-footer {
                color: #888;
                font-size: 0.78rem;
                text-align: center;
                margin-top: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Startup Validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_and_halt_if_broken() -> None:
    """
    Runs config validation at startup.
    If critical errors are found, renders a clear setup guide and stops
    execution so the user never sees a confusing traceback.
    """
    errors = config.validate()

    if not errors:
        return   # All good — continue rendering

    st.error("### ⚙️ Configuration Issues Detected")
    st.markdown(
        "Please fix the following before using the app. "
        "Edit your `.env` file (copy from `.env.example`) and restart Streamlit."
    )

    for err in errors:
        st.warning(f"• {err}")

    with st.expander("🛠️ Quick Setup Guide"):
        st.code(
            "# 1. Copy the template\n"
            "cp .env.example .env\n\n"
            "# 2. Edit .env and add your OpenAI API key\n"
            "OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n\n"
            "# 3. Restart Streamlit\n"
            "streamlit run app.py",
            language="bash",
        )

    st.stop()   # ← Halts further rendering; user must fix config first


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Application entry point.
    Orchestrates: styles → validation → sidebar → tab layout.
    """
    logger.info("App render cycle started.")

    # 1. Inject custom CSS
    _apply_styles()

    # 2. Hero banner
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🎤 Generative AI Event &amp; Script Copilot</h1>
            <p>
                Your intelligent assistant for festival &amp; conference management
                — answers, scripts, and live schedule adaptation powered by RAG.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Validate config — halt if broken (e.g., missing API key)
    _validate_and_halt_if_broken()

    # 4. Sidebar: Data Ingestion Hub (document upload + KB management)
    render_sidebar()

    # 5. Main content: two feature tabs
    tab_script, tab_schedule = st.tabs([
        "🎤  Script Copilot",
        "📅  Schedule Adapter",
    ])

    with tab_script:
        render_script_copilot()

    with tab_schedule:
        render_schedule_adapter()

    # 6. Footer info bar
    st.markdown(
        f"""
        <div class="config-footer">
            ⚡ LangChain · ChromaDB · OpenAI · Streamlit
            &nbsp;|&nbsp; Model: <code>{config.LLM_MODEL}</code>
            &nbsp;|&nbsp; Embeddings: <code>{config.EMBEDDING_PROVIDER}
            ({config.HUGGINGFACE_EMBEDDING_MODEL
              if config.EMBEDDING_PROVIDER == 'huggingface'
              else 'text-embedding-3-small'})</code>
            &nbsp;|&nbsp; Chunk size: <code>{config.CHUNK_SIZE}</code>
            &nbsp;|&nbsp; Retrieval K: <code>{config.RETRIEVAL_K}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
