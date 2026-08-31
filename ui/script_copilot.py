"""
ui/script_copilot.py — Script Copilot Tab
==========================================
A full chat interface for drafting anchoring scripts, speaker introductions,
session transitions, and other live-event content — grounded in uploaded docs.

Features:
  - Persistent conversation history (Streamlit session_state)
  - Quick-start template selector
  - Token-by-token streaming via st.write_stream()
  - Source attribution expander under each AI response
  - Downloadable script export
  - Clear chat button
"""

import streamlit as st

from core.rag_engine import run_script_copilot
from core.document_processor import get_document_count

# ─────────────────────────────────────────────────────────────────────────────
# Template Library
# ─────────────────────────────────────────────────────────────────────────────
# Each entry is a (label, prompt_text) tuple.
# Prompts are intentionally detailed so the LLM has clear instructions.

TEMPLATES = {
    "── Choose a template ──": "",

    "🎬  Opening Welcome Script": (
        "Write a warm, high-energy 2-minute opening welcome script for the emcee to deliver "
        "at the start of the event. Reference the official event name, venue, and overarching "
        "theme. Include [stage directions] in brackets. Tone: professional yet celebratory."
    ),

    "🎤  Keynote Speaker Introduction": (
        "Write a professional 60-second introduction script for the keynote speaker. "
        "Use their exact full name, current title, and organization as found in the documents. "
        "Mention their session topic and one or two key achievements. End with a strong "
        "call-to-action for the audience to welcome them."
    ),

    "🔄  Session Transition Script": (
        "Write a smooth 45-second transition script for moving from the morning sessions "
        "to the networking lunch break. Recap the key theme of the morning, thank the "
        "speakers by name (from the documents), and direct attendees to the lunch location "
        "with a clear time to reconvene."
    ),

    "🏆  Award Presentation Script": (
        "Draft an exciting 90-second script to present the Best Speaker Award. "
        "Build suspense before the reveal. Include the award category name, what criteria "
        "it recognises, and a placeholder for the winner's name formatted as [WINNER NAME]. "
        "Make it celebratory and memorable."
    ),

    "🤝  Sponsor Recognition Script": (
        "Write a 60-second script that formally recognizes and thanks the event sponsors. "
        "Use the sponsor names and tiers (e.g., Gold, Silver) as listed in the uploaded "
        "documents. Acknowledge their specific contributions to the event."
    ),

    "🎊  Closing Ceremony Script": (
        "Write a heartfelt 3-minute closing ceremony script. Recap the highlights of the "
        "event (use specific session names and speakers from the documents), thank the "
        "organizing team, sponsors, and attendees, and close with a memorable sign-off that "
        "teases the next edition of the event."
    ),

    "📢  Housekeeping Announcement": (
        "Write a friendly 30-second housekeeping announcement covering: Wi-Fi credentials, "
        "emergency exits, washroom locations, and the hashtag/social media handle for the "
        "event (use details from the uploaded documents where available)."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Session State Keys
# ─────────────────────────────────────────────────────────────────────────────

_CHAT_HISTORY_KEY = "script_copilot_history"
# Each entry: {"role": "user"|"assistant", "content": str, "sources": list[str]}


def _init_session_state() -> None:
    """Initialises session state keys on first load."""
    if _CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[_CHAT_HISTORY_KEY] = []


# ─────────────────────────────────────────────────────────────────────────────
# Main Render Function
# ─────────────────────────────────────────────────────────────────────────────

def render_script_copilot() -> None:
    """
    Renders the full Script Copilot tab.
    Entry point called from app.py.
    """
    _init_session_state()

    st.header("🎤 Script Copilot")
    st.caption(
        "Draft professional anchoring scripts, speaker introductions, and event content — "
        "all grounded in your uploaded documents. The AI will **only** use information "
        "from your event files."
    )

    # ── Guard: empty knowledge base ───────────────────────────────────────
    if get_document_count() == 0:
        _render_empty_state()
        return

    # ── Template Selector ─────────────────────────────────────────────────
    selected_label = st.selectbox(
        "⚡ Quick-start template (optional):",
        options=list(TEMPLATES.keys()),
        key="script_template_selector",
        help="Select a template to pre-fill the chat input, then customise as needed.",
    )

    # ── Chat History ──────────────────────────────────────────────────────
    _render_chat_history()

    # ── Chat Input ────────────────────────────────────────────────────────
    # Determine the default/pre-filled prompt text
    prefill = TEMPLATES.get(selected_label, "")

    user_input = st.chat_input(
        placeholder="e.g., 'Write a 60-second introduction for the afternoon panel host…'",
        key="script_chat_input",
    )

    # "Use Template" button fires the pre-filled template as the user message
    if prefill:
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("▶ Use Template", key="use_template_btn", type="secondary"):
                user_input = prefill   # override chat_input with template text

    # ── Handle User Input ─────────────────────────────────────────────────
    if user_input:
        _handle_user_message(user_input)

    # ── Footer Controls ───────────────────────────────────────────────────
    _render_footer_controls()


# ─────────────────────────────────────────────────────────────────────────────
# Private Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_empty_state() -> None:
    """Shown when no documents have been indexed yet."""
    st.info(
        "**No documents indexed yet.**\n\n"
        "Use the **Data Ingestion Hub** in the sidebar to upload your event documents "
        "(schedules, speaker bios, venue details, themes).\n\n"
        "Once uploaded, return here to start generating scripts.",
        icon="📂",
    )

    # Show a visual preview of what the feature can do
    with st.expander("👀 What can I do here?", expanded=True):
        st.markdown("""
        Once you've uploaded documents, you can ask the Script Copilot to:

        | Request | Example |
        |---------|---------|
        | Opening welcome | *"Write the MC's opening welcome for Day 1"* |
        | Speaker intro | *"Introduce Dr. Chen for her 2 PM keynote"* |
        | Session transition | *"Bridge the morning sessions to lunch"* |
        | Award ceremony | *"Script the Best Innovation Award reveal"* |
        | Closing remarks | *"Write the Day 2 closing ceremony script"* |
        """)


def _render_chat_history() -> None:
    """Renders all messages in the chat history."""
    history = st.session_state[_CHAT_HISTORY_KEY]

    for msg in history:
        avatar = "🎭" if msg["role"] == "user" else "🤖"

        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            # Sources attribution (only on assistant messages that have sources)
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Documents referenced", expanded=False):
                    for src in msg["sources"]:
                        st.caption(f"• {src}")


def _handle_user_message(user_input: str) -> None:
    """
    Processes a new user message:
        1. Appends to history & displays immediately.
        2. Calls the RAG pipeline (streaming).
        3. Displays the streamed response.
        4. Appends the completed response + sources to history.
    """
    # ── Show user message immediately ─────────────────────────────────────
    st.session_state[_CHAT_HISTORY_KEY].append({
        "role":    "user",
        "content": user_input,
        "sources": [],
    })
    with st.chat_message("user", avatar="🎭"):
        st.markdown(user_input)

    # ── Stream AI response ────────────────────────────────────────────────
    with st.chat_message("assistant", avatar="🤖"):
        try:
            stream, sources = run_script_copilot(user_input)

            # st.write_stream() consumes the generator token-by-token
            # and returns the completed string once done
            full_response = st.write_stream(stream)

            # Sources attribution
            if sources:
                with st.expander("📎 Documents referenced", expanded=False):
                    for src in sources:
                        st.caption(f"• {src}")

            # Save completed response to history
            st.session_state[_CHAT_HISTORY_KEY].append({
                "role":    "assistant",
                "content": full_response,
                "sources": sources,
            })

            # ── Download button for generated script ──────────────────────
            st.download_button(
                label="⬇️  Save Script as .txt",
                data=f"REQUEST:\n{user_input}\n\nGENERATED SCRIPT:\n{full_response}",
                file_name="event_script.txt",
                mime="text/plain",
                key=f"dl_script_{len(st.session_state[_CHAT_HISTORY_KEY])}",
            )

        except RuntimeError as exc:
            # KB empty mid-session (documents were cleared while chatting)
            error_msg = f"⚠️ **Knowledge Base Error:** {exc}"
            st.error(error_msg)
            st.session_state[_CHAT_HISTORY_KEY].append(
                {"role": "assistant", "content": error_msg, "sources": []}
            )

        except ValueError as exc:
            # Missing API key
            error_msg = f"🔑 **Configuration Error:** {exc}"
            st.error(error_msg)
            st.session_state[_CHAT_HISTORY_KEY].append(
                {"role": "assistant", "content": error_msg, "sources": []}
            )

        except Exception as exc:
            error_msg = f"❌ **Unexpected Error:** {type(exc).__name__}: {exc}"
            st.error(error_msg)
            st.session_state[_CHAT_HISTORY_KEY].append(
                {"role": "assistant", "content": error_msg, "sources": []}
            )


def _render_footer_controls() -> None:
    """Renders the 'Clear Chat' button at the bottom of the tab."""
    history = st.session_state.get(_CHAT_HISTORY_KEY, [])

    if history:
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col3:
            if st.button("🗑️ Clear Chat", key="clear_script_chat", type="secondary"):
                st.session_state[_CHAT_HISTORY_KEY] = []
                st.rerun()
