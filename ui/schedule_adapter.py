"""
ui/schedule_adapter.py — Schedule Adapter Tab
==============================================
Allows event organizers to report a schedule disruption in plain English.
The AI retrieves the relevant schedule from the knowledge base and outputs
a fully revised, chronological itinerary with a "Changes Made" summary.

Features:
  - Example disruption selector (pre-fills the input)
  - Streaming revised schedule output
  - Source attribution
  - Downloadable .txt output
  - Session-persistent adaptation history log
"""

import streamlit as st

from core.rag_engine import run_schedule_adapter
from core.document_processor import get_document_count

# ─────────────────────────────────────────────────────────────────────────────
# Example Disruptions
# ─────────────────────────────────────────────────────────────────────────────

DISRUPTION_EXAMPLES = [
    "── Select a sample disruption ──",
    "The 2:00 PM keynote is delayed by 30 minutes due to the speaker's flight delay. "
    "Please shift all subsequent sessions accordingly.",
    "The 10:30 AM 'Design Thinking' workshop has been cancelled entirely. "
    "Please remove it and close the gap in the schedule.",
    "The afternoon panel discussion at 3:00 PM needs to move from Hall A to Hall B. "
    "Timing stays the same.",
    "The Opening Ceremony must start 15 minutes early at 9:45 AM instead of 10:00 AM. "
    "Adjust all subsequent timings.",
    "The networking lunch break needs to be extended from 45 minutes to 75 minutes. "
    "Shift all afternoon sessions by 30 minutes.",
    "The 4:30 PM closing session has been extended by 20 minutes to accommodate a live Q&A. "
    "Update the schedule end time.",
]

# Session state key for adaptation history
_HISTORY_KEY = "schedule_adapter_history"
# Each entry: {"disruption": str, "schedule": str, "sources": list[str]}


def _init_session_state() -> None:
    if _HISTORY_KEY not in st.session_state:
        st.session_state[_HISTORY_KEY] = []


# ─────────────────────────────────────────────────────────────────────────────
# Main Render Function
# ─────────────────────────────────────────────────────────────────────────────

def render_schedule_adapter() -> None:
    """
    Renders the full Schedule Adapter tab.
    Entry point called from app.py.
    """
    _init_session_state()

    st.header("📅 Schedule Adapter")
    st.caption(
        "Report a real-time disruption to your event schedule. "
        "The AI will retrieve the affected sessions from your documents "
        "and generate a revised, cascaded itinerary — instantly."
    )

    # ── Guard: empty knowledge base ───────────────────────────────────────
    if get_document_count() == 0:
        _render_empty_state()
        return

    # ── Disruption Input Form ─────────────────────────────────────────────
    revised_schedule, sources = _render_input_form()

    # ── Adaptation History ────────────────────────────────────────────────
    _render_history()


# ─────────────────────────────────────────────────────────────────────────────
# Private Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_empty_state() -> None:
    """Shown when no documents have been indexed yet."""
    st.info(
        "**No schedule documents indexed yet.**\n\n"
        "Upload your event schedule (as .txt, .pdf, or .csv) using the "
        "**Data Ingestion Hub** in the sidebar.\n\n"
        "Example schedule format:\n"
        "```\n"
        "09:00 AM | Registration & Welcome Coffee | Foyer\n"
        "10:00 AM | Opening Keynote               | Grand Hall | Dr. Jane Smith\n"
        "11:00 AM | Panel: Future of AI            | Hall B     | Moderator: Alex Ray\n"
        "12:30 PM | Networking Lunch               | Terrace\n"
        "```",
        icon="📅",
    )


def _render_input_form():
    """
    Renders the disruption input form and handles submission.

    Returns:
        Tuple of (revised_schedule_str, sources_list) if a disruption was
        processed this render cycle; otherwise (None, None).
    """
    # ── Example selector ─────────────────────────────────────────────────
    st.subheader("Report a Disruption")

    col_ex, col_hint = st.columns([3, 2])
    with col_ex:
        selected_example = st.selectbox(
            "Load a sample:",
            options=DISRUPTION_EXAMPLES,
            key="disruption_example",
            label_visibility="collapsed",
        )
    with col_hint:
        st.caption("← Select a sample disruption or type your own below.")

    # Pre-fill text area when an example is chosen
    prefill = (
        ""
        if selected_example.startswith("──")
        else selected_example
    )

    # ── Text area ─────────────────────────────────────────────────────────
    disruption_text = st.text_area(
        label="Describe the disruption in plain English:",
        value=prefill,
        height=130,
        max_chars=1000,
        placeholder=(
            "e.g., 'The 2:00 PM keynote has been delayed by 30 minutes. "
            "The speaker's flight landed late. Please adjust the entire "
            "afternoon schedule accordingly.'"
        ),
        key="disruption_textarea",
        help=(
            "Be specific: include the time, session name, speaker, and the "
            "nature of the change (delay / cancellation / room change / extension)."
        ),
    )

    char_count = len(disruption_text)
    st.caption(f"Characters: {char_count} / 1000")

    # ── Submit button ─────────────────────────────────────────────────────
    submit_disabled = not disruption_text.strip()
    if not st.button(
        "⚡ Adapt Schedule",
        type="primary",
        disabled=submit_disabled,
        key="adapt_schedule_btn",
    ):
        return None, None

    # ── Processing ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Revised Schedule")

    info_box = st.info(
        "🔍 Retrieving schedule from knowledge base and calculating adjustments…",
        icon="⏳",
    )

    try:
        stream, sources = run_schedule_adapter(disruption_text)

        # Clear the "loading" box before streaming starts
        info_box.empty()

        # Stream the revised schedule token-by-token
        revised_schedule = st.write_stream(stream)

        # ── Source attribution ─────────────────────────────────────────────
        if sources:
            with st.expander("📎 Schedule documents referenced", expanded=True):
                for src in sources:
                    st.caption(f"• {src}")

        # ── Download button ────────────────────────────────────────────────
        download_content = (
            f"DISRUPTION REPORT\n"
            f"{'─' * 50}\n"
            f"{disruption_text}\n\n"
            f"REVISED SCHEDULE\n"
            f"{'─' * 50}\n"
            f"{revised_schedule}"
        )
        st.download_button(
            label="⬇️  Download Revised Schedule (.txt)",
            data=download_content,
            file_name="revised_schedule.txt",
            mime="text/plain",
            key=f"dl_schedule_{len(st.session_state[_HISTORY_KEY])}",
        )

        # ── Persist to history ─────────────────────────────────────────────
        st.session_state[_HISTORY_KEY].append({
            "disruption": disruption_text,
            "schedule":   revised_schedule,
            "sources":    sources,
        })

        return revised_schedule, sources

    except RuntimeError as exc:
        info_box.empty()
        st.error(f"⚠️ **Knowledge Base Error:** {exc}")
        return None, None

    except ValueError as exc:
        info_box.empty()
        st.error(f"🔑 **Configuration Error:** {exc}")
        return None, None

    except Exception as exc:
        info_box.empty()
        st.error(f"❌ **Unexpected Error:** {type(exc).__name__}: {exc}")
        return None, None


def _render_history() -> None:
    """Renders a collapsible log of all adaptation requests this session."""
    history: list = st.session_state.get(_HISTORY_KEY, [])

    if not history:
        return

    st.divider()

    with st.expander(
        f"🕐 Adaptation History — {len(history)} entr{'y' if len(history)==1 else 'ies'} this session",
        expanded=False,
    ):
        # Show in reverse-chronological order (newest first)
        for idx, entry in enumerate(reversed(history), start=1):
            ordinal = len(history) - idx + 1
            st.markdown(f"#### Entry #{ordinal}")

            st.markdown("**Disruption reported:**")
            st.info(entry["disruption"])

            with st.expander(f"View Revised Schedule (Entry #{ordinal})"):
                st.markdown(entry["schedule"])

                if entry.get("sources"):
                    st.caption("Sources: " + ", ".join(entry["sources"]))

            st.divider()

        # Bulk clear
        if st.button("🗑️ Clear History", key="clear_schedule_history", type="secondary"):
            st.session_state[_HISTORY_KEY] = []
            st.rerun()
