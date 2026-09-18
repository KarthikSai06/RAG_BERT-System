import html
import streamlit as st

import config
from src.pipeline import RAGPipeline

st.set_page_config(
    page_title="DSCE Institutional Helpdesk",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal, Institutional CSS ─────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

    :root {
        --bg: #ffffff;
        --surface: #ffffff;
        --border: #cccccc;
        --text-main: #333333;
        --text-muted: #666666;
        --primary: #003366; /* Deep University Blue */
        --primary-light: #e6f0fa;
    }

    /* Global base styling */
    .stApp {
        background-color: var(--bg);
        color: var(--text-main);
        font-family: 'Roboto', Arial, sans-serif;
    }

    [data-testid="stHeader"] {
        background-color: rgba(255, 255, 255, 0.95);
    }
    
    /* Ensure header icons (sidebar toggle) are visible */
    [data-testid="stHeader"] svg {
        fill: var(--text-main) !important;
        stroke: var(--text-main) !important;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f5f5f5;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text-main);
    }

    .sidebar-brand {
        padding-bottom: 1rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--primary);
    }

    .sidebar-brand-title {
        font-size: 20px;
        font-weight: 700;
        color: var(--primary);
    }

    .sidebar-header {
        font-size: 14px;
        font-weight: 700;
        color: var(--primary);
        margin: 1.5rem 0 0.5rem;
        text-transform: uppercase;
    }

    [data-testid="stSidebar"] .stButton button {
        background-color: #ffffff;
        color: var(--primary);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 13px;
        text-align: left;
        transition: none;
    }

    [data-testid="stSidebar"] .stButton button:hover {
        background-color: var(--primary-light);
        border-color: var(--primary);
    }

    /* Main Header */
    .hero-container {
        border-bottom: 2px solid var(--primary);
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }

    .hero-title {
        font-size: 28px;
        font-weight: 700;
        color: var(--primary);
        margin: 0 0 0.5rem 0;
    }

    .hero-desc {
        font-size: 15px;
        color: var(--text-muted);
        margin: 0;
    }

    /* Message Styling */
    .user-bubble {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #999999;
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 15px;
        color: #333333;
    }

    .bot-card {
        background-color: #ffffff;
        border: 1px solid var(--border);
        border-left: 4px solid var(--primary);
        padding: 16px 18px;
        margin: 12px 0;
    }

    .badge {
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 6px;
        margin-bottom: 8px;
        border: 1px solid var(--border);
        background-color: #f5f5f5;
        color: var(--text-muted);
    }

    .bot-text {
        font-size: 15px;
        line-height: 1.6;
        color: var(--text-main);
        white-space: pre-wrap;
    }

    .meta-bar {
        font-size: 12px;
        color: var(--text-muted);
        margin-top: 8px;
        border-top: 1px solid #eeeeee;
        padding-top: 4px;
    }

    .empty-prompt {
        text-align: left;
        padding: 1rem 0;
        color: var(--text-main);
        font-size: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    return RAGPipeline(init_generator=config.ENABLE_LLM)


try:
    pipeline = get_pipeline()
except Exception as exc:
    st.error(f"Unable to load the helpdesk pipeline: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick query selection state
selected_query = None

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">Dayananda Sagar College of Engineering</div>
            <div style="font-size: 13px; color: #666; margin-top: 4px;">Institutional Helpdesk</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-header">Common Inquiries</div>', unsafe_allow_html=True)
    sample_queries = [
        "What is the semester fee deadline?",
        "How do I apply for a fee waiver?",
        "What is the curfew time for the hostel?",
        "What is the minimum attendance required?",
    ]

    for q in sample_queries:
        if st.button(q, key=f"btn_{q}", use_container_width=True):
            selected_query = q

    st.markdown('<div class="sidebar-header">Options</div>', unsafe_allow_html=True)
    
    # Bug fix for clear history button: we omit st.rerun() so it clears and renders empty state correctly in the same pass.
    if st.button("Clear History", use_container_width=True):
        st.session_state.messages = []

    st.markdown(
        """
        <div style="margin-top: 2rem; font-size: 12px; color: #666;">
            Information is strictly sourced from official university documentation.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Main Header Banner ────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">Student Support & Knowledge Base</h1>
        <p class="hero-desc">
            Please enter your query below regarding admissions, academics, examinations, or facilities.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-prompt">
            <strong>Supported Topics:</strong>
            <ul>
                <li>Fees and Payments</li>
                <li>Examinations and Attendance</li>
                <li>Hostel and Campus Facilities</li>
                <li>Scholarships and Financial Aid</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Render Chat Messages ──────────────────────────────────────────────────────
def render_message(message):
    if message["role"] == "user":
        st.markdown(
            f'<div class="user-bubble"><b>User Query:</b><br>{html.escape(message["content"])}</div>',
            unsafe_allow_html=True,
        )
        return

    label = message.get("label", "assistant")
    if label == "in-domain-factual":
        badge_text = "Factual Response"
    elif label == "in-domain-procedural":
        badge_text = "Procedural Guideline"
    elif label == "out-of-domain":
        badge_text = "Out of Scope"
    else:
        badge_text = "Official Response"

    st.markdown(
        f"""
        <div class="bot-card">
            <span class="badge">{badge_text}</span>
            <div class="bot-text">{html.escape(message["content"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if message.get("sources"):
        with st.expander("Reference Documents"):
            for index, source in enumerate(message["sources"], 1):
                st.markdown(
                    f"**Document {index}: {html.escape(source['source_file'])}**\n\n"
                    f"{html.escape(source['text'])}"
                )

for message in st.session_state.messages:
    render_message(message)

# Handle input from user chat input or sample query click
chat_input = st.chat_input("Enter your institutional query here...")
prompt = selected_query or chat_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Retrieving official guidelines..."):
        result = pipeline.run(prompt)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "label": result["label"],
            "sources": result.get("sources", []),
            "timings": result.get("timings", {}),
        }
    )
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()
