import html

import streamlit as st

import config
from src.pipeline import RAGPipeline


st.set_page_config(
    page_title="CampusDesk | Institutional Helpdesk",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #172033;
        --muted: #64748b;
        --line: #e7eaf0;
        --blue: #315efb;
        --blue-dark: #2146c7;
        --soft-blue: #eef3ff;
        --green: #0c9b70;
        --soft-green: #eafaf4;
        --surface: #ffffff;
    }
    .stApp {
        background: linear-gradient(135deg, #f7f9fc 0%, #edf3ff 52%, #f9fbff 100%);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: #111a2d;
        border-right: 0;
    }
    [data-testid="stSidebar"] * { color: #dbe5ff; }
    [data-testid="stSidebar"] .stButton button {
        color: #e7edff;
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.12);
        text-align: left;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        border-color: #7c9cff;
        color: white;
    }
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 7rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
    .brand {
        display: flex; align-items: center; gap: 13px; margin-bottom: 2rem;
    }
    .brand-mark {
        width: 46px; height: 46px; border-radius: 14px;
        display: grid; place-items: center; color: white; font-weight: 700;
        font-family: 'Space Grotesk', sans-serif; font-size: 20px;
        background: linear-gradient(135deg, #315efb, #6a42e8);
        box-shadow: 0 10px 24px rgba(49,94,251,.25);
    }
    .brand-name { font: 700 21px 'Space Grotesk', sans-serif; letter-spacing: -.4px; }
    .brand-sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .hero {
        background: linear-gradient(120deg, #18284c, #315efb);
        border-radius: 25px; padding: 2.2rem 2.4rem; color: white;
        box-shadow: 0 18px 45px rgba(32,71,175,.18); margin-bottom: 1.5rem;
    }
    .hero h1 { color: white; font-size: 2.2rem; margin: 0 0 .55rem; letter-spacing: -.8px; }
    .hero p { color: #dce6ff; max-width: 690px; margin: 0; font-size: 1rem; line-height: 1.6; }
    .pill {
        display: inline-block; padding: 5px 11px; border-radius: 99px;
        font-size: 11px; font-weight: 700; letter-spacing: .4px; margin-bottom: 13px;
        background: rgba(255,255,255,.14); color: #eaf0ff;
    }
    .section-label {
        color: var(--muted); text-transform: uppercase; letter-spacing: 1.4px;
        font-size: 11px; font-weight: 700; margin: 1.4rem 0 .8rem;
    }
    .topic-card {
        background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
        padding: 15px 16px; min-height: 102px; box-shadow: 0 5px 18px rgba(26,42,76,.04);
    }
    .topic-icon { font-size: 21px; margin-bottom: 8px; }
    .topic-title { font-weight: 700; font-size: 14px; }
    .topic-copy { color: var(--muted); font-size: 12px; margin-top: 4px; line-height: 1.35; }
    .answer-card {
        background: var(--surface); border: 1px solid var(--line); border-radius: 18px;
        padding: 18px 20px; margin: 8px 0 12px; box-shadow: 0 7px 24px rgba(26,42,76,.05);
    }
    .answer-label { font-size: 11px; font-weight: 700; color: var(--blue); letter-spacing: .8px; }
    .answer-text { font-size: 15px; line-height: 1.7; margin-top: 7px; white-space: pre-wrap; }
    .user-card {
        background: #e9efff; border: 1px solid #d5e0ff; border-radius: 18px;
        padding: 15px 19px; margin: 18px 0 8px; font-size: 15px; line-height: 1.5;
    }
    .meta {
        color: var(--muted); font-size: 11px; margin: 4px 0 16px; padding-left: 2px;
    }
    .status {
        padding: 10px 12px; border-radius: 12px; font-size: 12px; margin: 5px 0 18px;
        background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.1);
    }
    .source-title { font-weight: 700; color: var(--ink); font-size: 13px; }
    .source-text { color: var(--muted); font-size: 12px; line-height: 1.55; margin-top: 5px; }
    div[data-testid="stChatInput"] {
        background: rgba(247,249,252,.96); padding: 12px 0 18px;
    }
    div[data-testid="stChatInput"] > div {
        max-width: 920px; margin: 0 auto; padding: 5px 7px 5px 12px;
        border: 1px solid #cbd6ed; border-radius: 18px; background: #fff;
        box-shadow: 0 12px 30px rgba(37,57,102,.13);
    }
    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] textarea:focus {
        color: #172033 !important; -webkit-text-fill-color: #172033 !important;
        caret-color: #315efb !important; background: #fff !important;
        border: 0 !important; box-shadow: none !important; font-size: 15px;
        line-height: 1.45; padding: 12px 8px !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #8a98ad !important; -webkit-text-fill-color: #8a98ad !important;
        opacity: 1 !important;
    }
    div[data-testid="stChatInput"] button {
        color: #fff !important; background: #315efb !important;
        border: 0 !important; border-radius: 12px !important;
        width: 42px; height: 42px; margin-right: 2px;
    }
    div[data-testid="stChatInput"] button:hover { background: #2146c7 !important; }
    div[data-testid="stChatInput"] button:disabled {
        color: #fff !important; background: #b8c5e5 !important; opacity: 1 !important;
    }
    .composer-hint {
        max-width: 920px; margin: 0 auto; padding: 7px 5px 0;
        color: #8491a5; font-size: 11px; text-align: right;
    }
    .empty-state { text-align: center; color: var(--muted); padding: 2.5rem 1rem 1rem; }
    .empty-state strong { display: block; color: var(--ink); font-size: 18px; margin-bottom: 7px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    return RAGPipeline(init_generator=config.ENABLE_LLM)


def reset_chat():
    st.session_state.messages = []


try:
    pipeline = get_pipeline()
except Exception as exc:
    st.error(f"Unable to load the helpdesk pipeline: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">C</div>
            <div><div class="brand-name" style="color:#fff">CampusDesk</div>
            <div class="brand-sub">Institutional knowledge assistant</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'margin:0 2px 4px;color:#64748b;font-size:12px">'
        '<span><b style="color:#315efb">Official knowledge base</b> &nbsp;·&nbsp; '
        'Answers cite uploaded institutional documents</span>'
        '<span style="color:#0c9b70;font-weight:700">● Secure session</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label" style="color:#8fa5d8">System status</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="status"><b style="color:#7cf0c5">● Online</b><br>'
        '<span style="color:#aebde0">BERT guardrail &nbsp;·&nbsp; FAISS retrieval</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label" style="color:#8fa5d8">What you can ask</div>', unsafe_allow_html=True)
    st.markdown(
        "Admissions, fees and payments, examinations, hostel rules, scholarships, fee waivers, and refunds.",
        unsafe_allow_html=True,
    )
    st.markdown("")
    if st.button("Clear conversation", use_container_width=True):
        reset_chat()
        st.rerun()
    st.caption("Answers are grounded only in the institution's uploaded documents.")

st.markdown(
    '<div class="hero"><div class="pill">AI-POWERED STUDENT SUPPORT</div>'
    '<h1>Your campus questions, answered.</h1>'
    '<p>Ask about institutional policies and procedures. CampusDesk searches official documents '
    'and shows the source passages behind every answer.</p></div>',
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown('<div class="section-label">Explore helpdesk topics</div>', unsafe_allow_html=True)
    topics = [
        ("◷", "Fees & payments", "Deadlines, late fees and receipts"),
        ("▣", "Examinations", "Attendance and registration"),
        ("⌂", "Hostel", "Applications, rules and curfew"),
        ("★", "Scholarships", "Merit and need-based aid"),
    ]
    cols = st.columns(4)
    for col, (icon, title, copy) in zip(cols, topics):
        with col:
            st.markdown(
                f'<div class="topic-card"><div class="topic-icon">{icon}</div>'
                f'<div class="topic-title">{title}</div><div class="topic-copy">{copy}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="empty-state"><strong>How can we help today?</strong>'
        'Try asking a question below, such as “What is the hostel curfew time?”</div>',
        unsafe_allow_html=True,
    )


def render_message(message):
    if message["role"] == "user":
        st.markdown(
            f'<div class="user-card">{html.escape(message["content"])}</div>',
            unsafe_allow_html=True,
        )
        return

    label = message.get("label", "assistant")
    label_text = {
        "in-domain-factual": "FACTUAL ANSWER",
        "in-domain-procedural": "PROCEDURE ANSWER",
        "out-of-domain": "OUTSIDE HELPdesk SCOPE",
    }.get(label, "HELPDESK ANSWER")
    st.markdown(
        f'<div class="answer-card"><div class="answer-label">{label_text}</div>'
        f'<div class="answer-text">{html.escape(message["content"])}</div></div>',
        unsafe_allow_html=True,
    )
    timings = message.get("timings", {})
    timing_parts = [f"Classification {timings.get('classification_ms', 0)} ms"]
    if "retrieval_ms" in timings:
        timing_parts.append(f"Retrieval {timings['retrieval_ms']} ms")
    if "generation_ms" in timings:
        timing_parts.append(f"Generation {timings['generation_ms']} ms")
    st.markdown(
        f'<div class="meta">{"  ·  ".join(timing_parts)}</div>',
        unsafe_allow_html=True,
    )
    if message.get("sources"):
        with st.expander(f"View {len(message['sources'])} source passages"):
            for index, source in enumerate(message["sources"], 1):
                st.markdown(
                    f'<div class="source-title">{index}. {html.escape(source["source_file"])} '
                    f'<span style="color:#315efb">· relevance {source["score"]:.3f}</span></div>'
                    f'<div class="source-text">{html.escape(source["text"])}</div>',
                    unsafe_allow_html=True,
                )
                if index < len(message["sources"]):
                    st.divider()


for message in st.session_state.messages:
    render_message(message)

prompt = st.chat_input("Ask about fees, exams, hostel, admissions or scholarships...")
st.markdown(
    '<div class="composer-hint">Press Enter to send &nbsp;·&nbsp; '
    'Try: “What documents are needed for a scholarship?”</div>',
    unsafe_allow_html=True,
)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Searching official institutional documents..."):
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
    st.rerun()
