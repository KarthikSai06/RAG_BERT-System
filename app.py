import streamlit as st
import time
from src.pipeline import RAGPipeline
import config

st.set_page_config(page_title="Institutional RAG Helpdesk", page_icon=None, layout="wide")

@st.cache_resource
def get_pipeline():
    return RAGPipeline(init_generator=True)

try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

st.title("Institutional Helpdesk — RAG + BERT Domain Guardrail")
st.markdown("Submit queries related to admissions, fees, examinations, hostel, or scholarship policies.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Source Passages"):
                for idx, src in enumerate(message["sources"]):
                    st.markdown(f"**[{idx+1}] {src['source_file']}** (Score: {src['score']:.4f})")
                    st.write(src["text"])
        if message.get("timings"):
            t = message["timings"]
            st.caption(
                f"Timings — Classification: {t.get('classification_ms', 0)} ms | "
                f"Retrieval: {t.get('retrieval_ms', 0)} ms | "
                f"Generation: {t.get('generation_ms', 0)} ms"
            )

if prompt := st.chat_input("Enter your query here..."):
    st.chat_message("user").markdown(prompt)

    with st.spinner("Processing..."):
        result = pipeline.run(prompt)

    label = result["label"]
    if label == "in-domain-factual":
        badge = ":green[IN-DOMAIN FACTUAL]"
    elif label == "in-domain-procedural":
        badge = ":blue[IN-DOMAIN PROCEDURAL]"
    else:
        badge = ":red[OUT-OF-DOMAIN]"

    response_content = f"{badge}\n\n{result['answer']}"

    with st.chat_message("assistant"):
        st.markdown(response_content)
        if result.get("sources"):
            with st.expander("Source Passages"):
                for idx, src in enumerate(result["sources"]):
                    st.markdown(f"**[{idx+1}] {src['source_file']}** (Score: {src['score']:.4f})")
                    st.write(src["text"])

        t = result.get("timings", {})
        timing_line = f"Timings — Classification: {t.get('classification_ms', 0)} ms"
        if not result["is_fallback"]:
            timing_line += (
                f" | Retrieval: {t.get('retrieval_ms', 0)} ms"
                f" | Generation: {t.get('generation_ms', 0)} ms"
            )
        st.caption(timing_line)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_content,
        "sources": result.get("sources", []),
        "timings": result.get("timings", {}),
    })
