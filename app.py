import streamlit as st
import time
from src.pipeline import RAGPipeline
import config

st.set_page_config(page_title="Domain RAG Helpdesk", page_icon="🎓", layout="wide")

# Initialize pipeline once and cache it
@st.cache_resource
def get_pipeline():
    # Set init_generator=False for faster UI testing without the LLM loaded, 
    # but normally it should be True
    return RAGPipeline(init_generator=True)

try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"Failed to load pipeline. {e}")
    st.stop()

st.title("🎓 Institutional Helpdesk (RAG + BERT Guardrail)")
st.markdown("Ask questions about **admissions, fees, examinations, and hostels**.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📄 Source Passages"):
                for idx, src in enumerate(message["sources"]):
                    st.markdown(f"**[{idx+1}] {src['source_file']}** (Score: {src['score']:.4f})")
                    st.write(src['text'])
        if "timings" in message and message["timings"]:
            st.caption(f"⏱️ Timings: Classification {message['timings'].get('classification_ms', 0)}ms | Retrieval {message['timings'].get('retrieval_ms', 0)}ms | Generation {message['timings'].get('generation_ms', 0)}ms")

# React to user input
if prompt := st.chat_input("Ask a question..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    with st.spinner("Processing query..."):
        result = pipeline.run(prompt)
    
    # Format the response
    label = result["label"]
    badge_color = "green" if "in-domain" in label else "red"
    response_header = f":{badge_color}[**{label.upper()}**]"
    
    response_content = f"{response_header}\n\n{result['answer']}"
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_content)
        if result["sources"]:
            with st.expander("📄 Source Passages"):
                for idx, src in enumerate(result["sources"]):
                    st.markdown(f"**[{idx+1}] {src['source_file']}** (Score: {src['score']:.4f})")
                    st.write(src['text'])
        
        # Display timings
        timings_str = f"⏱️ Timings: Classification {result['timings'].get('classification_ms', 0)}ms"
        if not result['is_fallback']:
            timings_str += f" | Retrieval {result['timings'].get('retrieval_ms', 0)}ms | Generation {result['timings'].get('generation_ms', 0)}ms"
        st.caption(timings_str)
        
    # Add to session state
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_content,
        "sources": result.get("sources", []),
        "timings": result.get("timings", {})
    })
