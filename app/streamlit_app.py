"""Streamlit chat and retrieval inspection UI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import streamlit as st
from helpdesk_rag.config import settings
from helpdesk_rag.pipeline import RAGPipeline

st.set_page_config(page_title="HelpDeskRAG", page_icon="?", layout="wide")
st.title("HelpDeskRAG")
st.caption("Inspectable retrieval-augmented IT support assistant")
@st.cache_resource
def load_pipeline():
    return RAGPipeline.from_csv(settings.data_path, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap, embedding_model=settings.embedding_model, llm_provider=settings.llm_provider, llm_model=settings.llm_model)
pipeline=load_pipeline()
with st.sidebar:
    st.header("Retrieval controls")
    mode=st.selectbox("Retriever", ["hybrid", "dense", "bm25"])
    top_k=st.slider("Top K", 1, 10, settings.top_k)
    inspect=st.checkbox("Debug / inspection mode", value=True)
question=st.text_input("Ask an IT support question", placeholder="How do I configure VPN access from home?")
if question:
    try:
        result=pipeline.query(question, top_k, mode)
        st.subheader("Answer")
        st.write(result["answer"])
        st.subheader("Sources")
        st.dataframe(result["sources"], use_container_width=True, hide_index=True)
        st.caption(f"Total latency: {result['latency']['total_latency_ms']} ms")
        if inspect:
            with st.expander("RAG inspection", expanded=True):
                st.json({"question": question, "retrieval_metadata": result["retrieval_metadata"], "latency": result["latency"]})
                for index, chunk in enumerate(result["retrieved_chunks"], 1):
                    st.markdown(f"**{index}. {chunk['title']} | score={chunk['score']} | {chunk['chunk_id']}**")
                    st.write(chunk["text"])
    except ValueError as exc:
        st.error(str(exc))
