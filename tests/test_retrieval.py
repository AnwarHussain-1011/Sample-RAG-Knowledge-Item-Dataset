import pandas as pd
import pytest
from helpdesk_rag.pipeline import RAGPipeline

def test_retrieval_finds_relevant_document():
    frame=pd.DataFrame([{"document_id":"D1","title":"VPN","topic":"network","text":"configure remote VPN access","question":"How VPN?","answer":"Use the VPN client","source":"test"},{"document_id":"D2","title":"Printer","topic":"hardware","text":"replace printer toner","question":"How printer?","answer":"Replace toner","source":"test"}])
    result=RAGPipeline(frame).query("How do I configure VPN access?", top_k=1)
    assert result["sources"][0]["chunk_id"].startswith("D1")

def test_empty_query_rejected():
    frame=pd.DataFrame([{"document_id":"D1","title":"T","topic":"x","text":"text","question":"q","answer":"a","source":"test"}])
    with pytest.raises(ValueError): RAGPipeline(frame).query(" ")
