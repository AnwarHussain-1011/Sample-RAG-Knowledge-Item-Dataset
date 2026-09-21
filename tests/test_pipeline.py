import pandas as pd
from helpdesk_rag.pipeline import RAGPipeline

def _pipeline():
    return RAGPipeline(pd.DataFrame([{"document_id":"D1","title":"VPN","topic":"network","text":"VPN access instructions","question":"VPN?","answer":"Open the VPN client","source":"test"}]))

def test_pipeline_returns_observability_and_sources():
    result=_pipeline().query("VPN access")
    assert result["answer"]
    assert result["sources"]
    assert result["latency"]["total_latency_ms"] >= 0

def test_no_context_response():
    result=_pipeline().query("completely unrelated zebra", top_k=1)
    assert "knowledge base" not in result["answer"].lower() or result["sources"]
