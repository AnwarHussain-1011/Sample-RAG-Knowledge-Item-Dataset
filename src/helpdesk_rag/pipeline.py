"""End-to-end query pipeline and observability payload."""
import time
from .chunking import chunk_documents
from .data_loader import load_dataset
from .generator import GroundedGenerator
from .retriever import LocalRetriever
from .reranker import rerank

class RAGPipeline:
    def __init__(self, records, chunk_size=400, chunk_overlap=60, embedding_model="tfidf", llm_provider="extractive", llm_model=""):
        self.records = records
        self.chunks = chunk_documents(records.to_dict("records"), chunk_size, chunk_overlap)
        self.retriever = LocalRetriever(self.chunks, embedding_model)
        self.generator = GroundedGenerator(llm_provider, llm_model)

    @classmethod
    def from_csv(cls, path, **kwargs):
        return cls(load_dataset(path), **kwargs)

    def query(self, question: str, top_k=5, mode="hybrid", threshold=0.0, use_reranker=False):
        started = time.perf_counter()
        hits, retrieval_meta = self.retriever.search(question, top_k, mode, threshold)
        if use_reranker:
            hits, rerank_latency = rerank(question, hits, top_k)
            retrieval_meta["reranking_latency_ms"] = rerank_latency
            retrieval_meta["reranked"] = True
        answer, generation_meta = self.generator.generate(question, hits)
        return {"answer": answer, "sources": [{"title": h["title"], "topic": h["topic"], "chunk_id": h["chunk_id"], "score": h["score"]} for h in hits], "retrieved_chunks": hits, "retrieval_metadata": retrieval_meta, "latency": {**generation_meta, "total_latency_ms": round((time.perf_counter() - started) * 1000, 3)}}
