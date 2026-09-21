"""Optional lightweight reranker; replace with CrossEncoder for a larger corpus."""
import re
import time

def rerank(question: str, hits: list[dict], top_k: int = 5):
    started = time.perf_counter()
    terms = set(re.findall(r"\w+", question.lower()))
    ranked=[]
    for hit in hits:
        overlap=len(terms & set(re.findall(r"\w+", hit["text"].lower()))) / max(len(terms), 1)
        item=dict(hit); item["rerank_score"]=round(0.7 * hit["score"] + 0.3 * overlap, 6); ranked.append(item)
    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return ranked[:top_k], round((time.perf_counter() - started) * 1000, 3)
