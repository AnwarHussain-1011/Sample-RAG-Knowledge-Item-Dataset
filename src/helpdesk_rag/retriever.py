"""Dense-style, lexical, hybrid, and optional reranked retrieval."""
from dataclasses import asdict
import math
import re
import time
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LocalRetriever:
    def __init__(self, chunks, embedding_model="tfidf"):
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks]) if chunks else None
        self.tokens = [re.findall(r"\w+", c.text.lower()) for c in chunks]
        self.document_frequency = Counter(token for row in self.tokens for token in set(row))
        self.avg_length = sum(map(len, self.tokens)) / max(len(self.tokens), 1)

    def _dense(self, query, limit):
        if self.matrix is None or not query.strip(): return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        return [(i, float(score), "dense") for i, score in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:limit]]

    def _bm25(self, query, limit, k1=1.5, b=0.75):
        query_tokens = re.findall(r"\w+", query.lower())
        results = []
        total = len(self.tokens)
        for index, tokens in enumerate(self.tokens):
            counts = Counter(tokens); score = 0.0
            for token in query_tokens:
                if token not in counts: continue
                df = self.document_frequency[token]
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                norm = counts[token] + k1 * (1 - b + b * len(tokens) / max(self.avg_length, 1))
                score += idf * counts[token] * (k1 + 1) / norm
            results.append((index, score, "bm25"))
        return sorted(results, key=lambda x: x[1], reverse=True)[:limit]

    def search(self, query: str, top_k=5, mode="hybrid", threshold=0.0):
        started = time.perf_counter()
        if not query or not query.strip(): raise ValueError("question must not be empty")
        candidate_limit = max(top_k * 4, 10)
        dense = self._dense(query, candidate_limit) if mode in {"dense", "hybrid"} else []
        lexical = self._bm25(query, candidate_limit) if mode in {"bm25", "hybrid"} else []
        combined = {}
        for index, score, method in dense + lexical:
            normalized = score / max((dense if method == "dense" else lexical)[0][1], 1e-9) if score else 0.0
            combined[index] = max(combined.get(index, 0.0), normalized)
        hits = []
        for index, score in sorted(combined.items(), key=lambda item: item[1], reverse=True)[:top_k]:
            if score >= threshold:
                hit = asdict(self.chunks[index]); hit.update({"score": round(float(score), 6), "retrieval_method": mode})
                hits.append(hit)
        return hits, {"retrieval_latency_ms": round((time.perf_counter() - started) * 1000, 3), "candidate_count": len(combined), "mode": mode}
