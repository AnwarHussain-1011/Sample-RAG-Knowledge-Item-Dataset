"""Grounded generation adapters with a deterministic local fallback."""
import os
import re
import time
from .prompts import build_prompt

class GroundedGenerator:
    def __init__(self, provider="extractive", model=""):
        self.provider, self.model = provider, model

    def generate(self, question: str, hits: list[dict]) -> tuple[str, dict]:
        started = time.perf_counter()
        if not hits:
            return "The knowledge base does not provide enough information to answer that.", {"generation_latency_ms": 0.0, "provider": "extractive", "grounded": False}
        if self.provider == "openai" and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                response = OpenAI().chat.completions.create(model=self.model or "gpt-4o-mini", temperature=0, messages=[{"role": "system", "content": build_prompt(question, self._context(hits))}])
                answer = response.choices[0].message.content.strip()
                provider = "openai"
            except Exception:
                answer, provider = self._extractive(question, hits), "extractive-fallback"
        else:
            answer, provider = self._extractive(question, hits), "extractive"
        return answer, {"generation_latency_ms": round((time.perf_counter() - started) * 1000, 3), "provider": provider, "grounded": True}

    @staticmethod
    def _context(hits):
        return "\n\n".join(f"[{h['title']}] {h['text']}" for h in hits)

    @staticmethod
    def _extractive(question, hits):
        question_terms = set(re.findall(r"\w+", question.lower()))
        candidates = []
        for hit in hits:
            text = hit.get("answer") or hit["text"]
            overlap = len(question_terms & set(re.findall(r"\w+", text.lower())))
            candidates.append((overlap, hit["score"], text))
        best = max(candidates, key=lambda item: (item[0], item[1]))
        return best[2] + f"\n\nSource: {next(h['title'] for h in hits if h.get('answer') == best[2])}"
