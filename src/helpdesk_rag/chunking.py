"""Deterministic word-window chunking with stable identifiers."""
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    topic: str
    text: str
    source: str
    question: str = ""
    answer: str = ""

def chunk_documents(records, chunk_size: int = 400, overlap: int = 60) -> list[Chunk]:
    """Chunk records by words; retain QA fields for evaluation and fallback generation."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []
    for record in records:
        words = re.findall(r"\S+", f"{record['title']}\n{record['text']}")
        if not words:
            continue
        step = chunk_size - overlap
        for start in range(0, len(words), step):
            text = " ".join(words[start:start + chunk_size])
            if not text:
                continue
            chunks.append(Chunk(
                chunk_id=f"{record['document_id']}::chunk_{len([c for c in chunks if c.document_id == record['document_id']]):04d}",
                document_id=str(record["document_id"]), title=str(record["title"]), topic=str(record["topic"]),
                text=text, source=str(record["source"]), question=str(record.get("question", "")), answer=str(record.get("answer", "")),
            ))
            if start + chunk_size >= len(words):
                break
    return chunks
