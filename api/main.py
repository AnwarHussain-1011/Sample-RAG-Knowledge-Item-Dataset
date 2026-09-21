"""FastAPI interface for HelpDeskRAG."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from helpdesk_rag.config import settings
from helpdesk_rag.pipeline import RAGPipeline

app = FastAPI(title="HelpDeskRAG API", version="0.1.0")
pipeline = RAGPipeline.from_csv(settings.data_path, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap, embedding_model=settings.embedding_model, llm_provider=settings.llm_provider, llm_model=settings.llm_model)

class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=settings.top_k, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(dense|bm25|hybrid)$")

@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(pipeline.chunks)}

@app.post("/query")
def query(request: QueryRequest):
    try:
        return pipeline.query(request.question, request.top_k, request.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Query processing failed") from exc
