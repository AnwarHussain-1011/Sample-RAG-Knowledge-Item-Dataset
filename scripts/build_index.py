"""Validate and build a local serialized index manifest."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from helpdesk_rag.config import settings
from helpdesk_rag.data_loader import load_dataset, profile_dataset
from helpdesk_rag.pipeline import RAGPipeline

frame=load_dataset(settings.data_path)
pipeline=RAGPipeline(frame, settings.chunk_size, settings.chunk_overlap, settings.embedding_model, settings.llm_provider, settings.llm_model)
settings.index_path.parent.mkdir(parents=True, exist_ok=True)
settings.index_path.write_text(json.dumps({"profile": profile_dataset(frame), "chunks": len(pipeline.chunks), "settings": {"chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap}}, indent=2))
print(json.dumps({"index_manifest": str(settings.index_path), "rows": len(frame), "chunks": len(pipeline.chunks)}, indent=2))
