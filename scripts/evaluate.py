"""Run retrieval and answer evaluations across the supported retrievers."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from helpdesk_rag.config import settings
from helpdesk_rag.data_loader import load_dataset
from helpdesk_rag.evaluation import evaluate_answers, evaluate_retrieval
from helpdesk_rag.pipeline import RAGPipeline

frame=load_dataset(settings.data_path)
pipeline=RAGPipeline(frame, settings.chunk_size, settings.chunk_overlap, settings.embedding_model, settings.llm_provider, settings.llm_model)
results=[]
for mode in ["dense", "bm25", "hybrid"]:
    retrieval=evaluate_retrieval(pipeline, frame, settings.top_k, mode)
    answers=evaluate_answers(pipeline, frame, settings.top_k, mode)
    results.append({"configuration": mode, "retrieval": retrieval, "generation": answers})
out=Path("evaluation/results")
out.mkdir(parents=True, exist_ok=True)
path=out / "latest.json"
path.write_text(json.dumps({"dataset": str(settings.data_path), "results": results, "limitations": ["The supplied file has 199 duplicate document bodies.", "Retrieval metrics use document_id relevance and are not representative of a diverse corpus."]}, indent=2))
for item in results:
    print(item["configuration"], item["retrieval"]["hit_rate_at_k"], item["retrieval"]["mrr"], item["generation"]["mean_token_overlap"])
print(f"\nWrote {path}")
