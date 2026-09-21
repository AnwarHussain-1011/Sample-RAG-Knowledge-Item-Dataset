"""Environment-driven application configuration."""
from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "rag_question_answer_dataset.csv"
if not DEFAULT_DATA_PATH.exists():
    DEFAULT_DATA_PATH = ROOT_DIR / "archive" / "rag_question_answer_dataset.csv"

@dataclass(frozen=True)
class Settings:
    data_path: Path = DEFAULT_DATA_PATH
    index_path: Path = ROOT_DIR / "data" / "index.json"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "tfidf")
    llm_provider: str = os.getenv("LLM_PROVIDER", "extractive")
    llm_model: str = os.getenv("LLM_MODEL", "")
    vector_store: str = os.getenv("VECTOR_STORE", "local")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "60"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.0"))
    candidate_k: int = int(os.getenv("CANDIDATE_K", "20"))
    use_reranker: bool = os.getenv("USE_RERANKER", "false").lower() == "true"

settings = Settings()
