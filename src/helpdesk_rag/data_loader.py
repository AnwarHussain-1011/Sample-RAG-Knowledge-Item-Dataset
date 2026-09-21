"""Load and normalize both the supplied sample schema and the named Kaggle schema."""
from pathlib import Path
from typing import Any
import pandas as pd

REQUIRED_CANONICAL = {"document_id", "title", "topic", "text", "question", "answer"}

class DataValidationError(ValueError):
    """Raised when the knowledge CSV cannot be mapped to the canonical schema."""

def _first_column(frame: pd.DataFrame, names: list[str], default: str = "") -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name].fillna("").astype(str)
    return pd.Series([default] * len(frame), index=frame.index)

def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read a CSV and map known dataset variants to stable application fields."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    raw = pd.read_csv(source)
    result = pd.DataFrame({
        "document_id": _first_column(raw, ["document_id", "id", "ki_topic"]),
        "title": _first_column(raw, ["document_title", "title", "ki_topic"], "Untitled knowledge item"),
        "topic": _first_column(raw, ["category", "topic", "ki_topic"], "Uncategorized"),
        "text": _first_column(raw, ["document", "ki_text"]),
        "question": _first_column(raw, ["question", "sample_question"]),
        "answer": _first_column(raw, ["answer", "sample_ground_truth"]),
        "keywords": _first_column(raw, ["keywords"]),
        "source": _first_column(raw, ["source", "source_type"], "CSV"),
    })
    if result[["text", "question", "answer"]].eq("").all(axis=1).any():
        raise DataValidationError("Rows must contain document text, a question, and an answer")
    result["document_id"] = result["document_id"].replace("", pd.NA).fillna(pd.Series(result.index, index=result.index).map(lambda x: f"DOC_{x+1:06d}"))
    return result.reset_index(drop=True)

def profile_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-serializable EDA metrics used by the notebook and README."""
    lengths = {column: frame[column].str.len().describe().to_dict() for column in ["text", "question", "answer"]}
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_values": {k: int(v) for k, v in frame.isna().sum().to_dict().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_documents": int(frame["text"].duplicated().sum()),
        "lengths_characters": lengths,
        "topic_distribution": {str(k): int(v) for k, v in frame["topic"].value_counts(dropna=False).to_dict().items()},
        "unique_values": {column: int(frame[column].nunique(dropna=False)) for column in frame.columns},
    }
