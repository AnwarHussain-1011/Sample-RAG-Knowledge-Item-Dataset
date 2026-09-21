"""Retrieval metrics and transparent answer checks for small QA datasets."""
from difflib import SequenceMatcher
import re

def _terms(value): return set(re.findall(r"\w+", str(value).lower()))

def evaluate_retrieval(pipeline, records, top_k=5, mode="hybrid"):
    rows=[]; reciprocal=[]; hits=[]
    for record in records.to_dict("records"):
        result=pipeline.query(record["question"], top_k, mode)
        relevant=str(record["document_id"])
        ids=[item["document_id"] for item in result["retrieved_chunks"]]
        rank=ids.index(relevant)+1 if relevant in ids else 0
        hits.append(bool(rank)); reciprocal.append(1/rank if rank else 0)
        rows.append({"question":record["question"],"expected_document_id":relevant,"rank":rank,"retrieved":ids})
    return {"mode":mode,"k":top_k,"queries":len(rows),"hit_rate_at_k":sum(hits)/max(len(hits),1),"recall_at_k":sum(hits)/max(len(hits),1),"mrr":sum(reciprocal)/max(len(reciprocal),1),"details":rows}

def evaluate_answers(pipeline, records, top_k=5, mode="hybrid"):
    scores=[]
    for record in records.to_dict("records"):
        result=pipeline.query(record["question"], top_k, mode)
        expected=_terms(record["answer"]); actual=_terms(result["answer"])
        scores.append({"question":record["question"],"token_overlap":len(expected & actual)/max(len(expected),1),"sequence_similarity":SequenceMatcher(None,record["answer"].lower(),result["answer"].lower()).ratio()})
    return {"queries":len(scores),"mean_token_overlap":sum(x["token_overlap"] for x in scores)/max(len(scores),1),"mean_sequence_similarity":sum(x["sequence_similarity"] for x in scores)/max(len(scores),1),"details":scores}
