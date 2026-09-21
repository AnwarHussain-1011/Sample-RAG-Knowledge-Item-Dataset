# Evaluation reports

`python scripts/evaluate.py` writes measured JSON results to `evaluation/results/latest.json`. Interpret metrics together with the dataset profile: the supplied sample has 199 duplicate document bodies, so document-level recall is a data-quality diagnostic rather than a production performance claim.
