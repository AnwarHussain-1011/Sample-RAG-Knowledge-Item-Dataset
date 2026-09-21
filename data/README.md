# Data

Place the Kaggle CSV here as `rag_question_answer_dataset.csv` (or set the data path in the application configuration). The repository currently profiles the supplied `archive/rag_question_answer_dataset.csv` file, which is a 200-row synthetic dataset with 199 duplicate document bodies. It is retained outside `data/` so the app does not silently ship a misleading corpus.

Expected supported schemas:

- Kaggle-style: `ki_topic`, `ki_text`, `sample_question`, `sample_ground_truth`
- Supplied sample: `document_id`, `document_title`, `category`, `document`, `question`, `answer`
