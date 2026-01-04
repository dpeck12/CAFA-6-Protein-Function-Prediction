# CAFA-6 Protein Function Prediction — Baseline

Overview
- Predict Gene Ontology (GO) terms (MF, BP, CC) for proteins from sequences.
- This baseline uses 3-mer frequency features with one-vs-rest logistic regression.
- Outputs a single TSV submission combining MF/BP/CC with probabilities in (0, 1.000], up to 3 significant figures.

Setup (Windows)
1) Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Download Kaggle data into `data/raw/` (requires Kaggle CLI configured):

```powershell
kaggle competitions download -c cafa-6-protein-function-prediction -p data/raw
Expand-Archive -Path data\raw\cafa-6-protein-function-prediction.zip -DestinationPath data\raw
```

Expected files in `data/raw/` (from competition Data page):
- `train_sequences.fasta`, `train_terms.tsv`, `train_taxonomy.tsv`
- `go-basic.obo`, `IA.tsv`
- `testsuperset.fasta`, `testsuperset-taxon-list.tsv`
- `sample_submission.tsv`

Quick Start
Train the baseline model and generate a submission TSV:

```powershell
python -m src.train --data-root data/raw --model-path models/baseline_logreg.joblib --min-positive 20
python -m src.predict --data-root data/raw --model-path models/baseline_logreg.joblib --obo-path data/raw/go-basic.obo --out-path outputs/submission.tsv --min-prob 0.001 --max-terms 1500
```

Nearest-Neighbor Baseline (TF‑IDF 2‑gram)
Build a sequence-similarity baseline using character 2‑gram TF‑IDF and cosine nearest neighbors:

```powershell
# Quick dev run (skip DAG/ancestors to iterate fast)
python -m src.nn_baseline --data-root data/raw --out-path outputs/submission_nn_partial.tsv --ngram-n 2 --n-neighbors 20 --min-positive 20 --min-prob 0.01 --max-terms 800 --max-train 8000 --max-test 400 --skip-ancestors --ia-path data/raw/IA.tsv

# Full run with ancestor propagation
python -m src.nn_baseline --data-root data/raw --out-path outputs/submission_nn.tsv --ngram-n 2 --n-neighbors 25 --min-positive 20 --min-prob 0.005 --max-terms 1500 --max-train 20000 --ia-path data/raw/IA.tsv --taxon-weight-same 1.2
```

Notes:
- Training/test caps (`--max-train`, `--max-test`) speed up iteration; remove for full runs.
- Ancestor propagation is applied to neighbor-aggregated scores via the GO DAG.
- Predictions are filtered to evaluation terms listed in `IA.tsv`.
- Optional taxonomy-aware weighting boosts neighbors from the same taxonomy ID.

Embedding NN Baseline (Word2Vec k‑mers)
Train k‑mer Word2Vec embeddings, average per sequence, and run cosine NN:

```powershell
# Quick dev run
python -m src.embedding_nn --data-root data/raw --out-path outputs/submission_embed_partial.tsv --k 3 --vector-size 128 --window 10 --n-neighbors 80 --min-positive 20 --min-prob 0.001 --max-terms 1200 --max-train 10000 --max-test 1000 --ia-path data/raw/IA.tsv

# Full run (ancestor propagation enabled)
python -m src.embedding_nn --data-root data/raw --out-path outputs/submission_embed.tsv --k 3 --vector-size 256 --window 12 --n-neighbors 120 --min-positive 20 --min-prob 0.0008 --max-terms 1500 --max-train 30000 --ia-path data/raw/IA.tsv
```

Notes:
- Sequence embedding is the mean of k‑mer vectors (skip‑gram); taxonomy weighting and softmax neighbor weights applied.
- Use `--ia-path` for evaluation term filtering; ancestor propagation improves recall.
- Probabilities are normalized neighbor-similarity weights and formatted as above.

Notes
- Ancestor propagation is performed after prediction using the GO DAG (from `go-basic.obo`). Parent scores are the max of children scores.
- Probabilities are formatted with 3 significant figures (e.g., 0.123, 0.0123) and must be > 0.
- Cap per protein is enforced at 1,500 terms across MF+BP+CC.
- For a sanity check without Kaggle data:

```powershell
python -m src.sanity_check
```

Project Structure
- `src/data_ingest.py`: Load sequences and labels.
- `src/features.py`: 3-mer features.
- `src/train.py`: Train logistic regression one-vs-rest model.
- `src/go_graph.py`: Parse GO DAG and compute ancestors.
- `src/predict.py`: Predict, propagate ancestors, and write TSV.
- `src/sanity_check.py`: Tiny self-contained test run.