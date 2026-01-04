import argparse
import os
import joblib
import numpy as np
from typing import Dict, List, Tuple

from .data_ingest import get_test_pairs
from .features import batch_kmer_matrix
from .go_graph import load_go_dag, build_ancestors_map


def format_prob(p: float) -> str:
    # Clamp to (0, 1.0]
    p = max(min(p, 1.0), 1e-12)
    # 3 significant figures
    s = f"{p:.3g}"
    # Ensure upper bound shows as 1.000 if exactly 1.0
    if p == 1.0:
        s = "1.000"
    return s


def predict_and_write(data_root: str, model_path: str, obo_path: str, out_path: str, min_prob: float = 0.001, max_terms: int = 1500, max_test: int | None = None):
    testsuperset = os.path.join(data_root, "testsuperset.fasta")
    accs, seqs = get_test_pairs(testsuperset)
    if max_test is not None and len(accs) > max_test:
        accs = accs[:max_test]
        seqs = seqs[:max_test]
        print(f"Loaded {len(accs)} test proteins (capped via max_test={max_test})")
    else:
        print(f"Loaded {len(accs)} test proteins")

    # Load model
    bundle = joblib.load(model_path)
    clf = bundle['model']
    mlb = bundle['mlb']
    k = bundle.get('k', 3)

    # Features
    X = batch_kmer_matrix(seqs, k=k, sparse_output=True)

    # Predict probabilities
    print("Scoring...")
    probs = clf.predict_proba(X)  # shape (n_samples, n_classes)
    classes = list(mlb.classes_)

    # GO DAG and ancestors
    print("Loading GO DAG...")
    dag = load_go_dag(obo_path)
    ancestors_map = build_ancestors_map(classes, dag)

    # Write TSV
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines_written = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, acc in enumerate(accs):
            term_probs: Dict[str, float] = {}
            # Base probabilities
            for j, term in enumerate(classes):
                p = float(probs[i, j])
                if p >= min_prob:
                    term_probs[term] = max(term_probs.get(term, 0.0), p)
                    # Propagate to ancestors (max over children)
                    for anc in ancestors_map.get(term, ()): 
                        term_probs[anc] = max(term_probs.get(anc, 0.0), p)
            if not term_probs:
                continue
            # Enforce cap
            # Sort by probability descending and take top max_terms
            items = sorted(term_probs.items(), key=lambda kv: kv[1], reverse=True)[:max_terms]
            for term, p in items:
                f.write(f"{acc}\t{term}\t{format_prob(p)}\n")
                lines_written += 1
    print(f"Wrote {lines_written} lines to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict and write submission TSV")
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--obo-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--min-prob', type=float, default=0.001)
    parser.add_argument('--max-terms', type=int, default=1500)
    parser.add_argument('--max-test', type=int, default=None)
    args = parser.parse_args()

    predict_and_write(
        data_root=args.data_root,
        model_path=args.model_path,
        obo_path=args.obo_path,
        out_path=args.out_path,
        min_prob=args.min_prob,
        max_terms=args.max_terms,
        max_test=args.max_test,
    )
