import argparse
import os
from typing import List
import joblib
import numpy as np
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from .data_ingest import read_fasta_sequences, read_train_terms, build_label_space, build_accession_labels, get_train_pairs
from .features import batch_kmer_matrix


def train_model(data_root: str, model_path: str, k: int = 3, min_positive: int = 1, max_iter: int = 1000, C: float = 1.0, max_train: int | None = None):
    # Paths
    fasta_path = os.path.join(data_root, "train_sequences.fasta")
    terms_path = os.path.join(data_root, "train_terms.tsv")

    # Ingest
    print("Loading training sequences...")
    seqs = read_fasta_sequences(fasta_path)
    print(f"Loaded {len(seqs)} sequences")

    print("Loading training labels...")
    df = read_train_terms(terms_path, min_positive=min_positive)
    print(f"Labels after filtering: {df['term'].nunique()} terms, {df['accession'].nunique()} proteins")

    labels_map = build_accession_labels(df)
    label_space = build_label_space(df)

    accs, seq_list = get_train_pairs(seqs, labels_map)
    if max_train is not None and len(accs) > max_train:
        accs = accs[:max_train]
        seq_list = seq_list[:max_train]
        print(f"Training on capped subset: {len(accs)} proteins (max_train={max_train})")
    else:
        print(f"Training on {len(accs)} proteins")

    # Features
    print("Building k-mer features...")
    X = batch_kmer_matrix(seq_list, k=k, sparse_output=True)

    # Targets
    print("Binarizing labels...")
    mlb = MultiLabelBinarizer(classes=label_space)
    Y = mlb.fit_transform([labels_map[a] for a in accs])

    # Class weights (per term) via balanced heuristic.
    # LogisticRegression can accept class_weight='balanced' for binary tasks; in OVR it's applied per estimator.
    base_clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight='balanced',
        solver='saga',
        penalty='l2',
        n_jobs=-1,
    )
    clf = OneVsRestClassifier(base_clf, n_jobs=-1)

    print("Training OneVsRest LogisticRegression...")
    clf.fit(X, Y)

    # Persist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({
        'model': clf,
        'mlb': mlb,
        'k': k,
    }, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline CAFA6 model")
    parser.add_argument('--data-root', type=str, required=True, help='Path to data/raw')
    parser.add_argument('--model-path', type=str, required=True, help='Output path for joblib model')
    parser.add_argument('--k', type=int, default=3, help='K-mer length')
    parser.add_argument('--min-positive', type=int, default=1, help='Minimum proteins per GO term to keep')
    parser.add_argument('--max-iter', type=int, default=1000, help='Max iterations for logistic regression')
    parser.add_argument('--C', type=float, default=1.0, help='Inverse regularization strength')
    parser.add_argument('--max-train', type=int, default=None, help='Cap number of training proteins to speed up training')
    args = parser.parse_args()

    train_model(
        data_root=args.data_root,
        model_path=args.model_path,
        k=args.k,
        min_positive=args.min_positive,
        max_iter=args.max_iter,
        C=args.C,
        max_train=args.max_train,
    )
