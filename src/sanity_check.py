"""A tiny sanity check that trains on synthetic data and predicts.
This does not require Kaggle data and ensures the pipeline runs.
"""
import os
import tempfile
import joblib

from .features import batch_kmer_matrix
from .data_ingest import build_label_space
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def run():
    # Synthetic sequences and labels
    accs = ["P00001", "P00002", "P00003", "P00004"]
    seqs = [
        "ACDEFGHIKLMNPQRSTVWYACD",  # diverse
        "AAAAAAAAAAAAAAAAAAAAAAA",    # A-rich
        "RRRRRRRRRRRRRRRRRRRRR",     # R-rich
        "KKKKKKKKKKKKKKKKKKKKK",     # K-rich
    ]
    labels_map = {
        "P00001": ["GO:0003674", "GO:0008150"],
        "P00002": ["GO:0003674"],
        "P00003": ["GO:0008150"],
        "P00004": ["GO:0005575"],
    }
    label_space = sorted({t for ts in labels_map.values() for t in ts})

    X = batch_kmer_matrix(seqs, k=3, sparse_output=True)
    mlb = MultiLabelBinarizer(classes=label_space)
    Y = mlb.fit_transform([labels_map[a] for a in accs])

    clf = OneVsRestClassifier(LogisticRegression(max_iter=500, class_weight='balanced', solver='saga'))
    clf.fit(X, Y)

    # Predict on same data
    probs = clf.predict_proba(X)
    print("Sanity check: predicted probs shape:", probs.shape)

    # Save and reload bundle
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.joblib")
        joblib.dump({'model': clf, 'mlb': mlb, 'k': 3}, path)
        bundle = joblib.load(path)
        assert bundle['mlb'].classes_.tolist() == label_space
        print("Bundle save/load works.")


if __name__ == "__main__":
    run()
