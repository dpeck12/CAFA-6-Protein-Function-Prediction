import argparse
import os
from typing import Dict, List, Tuple, Set

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .data_ingest import read_fasta_sequences, read_train_terms, build_accession_labels, get_test_pairs
from .go_graph import load_go_dag, build_ancestors_map


AA = "ACDEFGHIKLMNPQRSTVWY"


def clean_sequence(seq: str) -> str:
    return "".join([c for c in seq.upper() if c in AA])


def build_vocab_ngrams(min_n: int = 2, max_n: int = 2) -> List[str]:
    toks: List[str] = []
    for n in range(min_n, max_n + 1):
        if n == 2:
            toks.extend([a + b for a in AA for b in AA])
        elif n == 3:
            toks.extend([a + b + c for a in AA for b in AA for c in AA])
        else:
            raise ValueError("n must be 2 or 3")
    return toks


def tfidf_features(seqs: List[str], min_n: int = 2, max_n: int = 2) -> Tuple[TfidfVectorizer, np.ndarray]:
    vocab = build_vocab_ngrams(min_n, max_n)
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(min_n, max_n),
        vocabulary=vocab,
        dtype=np.float32,
        lowercase=False,
    )
    X = vec.fit_transform(seqs)
    return vec, X


def transform_features(vec: TfidfVectorizer, seqs: List[str]) -> np.ndarray:
    return vec.transform(seqs)


def aggregate_neighbor_labels(
    neighbors: List[Tuple[str, float]],
    labels_map: Dict[str, List[str]],
    ancestors_map: Dict[str, List[str]] | Dict[str, set] | None,
    allowed_terms: Set[str] | None,
    min_prob: float,
    max_terms: int,
) -> List[Tuple[str, float]]:
    # Accumulate weighted scores per term
    scores: Dict[str, float] = {}
    total_weight = sum(w for _, w in neighbors)
    if total_weight <= 0:
        return []
    for acc, w in neighbors:
        terms = labels_map.get(acc, [])
        if not terms:
            continue
        weight = w / total_weight
        for t in terms:
            if allowed_terms is not None and t not in allowed_terms:
                continue
            scores[t] = scores.get(t, 0.0) + weight
            # Propagate to ancestors (max over children)
            if ancestors_map is not None:
                for anc in ancestors_map.get(t, ()):  # type: ignore
                    if allowed_terms is not None and anc not in allowed_terms:
                        continue
                    scores[anc] = max(scores.get(anc, 0.0), scores[t])
    # Filter and cap
    items = [(t, s) for t, s in scores.items() if s >= min_prob]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items[:max_terms]


def format_prob(p: float) -> str:
    p = max(min(p, 1.0), 1e-12)
    s = f"{p:.3g}"
    if p == 1.0:
        s = "1.000"
    return s


def run_nn_baseline(
    data_root: str,
    out_path: str,
    ngram_n: int = 2,
    ngram_max: int | None = None,
    n_neighbors: int = 25,
    min_positive: int = 20,
    min_prob: float = 0.001,
    max_terms: int = 1500,
    max_train: int | None = None,
    max_test: int | None = None,
    skip_ancestors: bool = False,
    ia_path: str | None = None,
    taxon_weight_same: float = 1.2,
    sim_beta: float = 8.0,
    cache_dir: str | None = None,
    batch_size: int = 500,
):
    # Load training sequences and labels
    train_fasta = os.path.join(data_root, "train_sequences.fasta")
    terms_path = os.path.join(data_root, "train_terms.tsv")
    train_seqs = read_fasta_sequences(train_fasta)
    df_terms = read_train_terms(terms_path, min_positive=min_positive)
    allowed_terms: Set[str] | None = None
    if ia_path is None:
        default_ia = os.path.join(data_root, "IA.tsv")
        if os.path.exists(default_ia):
            ia_path = default_ia
    if ia_path is not None and os.path.exists(ia_path):
        print(f"Filtering to IA terms from {ia_path}")
        allowed_terms = set()
        with open(ia_path, "r", encoding="utf-8") as iaf:
            for line in iaf:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if parts:
                    tok = parts[0]
                    if tok.startswith("GO:"):
                        allowed_terms.add(tok)
        df_terms = df_terms[df_terms["term"].isin(allowed_terms)]
    labels_map = build_accession_labels(df_terms)
    train_accs = [acc for acc in train_seqs.keys() if acc in labels_map]
    if max_train is not None and len(train_accs) > max_train:
        train_accs = train_accs[:max_train]
    train_corpus = [clean_sequence(train_seqs[a]) for a in train_accs]

    # Build TF-IDF features on training corpus
    if ngram_max is None:
        ngram_max = ngram_n
    # Optional cache for TF-IDF features
    if cache_dir is None:
        cache_dir = os.path.join("models", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    tfidf_cache_path = os.path.join(cache_dir, f"tfidf_{ngram_n}-{ngram_max}_{len(train_accs)}.joblib")
    if os.path.exists(tfidf_cache_path):
        print(f"Loading cached TF-IDF from {tfidf_cache_path}")
        cache_obj = joblib.load(tfidf_cache_path)
        vec = cache_obj["vec"]
        X_train = cache_obj["X_train"]
        if cache_obj.get("train_accs") != train_accs:
            print("Cached train set mismatch; rebuilding TF-IDF")
            vec, X_train = tfidf_features(train_corpus, min_n=ngram_n, max_n=ngram_max)
            joblib.dump({"vec": vec, "X_train": X_train, "train_accs": train_accs}, tfidf_cache_path)
    else:
        print(f"Building TF-IDF {ngram_n}-{ngram_max}-gram features for {len(train_corpus)} training proteins...")
        vec, X_train = tfidf_features(train_corpus, min_n=ngram_n, max_n=ngram_max)
        joblib.dump({"vec": vec, "X_train": X_train, "train_accs": train_accs}, tfidf_cache_path)

    # NearestNeighbors index
    print("Fitting NearestNeighbors (cosine, brute)...")
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(X_train)

    # Prepare GO ancestors (optional for quicker iteration)
    if skip_ancestors:
        anc_map = None
        print("Skipping ancestor propagation (dev mode)")
    else:
        classes = sorted(df_terms["term"].unique())
        anc_cache_path = os.path.join(cache_dir, f"anc_map_{len(classes)}.joblib")
        if os.path.exists(anc_cache_path):
            print(f"Loading cached GO ancestors from {anc_cache_path}")
            anc_map = joblib.load(anc_cache_path)
        else:
            dag = load_go_dag(os.path.join(data_root, "go-basic.obo"))
            anc_map = build_ancestors_map(classes, dag)
            joblib.dump(anc_map, anc_cache_path)

    # Load test sequences
    test_fasta = os.path.join(data_root, "testsuperset.fasta")
    test_accs, test_seqs = get_test_pairs(test_fasta)

    # Optional taxonomy-aware neighbor weighting
    train_tax_path = os.path.join(data_root, "train_taxonomy.tsv")
    test_tax_path = os.path.join(data_root, "testsuperset-taxon-list.tsv")
    train_tax: Dict[str, str] = {}
    test_tax: Dict[str, str] = {}
    if os.path.exists(train_tax_path):
        with open(train_tax_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    train_tax[parts[0]] = parts[1]
    if os.path.exists(test_tax_path):
        with open(test_tax_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    test_tax[parts[0]] = parts[1]
    if max_test is not None and len(test_accs) > max_test:
        test_accs = test_accs[:max_test]
        test_seqs = test_seqs[:max_test]
        print(f"Scoring capped test set: {len(test_accs)} proteins")
    else:
        print(f"Scoring full test set: {len(test_accs)} proteins")

    # Transform test features
    X_test = transform_features(vec, [clean_sequence(s) for s in test_seqs])

    # Query neighbors in batches
    print("Querying nearest neighbors in batches...")
    n_test = X_test.shape[0]
    dists_list: List[np.ndarray] = []
    indices_list: List[np.ndarray] = []
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        d, idx = nn.kneighbors(X_test[start:end], return_distance=True)
        dists_list.append(d)
        indices_list.append(idx)
    dists = np.vstack(dists_list)
    indices = np.vstack(indices_list)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, acc in enumerate(test_accs):
            idxs = indices[i]
            ds = dists[i]
            # Convert cosine distance to similarity, apply taxonomy weighting, then softmax
            sims: List[float] = []
            t_tax = test_tax.get(acc)
            for k, j in enumerate(idxs):
                sim = float(1.0 - ds[k])
                if t_tax is not None:
                    tr_tax = train_tax.get(train_accs[j])
                    if tr_tax is not None and tr_tax == t_tax:
                        sim *= taxon_weight_same
                sims.append(sim)
            # Softmax weights
            ws = np.exp(sim_beta * np.array(sims, dtype=np.float32))
            ws = ws / (ws.sum() + 1e-12)
            neighbors = [(train_accs[j], float(ws[k])) for k, j in enumerate(idxs)]
            preds = aggregate_neighbor_labels(neighbors, labels_map, anc_map, allowed_terms, min_prob=min_prob, max_terms=max_terms)
            for term, p in preds:
                f.write(f"{acc}\t{term}\t{format_prob(p)}\n")
                lines_written += 1
    print(f"Wrote {lines_written} lines to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nearest-neighbor TF-IDF baseline for CAFA6")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--ngram-n", type=int, default=2)
    parser.add_argument("--n-neighbors", type=int, default=25)
    parser.add_argument("--min-positive", type=int, default=20)
    parser.add_argument("--min-prob", type=float, default=0.005)
    parser.add_argument("--max-terms", type=int, default=1500)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--skip-ancestors", action="store_true")
    parser.add_argument("--ia-path", type=str, default=None)
    parser.add_argument("--taxon-weight-same", type=float, default=1.2)
    parser.add_argument("--ngram-max", type=int, default=None)
    parser.add_argument("--sim-beta", type=float, default=8.0)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    run_nn_baseline(
        data_root=args.data_root,
        out_path=args.out_path,
        ngram_n=args.ngram_n,
        n_neighbors=args.n_neighbors,
        ngram_max=args.ngram_max,
        min_positive=args.min_positive,
        min_prob=args.min_prob,
        max_terms=args.max_terms,
        max_train=args.max_train,
        max_test=args.max_test,
        skip_ancestors=args.skip_ancestors,
        ia_path=args.ia_path,
        taxon_weight_same=args.taxon_weight_same,
        sim_beta=args.sim_beta,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
    )
