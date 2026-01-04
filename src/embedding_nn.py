import argparse
import os
from typing import Dict, List, Tuple, Set

import joblib
import numpy as np
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors

from .data_ingest import read_fasta_sequences, read_train_terms, build_accession_labels, get_test_pairs
from .go_graph import load_go_dag, build_ancestors_map


AA = "ACDEFGHIKLMNPQRSTVWY"


def clean_sequence(seq: str) -> str:
    return "".join([c for c in seq.upper() if c in AA])


def tokenize_kmers(seq: str, k: int) -> List[str]:
    s = clean_sequence(seq)
    if len(s) < k:
        return []
    return [s[i : i + k] for i in range(len(s) - k + 1)]


def sequence_embedding(tokens: List[str], w2v: Word2Vec) -> np.ndarray:
    if not tokens:
        return np.zeros(w2v.vector_size, dtype=np.float32)
    vecs = []
    for t in tokens:
        if t in w2v.wv:
            vecs.append(w2v.wv[t])
    if not vecs:
        return np.zeros(w2v.vector_size, dtype=np.float32)
    return np.mean(np.asarray(vecs, dtype=np.float32), axis=0)


def aggregate_neighbor_labels(
    neighbors: List[Tuple[str, float]],
    labels_map: Dict[str, List[str]],
    ancestors_map: Dict[str, List[str]] | Dict[str, set] | None,
    allowed_terms: Set[str] | None,
    min_prob: float,
    max_terms: int,
) -> List[Tuple[str, float]]:
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
            if ancestors_map is not None:
                for anc in ancestors_map.get(t, ()):  # type: ignore
                    if allowed_terms is not None and anc not in allowed_terms:
                        continue
                    scores[anc] = max(scores.get(anc, 0.0), scores[t])
    items = [(t, s) for t, s in scores.items() if s >= min_prob]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items[:max_terms]


def format_prob(p: float) -> str:
    p = max(min(p, 1.0), 1e-12)
    s = f"{p:.3g}"
    if p == 1.0:
        s = "1.000"
    return s


def run_embedding_nn(
    data_root: str,
    out_path: str,
    k: int = 3,
    vector_size: int = 128,
    window: int = 10,
    n_neighbors: int = 100,
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

    # Tokenize corpus
    train_tokens = [tokenize_kmers(train_seqs[a], k) for a in train_accs]

    # Cache setup
    if cache_dir is None:
        cache_dir = os.path.join("models", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    w2v_path = os.path.join(cache_dir, f"w2v_k{k}_vs{vector_size}_win{window}_{len(train_accs)}.model")

    # Train or load Word2Vec
    if os.path.exists(w2v_path):
        print(f"Loading Word2Vec from {w2v_path}")
        w2v = Word2Vec.load(w2v_path)
    else:
        print(f"Training Word2Vec (k={k}, vector_size={vector_size}, window={window}) on {len(train_accs)} proteins...")
        w2v = Word2Vec(
            sentences=train_tokens,
            vector_size=vector_size,
            window=window,
            min_count=1,
            sg=1,
            workers=1,
            epochs=5,
        )
        w2v.save(w2v_path)

    # Build train embeddings
    print("Building train embeddings...")
    train_embeds = np.vstack([sequence_embedding(tokens, w2v) for tokens in train_tokens])

    # NearestNeighbors index
    print("Fitting NearestNeighbors (cosine, brute) on embeddings...")
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(train_embeds)

    # Prepare GO ancestors
    if skip_ancestors:
        anc_map = None
        print("Skipping ancestor propagation (dev mode)")
    else:
        classes = sorted(df_terms["term"].unique())
        dag = load_go_dag(os.path.join(data_root, "go-basic.obo"))
        anc_map = build_ancestors_map(classes, dag)

    # Load test sequences
    test_fasta = os.path.join(data_root, "testsuperset.fasta")
    test_accs, test_seqs = get_test_pairs(test_fasta)
    if max_test is not None and len(test_accs) > max_test:
        test_accs = test_accs[:max_test]
        test_seqs = test_seqs[:max_test]
        print(f"Scoring capped test set: {len(test_accs)} proteins")
    else:
        print(f"Scoring full test set: {len(test_accs)} proteins")

    # Taxonomy maps
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

    # Build test embeddings
    print("Building test embeddings...")
    test_tokens = [tokenize_kmers(s, k) for s in test_seqs]
    test_embeds = np.vstack([sequence_embedding(tokens, w2v) for tokens in test_tokens])

    # Query neighbors
    print("Querying nearest neighbors...")
    dists, indices = nn.kneighbors(test_embeds, return_distance=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, acc in enumerate(test_accs):
            idxs = indices[i]
            ds = dists[i]
            sims: List[float] = []
            t_tax = test_tax.get(acc)
            for k_idx, j in enumerate(idxs):
                sim = float(1.0 - ds[k_idx])
                if t_tax is not None:
                    tr_tax = train_tax.get(train_accs[j])
                    if tr_tax is not None and tr_tax == t_tax:
                        sim *= taxon_weight_same
                sims.append(sim)
            ws = np.exp(sim_beta * np.array(sims, dtype=np.float32))
            ws = ws / (ws.sum() + 1e-12)
            neighbors = [(train_accs[j], float(ws[k_idx])) for k_idx, j in enumerate(idxs)]
            preds = aggregate_neighbor_labels(neighbors, labels_map, anc_map, allowed_terms, min_prob=min_prob, max_terms=max_terms)
            for term, p in preds:
                f.write(f"{acc}\t{term}\t{format_prob(p)}\n")
                lines_written += 1
    print(f"Wrote {lines_written} lines to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Word2Vec k-mer embedding NN baseline for CAFA6")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--n-neighbors", type=int, default=100)
    parser.add_argument("--min-positive", type=int, default=20)
    parser.add_argument("--min-prob", type=float, default=0.001)
    parser.add_argument("--max-terms", type=int, default=1500)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--skip-ancestors", action="store_true")
    parser.add_argument("--ia-path", type=str, default=None)
    parser.add_argument("--taxon-weight-same", type=float, default=1.2)
    parser.add_argument("--sim-beta", type=float, default=8.0)
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    run_embedding_nn(
        data_root=args.data_root,
        out_path=args.out_path,
        k=args.k,
        vector_size=args.vector_size,
        window=args.window,
        n_neighbors=args.n_neighbors,
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
    )
