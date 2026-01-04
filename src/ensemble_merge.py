import argparse
import os
from typing import Dict, Tuple, Set


def read_tsv(path: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            acc, term, prob_s = parts
            try:
                p = float(prob_s)
            except ValueError:
                continue
            out.setdefault(acc, {})[term] = max(out.get(acc, {}).get(term, 0.0), p)
    return out


def read_ia_terms(path: str | None) -> Set[str] | None:
    if path is None or not os.path.exists(path):
        return None
    allowed: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if parts:
                tok = parts[0]
                if tok.startswith("GO:"):
                    allowed.add(tok)
    return allowed


def format_prob(p: float) -> str:
    p = max(min(p, 1.0), 1e-12)
    s = f"{p:.3g}"
    if p == 1.0:
        s = "1.000"
    return s


def merge_and_write(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]], out_path: str, ia_terms: Set[str] | None, max_terms: int) -> int:
    # Combine using max per term
    merged: Dict[str, Dict[str, float]] = {}
    for acc in set(a) | set(b):
        terms: Dict[str, float] = {}
        ta = a.get(acc, {})
        tb = b.get(acc, {})
        for t, p in ta.items():
            if ia_terms is not None and t not in ia_terms:
                continue
            terms[t] = max(terms.get(t, 0.0), p)
        for t, p in tb.items():
            if ia_terms is not None and t not in ia_terms:
                continue
            terms[t] = max(terms.get(t, 0.0), p)
        # Cap per protein
        items = sorted(terms.items(), key=lambda kv: kv[1], reverse=True)[:max_terms]
        merged[acc] = dict(items)

    lines = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for acc, terms in merged.items():
            for t, p in sorted(terms.items(), key=lambda kv: kv[1], reverse=True):
                f.write(f"{acc}\t{t}\t{format_prob(p)}\n")
                lines += 1
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two CAFA6 TSV submissions by max probability")
    parser.add_argument("--tsv-a", type=str, required=True)
    parser.add_argument("--tsv-b", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--ia-path", type=str, default=None)
    parser.add_argument("--max-terms", type=int, default=1500)
    args = parser.parse_args()

    a = read_tsv(args.tsv_a)
    b = read_tsv(args.tsv_b)
    ia_terms = read_ia_terms(args.ia_path)
    wrote = merge_and_write(a, b, args.out_path, ia_terms, args.max_terms)
    print(f"Merged and wrote {wrote} lines to {args.out_path}")
