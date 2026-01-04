from typing import List
import numpy as np
from scipy import sparse

# Standard amino acids
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA)}


def kmer_vector(sequence: str, k: int = 3) -> np.ndarray:
    """Compute normalized k-mer counts for a sequence.
    Uses 20 standard amino acids; k-mers containing non-standard letters are skipped.
    Returns a dense vector of size 20**k.
    """
    n = len(sequence)
    if n < k:
        return np.zeros((20 ** k,), dtype=np.float32)
    # Precompute powers for index mapping
    base = 20
    powers = np.array([base ** p for p in range(k - 1, -1, -1)], dtype=np.int64)
    vec = np.zeros((base ** k,), dtype=np.float32)
    # Sliding window
    for i in range(n - k + 1):
        idxs = []
        valid = True
        for j in range(k):
            aa = sequence[i + j]
            if aa not in AA_INDEX:
                valid = False
                break
            idxs.append(AA_INDEX[aa])
        if not valid:
            continue
        index = int(np.dot(np.array(idxs, dtype=np.int64), powers))
        vec[index] += 1.0
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def batch_kmer_matrix(sequences: List[str], k: int = 3, sparse_output: bool = True):
    """Compute feature matrix for a batch of sequences.
    Returns CSR sparse matrix if `sparse_output` is True, else dense ndarray.
    """
    X = [kmer_vector(seq, k=k) for seq in sequences]
    X = np.stack(X, axis=0)
    if sparse_output:
        return sparse.csr_matrix(X)
    return X
