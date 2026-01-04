import os
from typing import Dict, List, Tuple, Set
import pandas as pd
from Bio import SeqIO

AMINO_ACIDS: Set[str] = set("ACDEFGHIKLMNPQRSTVWY")


def parse_uniprot_accession(header: str) -> str:
    """Extract UniProt accession from a FASTA header.
    Handles headers like 'sp|P12345|...' or plain 'P12345'.
    """
    h = header.strip()
    # Split at whitespace first; take the first token
    token = h.split()[0]
    parts = token.split("|")
    if len(parts) >= 2:
        return parts[1]
    return parts[0]


def read_fasta_sequences(fasta_path: str) -> Dict[str, str]:
    """Read sequences from FASTA into a dict accession -> sequence (uppercase).
    Filters to standard amino acid letters; non-standard letters retained but marked for skipping in features.
    """
    seqs: Dict[str, str] = {}
    with open(fasta_path, "r", encoding="utf-8") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            acc = parse_uniprot_accession(record.description)
            seq = str(record.seq).upper()
            seqs[acc] = seq
    return seqs


def read_train_terms(terms_path: str, min_positive: int = 1) -> pd.DataFrame:
    """Read training GO annotations.
    Returns a DataFrame with columns: accession, term, ontology.
    Optionally filters to terms with at least `min_positive` proteins.
    """
    df = pd.read_csv(terms_path, sep="\t", header=None, names=["accession", "term", "ontology"], dtype=str)
    if min_positive > 1:
        counts = df.groupby("term")["accession"].nunique()
        keep_terms = set(counts[counts >= min_positive].index)
        df = df[df["term"].isin(keep_terms)].reset_index(drop=True)
    return df


def build_label_space(df: pd.DataFrame) -> List[str]:
    """Return sorted list of unique GO terms present in the annotations."""
    return sorted(df["term"].unique())


def build_accession_labels(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Map accession -> list of GO terms (multi-label)."""
    grouped = df.groupby("accession")["term"].apply(list)
    return grouped.to_dict()


def get_train_pairs(seqs: Dict[str, str], labels: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """Return aligned lists of accessions and sequences where labels exist."""
    accs = []
    seq_list = []
    for acc, seq in seqs.items():
        if acc in labels:
            accs.append(acc)
            seq_list.append(seq)
    return accs, seq_list


def get_test_pairs(fasta_path: str) -> Tuple[List[str], List[str]]:
    seqs = read_fasta_sequences(fasta_path)
    accs = list(seqs.keys())
    seq_list = [seqs[a] for a in accs]
    return accs, seq_list
