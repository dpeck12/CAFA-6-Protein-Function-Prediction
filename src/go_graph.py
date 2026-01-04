from typing import Dict, Set, Iterable
from goatools.obo_parser import GODag


def load_go_dag(obo_path: str) -> GODag:
    return GODag(obo_path, optional_attrs={'relationship'})


def build_ancestors_map(go_ids: Iterable[str], dag: GODag) -> Dict[str, Set[str]]:
    """Return term -> set of ancestor GO IDs (excluding the term itself)."""
    out: Dict[str, Set[str]] = {}
    for go in go_ids:
        if go in dag:
            term = dag[go]
            # goatools provides 'get_all_parents()' which returns parents recursively
            ancestors = set(term.get_all_parents())
            out[go] = ancestors
        else:
            out[go] = set()
    return out
