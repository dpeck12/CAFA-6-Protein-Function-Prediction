import argparse
import os
import time
from typing import Optional

from .ensemble_merge import read_tsv, read_ia_terms, merge_and_write


def wait_for_update(path: str, poll_seconds: int, min_size_increase: int = 1) -> None:
    last_mtime: Optional[float] = None
    last_size: Optional[int] = None
    while True:
        try:
            stat = os.stat(path)
            mtime = stat.st_mtime
            size = stat.st_size
            if last_mtime is None:
                last_mtime = mtime
                last_size = size
            else:
                if (mtime > last_mtime) and (size is not None and last_size is not None and size - last_size >= min_size_increase):
                    # updated
                    return
            time.sleep(poll_seconds)
        except FileNotFoundError:
            time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description="Wait for NN TSV update, then ensemble with logistic TSV and write submission")
    parser.add_argument("--nn-tsv", type=str, required=True)
    parser.add_argument("--logreg-tsv", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--ia-path", type=str, default=None)
    parser.add_argument("--max-terms", type=int, default=1500)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    print(f"Waiting for update to {args.nn_tsv}...")
    wait_for_update(args.nn_tsv, args.poll_seconds)
    print("Detected update. Merging...")

    a = read_tsv(args.nn_tsv)
    b = read_tsv(args.logreg_tsv)
    ia_terms = read_ia_terms(args.ia_path)
    wrote = merge_and_write(a, b, args.out_path, ia_terms, args.max_terms)
    print(f"Merged and wrote {wrote} lines to {args.out_path}")


if __name__ == "__main__":
    main()
