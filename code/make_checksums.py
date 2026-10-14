#!/usr/bin/env python3
"""Regenerate CHECKSUMS.txt for every released data file.

Run from the repository root (or anywhere; paths resolve relative to the repo):

    python code/make_checksums.py

Covers corpus/, sources/, predictions/, results/, figures/, splits/ and robustness/.
Excludes CHECKSUMS.txt itself and compiled Python. Verify afterwards with:

    sha256sum -c CHECKSUMS.txt
"""
import hashlib
import os
import sys
from pathlib import Path

TRACKED = ("corpus/", "sources/", "predictions/", "results/",
           "figures/", "splits/", "robustness/")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    rows = []
    for dirpath, dirnames, filenames in os.walk("."):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        for name in sorted(filenames):
            rel = os.path.join(dirpath, name).replace("./", "", 1)
            if rel == "CHECKSUMS.txt" or rel.endswith(".pyc"):
                continue
            if rel.startswith(TRACKED):
                rows.append(f"{sha256(rel)}  {rel}")
    rows.sort(key=lambda r: r.split("  ", 1)[1])
    Path("CHECKSUMS.txt").write_text(
        "# SHA-256 of every released data file.\n"
        "# Verify with:  sha256sum -c CHECKSUMS.txt\n\n" + "\n".join(rows) + "\n")
    print(f"CHECKSUMS.txt written: {len(rows)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
