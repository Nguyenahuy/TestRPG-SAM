"""Convert ``notebooks/kaggle_rpgsam_gaps.py`` into the .ipynb Kaggle wants.

Keeping the source as a .py means the notebook diffs like code instead of like JSON.
Cells are split on ``# %%``; a ``# %% [markdown]`` block becomes a markdown cell with the
leading ``# `` stripped.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "kaggle_rpgsam_gaps.py"
DST = ROOT / "notebooks" / "kaggle_rpgsam_gaps.ipynb"


def cells(text: str):
    out, kind, buf = [], "code", []

    def flush():
        body = "\n".join(buf).strip("\n")
        if body:
            out.append((kind, body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    return out


def main():
    text = SRC.read_text(encoding="utf-8")
    # drop the module docstring: it documents the .py, not the notebook
    if text.startswith('"""'):
        text = text.split('"""', 2)[2]

    nb_cells = []
    for kind, body in cells(text):
        if kind == "markdown":
            body = "\n".join(ln[2:] if ln.startswith("# ") else ln.lstrip("#")
                             for ln in body.splitlines())
            nb_cells.append({"cell_type": "markdown", "metadata": {},
                             "source": body.splitlines(keepends=True)})
        else:
            nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                             "outputs": [], "source": body.splitlines(keepends=True)})

    nb = {"cells": nb_cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"},
                       "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    DST.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"[wrote] {DST}  ({len(nb_cells)} cells)")


if __name__ == "__main__":
    main()
