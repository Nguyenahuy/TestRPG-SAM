"""Where the data and the SAM2 checkpoint live.

Self-contained: ``opsam`` is a sibling package inside this repo, so nothing is imported
from outside it. Both locations are environment-overridable, which is what makes the
same tree run unchanged on a laptop and on Kaggle:

    RPGSAM_DATA=/kaggle/input/kvasir-seg   RPGSAM_CKPT=/kaggle/working/sam2.1_hiera_large.pt
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = Path(os.environ.get("RPGSAM_DATA", ROOT / "data"))
SAM2_CKPT = Path(os.environ.get("RPGSAM_CKPT", ROOT / "checkpoints" / "sam2.1_hiera_large.pt"))
OUTPUTS = Path(os.environ.get("RPGSAM_OUTPUTS", ROOT / "outputs"))


def kvasir_root() -> Path:
    """DATA may point either at the Kvasir folder itself or at a parent holding it."""
    for cand in (DATA / "Kvasir-SEG", DATA / "kvasir-seg", DATA):
        if (cand / "images").is_dir() and (cand / "masks").is_dir():
            return cand
    return DATA / "Kvasir-SEG"
