"""Dataset adapters for the five polyp benchmarks used in the paper."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass
class Sample:
    name: str
    image_path: str
    mask_path: str

    def load(self):
        img = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
        m = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        if m.shape != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return img, (m > 127).astype(np.uint8)


def _pair_dirs(img_dir: Path, mask_dir: Path) -> list[Sample]:
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    out = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in IMG_EXT and p.stem in masks:
            out.append(Sample(p.stem, str(p), str(masks[p.stem])))
    return out


# Directory layouts differ per benchmark; each entry is (image_subdir, mask_subdir).
LAYOUTS = {
    "kvasir": ("images", "masks"),
    "clinicdb": ("images", "masks"),
    "colondb": ("images", "masks"),
    "piccolo": ("images", "masks"),
    "polypgen": ("images", "masks"),
}


def load_dataset(name: str, root: str) -> list[Sample]:
    """Load a benchmark, falling back to a recursive images/masks search."""
    root = Path(root)
    img_sub, mask_sub = LAYOUTS.get(name, ("images", "masks"))
    if (root / img_sub).is_dir() and (root / mask_sub).is_dir():
        return _pair_dirs(root / img_sub, root / mask_sub)

    samples: list[Sample] = []
    for dirpath, dirnames, _ in os.walk(root):
        d = Path(dirpath)
        if img_sub in dirnames and mask_sub in dirnames:
            samples += _pair_dirs(d / img_sub, d / mask_sub)
    if not samples:
        raise FileNotFoundError(f"no image/mask pairs found for '{name}' under {root}")
    return samples


def polyp_coverage(sample: Sample) -> float:
    m = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
    return float((m > 127).mean())


def kvasir_hard(samples: list[Sample], low=0.03, high=0.50) -> list[Sample]:
    """Kvasir-H: extreme-size polyps, coverage <= 3% or >= 50% (paper Sec. 4.1)."""
    return [s for s in samples if (c := polyp_coverage(s)) <= low or c >= high]
