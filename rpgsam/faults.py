"""Controlled degradations of the support pair, for experiment A1(iii).

The point of A1 is to make `M_prior` wrong *in a specific, measurable way* and then ask
whether PIR still recovers. Corrupting the support image/mask is the cleanest lever:
it degrades RWPM at its input while leaving GAS and PIR byte-identical.

Every corruption is deterministic given ``seed`` so a run can be replayed exactly.
"""
from __future__ import annotations

import cv2
import numpy as np

FAULTS = ("none", "specular", "blur", "noise", "erode", "dilate", "shift")


def _specular(img: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Paste bright saturated blobs -- the reflection artefact endoscopy is full of.

    RWPM's Contrast Factor is meant to down-weight prototypes that fire on exactly this,
    so it is the corruption the reliability term claims to defend against.
    """
    out = img.astype(np.float32).copy()
    h, w = img.shape[:2]
    n = max(1, int(12 * strength))
    for _ in range(n):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        r = int(rng.integers(int(0.02 * h), int(0.07 * h) + 1))
        blob = np.zeros((h, w), np.float32)
        cv2.circle(blob, (int(cx), int(cy)), r, 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), r / 2.0)
        out += 255.0 * strength * blob[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def corrupt_support(image: np.ndarray, mask: np.ndarray, fault: str,
                    strength: float = 1.0, seed: int = 0):
    """Return a degraded ``(image, mask)`` pair. ``strength`` scales the damage."""
    rng = np.random.default_rng(seed)
    img, m = image.copy(), mask.copy()

    if fault == "none":
        return img, m
    if fault == "specular":
        img = _specular(img, rng, 0.35 * strength)
    elif fault == "blur":
        k = max(3, int(2 * round(6 * strength) + 1))
        img = cv2.GaussianBlur(img, (k, k), 0)
    elif fault == "noise":
        img = np.clip(img.astype(np.float32) + rng.normal(0, 25 * strength, img.shape),
                      0, 255).astype(np.uint8)
    elif fault in ("erode", "dilate"):
        # annotation noise: the support mask is right about *where* but wrong about *how big*
        r = max(1, int(0.06 * strength * np.sqrt(mask.sum())))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        m = (cv2.erode(m, k) if fault == "erode" else cv2.dilate(m, k))
    elif fault == "shift":
        # the mask is displaced: prototypes are mined from the wrong pixels
        d = int(0.08 * strength * min(mask.shape))
        M = np.float32([[1, 0, d], [0, 1, d]])
        m = cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST)
    else:
        raise ValueError(f"unknown fault '{fault}' (choose from {FAULTS})")

    if m.sum() == 0:                      # never hand RWPM an empty support mask
        m = mask.copy()
    return img, m
