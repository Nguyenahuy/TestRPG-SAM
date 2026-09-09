"""Geometric Adaptive threshold Selection (paper Sec. 2.2, Eq. 5).

``geo_mode`` switches between Eq. 5 as written and the ablations of experiment C3:
the paper's score is ``weighted_solidity x min(1, A/A_ref)``, which (a) rewards convex
shapes and (b) has no penalty at all for a candidate *larger* than ``A_ref``.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

GEO_MODES = ("full", "no_solidity", "no_scale", "perim", "sym_scale", "perim_sym")


@dataclass
class GASConfig:
    tau_min: float = 0.4
    tau_max: float = 0.7
    stride: float = 0.05
    comp_ratio: float = 0.2     # keep components >= 20% of the largest one
    aref_mode: str = "support"  # support | frac | none  -- A_ref is left undefined
    aref_frac: float = 0.10     # used when aref_mode == "frac"
    fixed_tau: float | None = None   # ablation: skip the scan, binarise at one level
    geo_mode: str = "full"      # C3: which reading of Eq. 5 to score with


def thresholds(cfg: GASConfig) -> list[float]:
    n = int(round((cfg.tau_max - cfg.tau_min) / cfg.stride)) + 1
    return [round(cfg.tau_min + i * cfg.stride, 4) for i in range(n)]


def refine(binary: np.ndarray, comp_ratio: float):
    """Drop small components, fill internal holes, return (mask, [components])."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if n <= 1:
        return None, []
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i in range(len(areas)) if areas[i] >= comp_ratio * areas.max()]
    m = ndimage.binary_fill_holes(np.isin(lab, keep))
    n2, lab2 = cv2.connectedComponents(m.astype(np.uint8), 8)
    return m, [lab2 == i for i in range(1, n2)]


def solidity(comp: np.ndarray) -> float:
    """|C| / |Hull(C)| via the convex hull of the component's outer contour."""
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    pts = np.vstack(cnts)
    hull = cv2.contourArea(cv2.convexHull(pts))
    area = float(comp.sum())
    return area / hull if hull > 0 else 0.0


def perim_convexity(comp: np.ndarray) -> float:
    """P(Hull(C)) / P(C) -- convexity measured on the boundary instead of the area.

    The "less convexity-biased" shape term of experiment C3(iv). It separates the two
    things area solidity conflates. Measured on synthetic shapes (400x400):

        shape           area solidity   perimeter convexity
        disc                    1.002                 0.949
        flat ellipse            1.004                 0.952
        crescent                0.496                 0.782
        ragged disc             0.984                 0.914

    A sessile polyp is concave but *smoothly* concave -- the crescent row: area solidity
    halves, perimeter convexity barely moves. A ragged threshold artefact is the opposite:
    its notches cost little area, so solidity hardly reacts, while the inflated boundary
    shows up here. Convex shapes sit near 0.95 rather than 1.0 because a pixelated contour
    is longer than its hull polygon; only the ranking across thresholds matters.
    """
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    pts = np.vstack(cnts)
    p = sum(cv2.arcLength(c, True) for c in cnts)
    ph = cv2.arcLength(cv2.convexHull(pts), True)
    return float(min(1.0, ph / p)) if p > 0 else 0.0


def scale_term(total: float, a_ref: float | None, mode: str) -> float:
    """Eq. 5's Scale Consensus, and the two-sided repair of it.

    The paper writes ``min(1, A/A_ref)``: candidates below the reference area are damped,
    candidates above it are not penalised at all. Because the thresholded area falls
    monotonically as tau rises, that makes the whole score near-monotone in tau and GAS
    degenerates to "take the bottom of the scan band". ``sym`` restores the missing side.
    """
    if not a_ref:
        return 1.0
    if mode == "sym":
        return float(min(total / a_ref, a_ref / max(total, 1e-6)))
    return float(min(1.0, total / a_ref))


def geo_score(mask: np.ndarray, comps: list[np.ndarray], a_ref: float | None,
              geo_mode: str = "full") -> float:
    """Eq. 5 and its C3 variants."""
    total = float(mask.sum())
    if total <= 0:
        return 0.0

    shape_fn = perim_convexity if geo_mode in ("perim", "perim_sym") else solidity
    if geo_mode == "no_solidity":
        wsol = 1.0
    else:
        wsol = sum((float(c.sum()) / total) * shape_fn(c) for c in comps)

    if geo_mode == "no_scale":
        scale = 1.0
    else:
        scale = scale_term(total, a_ref,
                           "sym" if geo_mode in ("sym_scale", "perim_sym") else "min")
    return wsol * scale


def select(heat: np.ndarray, cfg: GASConfig, support_area_ratio: float = 0.0):
    """Scan the confidence range and keep the most polyp-shaped candidate."""
    hw = heat.size
    if cfg.aref_mode == "support":
        a_ref = support_area_ratio * hw
    elif cfg.aref_mode == "frac":
        a_ref = cfg.aref_frac * hw
    else:
        a_ref = None

    taus = [cfg.fixed_tau] if cfg.fixed_tau is not None else thresholds(cfg)
    best, best_s, best_tau, scan = None, -1.0, None, []
    for t in taus:
        m, comps = refine(heat > t, cfg.comp_ratio)
        s = geo_score(m, comps, a_ref, cfg.geo_mode) if m is not None else 0.0
        scan.append({"tau": t, "score": round(float(s), 4),
                     "area": int(m.sum()) if m is not None else 0})
        if m is not None and s > best_s:
            best, best_s, best_tau = m, s, t

    if best is None:                     # nothing survived: fall back to the peak response
        best = heat >= heat.max()
        best_tau, best_s = float(heat.max()), 0.0
    return best, {"tau": best_tau, "S_geo": round(float(best_s), 4), "scan": scan}
