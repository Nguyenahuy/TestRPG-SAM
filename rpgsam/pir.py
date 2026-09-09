"""Prior-guided Iterative SAM2 Refinement (paper Sec. 2.3)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from opsam.epe import edt_center


@dataclass
class PIRConfig:
    tau_cov: float = 0.9
    tau_iou: float = 0.8
    max_iter: int = 5
    cand_score: str = "sam_iou"     # sam_iou | prior_iou -- SAM's ambiguity candidates
    enabled: bool = True


@dataclass
class PIRTrace:
    rounds: int = 0
    points: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    covs: list = field(default_factory=list)
    ious: list = field(default_factory=list)
    negatives: int = 0
    stopped: str = "max_iter"


def _cov_iou(mask: np.ndarray, prior: np.ndarray) -> tuple[float, float]:
    inter = np.logical_and(mask, prior).sum()
    p = prior.sum()
    union = np.logical_or(mask, prior).sum()
    cov = float(inter / p) if p else 1.0
    iou = float(inter / union) if union else 0.0
    return cov, iou


def pir_segment(prior: np.ndarray, predict, cfg: PIRConfig):
    """``predict(points, labels, scorer) -> (mask, sam_iou)``.

    Prompts are cumulative: each iteration appends one point and re-runs SAM2 with the
    full history, exactly as the false-negative / false-positive correction describes.
    """
    trace = PIRTrace()
    if not prior.any():
        return np.zeros_like(prior), trace

    def scorer(cands):
        if cfg.cand_score == "sam_iou":
            return None
        return [_cov_iou(m, prior)[1] for m in cands]

    pts: list[list[int]] = []
    labels: list[int] = []

    def step(region: np.ndarray, label: int):
        pt = edt_center(region)
        if pt is None:
            return None
        pts.append([pt[0], pt[1]])
        labels.append(label)
        sc = None if cfg.cand_score == "sam_iou" else scorer
        mask, _ = predict(np.array(pts), np.array(labels), sc)
        trace.points.append([pt[0], pt[1]])
        trace.labels.append(label)
        trace.rounds += 1
        return mask

    mask = step(prior, 1)
    if mask is None:
        return np.zeros_like(prior), trace
    cov, iou = _cov_iou(mask, prior)
    trace.covs.append(cov)
    trace.ious.append(iou)
    history = [(iou, mask)]

    max_iter = 1 if not cfg.enabled else cfg.max_iter
    while trace.rounds < max_iter:
        if cov >= cfg.tau_cov and iou >= cfg.tau_iou:
            trace.stopped = "converged"
            break
        if cov < cfg.tau_cov:
            region, label = prior & ~mask, 1          # R_FN: expand
        else:
            region, label = mask & ~prior, 0          # R_FP: suppress
            trace.negatives += 1
        if not region.any():
            trace.stopped = "no_region"
            break
        nxt = step(region, label)
        if nxt is None:
            trace.stopped = "no_region"
            break
        mask = nxt
        cov, iou = _cov_iou(mask, prior)
        trace.covs.append(cov)
        trace.ious.append(iou)
        history.append((iou, mask))

    # "the mask with the highest IoU relative to the prior is selected"
    best = max(history, key=lambda x: x[0])[1]
    return best, trace
