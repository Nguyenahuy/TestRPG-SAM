"""Euclidean Prompt Evolution (paper Sec. 3.3, Algorithm 1)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


def edt_center(binary: np.ndarray):
    """Point furthest from the background -- the EDT centre used by EPE."""
    if not binary.any():
        return None
    dist = ndimage.distance_transform_edt(binary)
    y, x = np.unravel_index(int(dist.argmax()), dist.shape)
    return int(x), int(y)


def bbox_center(binary: np.ndarray):
    """Bounding-box centre of the non-zero region (the BBC baseline it is ablated against)."""
    if not binary.any():
        return None
    ys, xs = np.nonzero(binary)
    return int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2)


PICKERS = {"edt": edt_center, "bbox": bbox_center}


def coverage(mask: np.ndarray, prior_bin: np.ndarray) -> float:
    denom = prior_bin.sum()
    if denom == 0:
        return 1.0
    return float((mask & prior_bin).sum() / denom)


@dataclass
class EPEConfig:
    theta_t: float = 0.7      # tight prior threshold  (vartheta_t)
    theta_l: float = 0.5      # loose prior threshold  (vartheta_l)
    score_thresh: float = 0.85  # theta
    eta_ratio: float = 0.5    # negative-area threshold, as a fraction of |p_l|
    max_rounds: int = 5
    picker: str = "edt"


@dataclass
class EPETrace:
    rounds: int = 0
    points: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    covs: list = field(default_factory=list)
    ious: list = field(default_factory=list)
    negatives: int = 0


def epe_segment(prior: np.ndarray, predictor, cfg: EPEConfig):
    """Run the prompt/segment/evaluate loop; ``predictor(points, labels, scorer) -> (mask, iou)``."""
    pick = PICKERS[cfg.picker]
    p_t = prior > cfg.theta_t
    p_l = prior > cfg.theta_l
    if not p_l.any():
        p_l = prior >= prior.max()
    if not p_t.any():
        p_t = p_l

    eta = max(1.0, cfg.eta_ratio * float(p_l.sum()))
    t_area = max(int(p_t.sum()), 1)

    def scorer(cands):
        # coverage of the confident prior x precision w.r.t. the loose prior --
        # the same two signals the loop already evaluates
        return [
            (m & p_t).sum() / t_area * (m & p_l).sum() / max(int(m.sum()), 1)
            for m in cands
        ]

    pts: list[list[int]] = []
    labels: list[int] = []
    accepted: list[np.ndarray] = []
    trace = EPETrace()

    def euc_seg(prior_in: np.ndarray, label: int):
        pt = pick(prior_in)
        if pt is None:
            return None, 0.0, 0.0
        pts.append([pt[0], pt[1]])
        labels.append(label)
        mask, iou = predictor(np.array(pts), np.array(labels), scorer)
        trace.points.append([pt[0], pt[1]])
        trace.labels.append(label)
        trace.rounds += 1
        return mask, coverage(mask, p_t), iou

    mask, cov, iou = euc_seg(p_t, 1)
    if mask is None:
        return np.zeros_like(p_l), trace
    accepted.append(mask)
    trace.covs.append(cov)
    trace.ious.append(iou)
    done = cov >= cfg.score_thresh and iou >= cfg.score_thresh

    while not done and trace.rounds < cfg.max_rounds:
        prev = accepted[-1] if accepted else np.zeros_like(p_l)
        cand = (~prev) & (p_t if cov < cfg.score_thresh else p_l)
        if not cand.any():
            break

        mask, cov, iou = euc_seg(cand, 1)
        if mask is None:
            break

        spill = mask & (~p_l)
        if spill.sum() >= eta and trace.rounds < cfg.max_rounds:
            # Alg. 1 l.22-25: drop the noisy mask and add a negative prompt on the
            # spill. L is cumulative, so the offending positive prompt stays in it.
            trace.negatives += 1
            mask, cov, iou = euc_seg(spill, 0)
            if mask is None:
                break

        accepted.append(mask)
        trace.covs.append(cov)
        trace.ious.append(iou)
        done = cov >= cfg.score_thresh and iou >= cfg.score_thresh

    out = np.zeros_like(p_l)
    for m in accepted:
        out |= m
    return out, trace
