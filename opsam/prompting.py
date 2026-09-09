"""Prompting strategies compared in Table 6, plus the Oracle protocol."""
from __future__ import annotations

import numpy as np

from .epe import EPEConfig, edt_center, epe_segment


def _pred(prompter, pts, labels, box=None, scorer=None):
    mask, iou, _ = prompter.prompt(np.array(pts, np.float32), np.array(labels, np.int32),
                                   box=box, scorer=scorer)
    return mask, iou


def prior_scorer(prior: np.ndarray, theta_t: float, theta_l: float):
    """Rank SAM's ambiguity candidates by agreement with the prior.

    Shared by every prompting strategy so Table 6 isolates the prompting choice.
    """
    p_t, p_l = prior > theta_t, prior > theta_l
    if not p_l.any():
        p_l = prior >= prior.max()
    if not p_t.any():
        p_t = p_l
    t_area = max(int(p_t.sum()), 1)

    def score(cands):
        return [(m & p_t).sum() / t_area * (m & p_l).sum() / max(int(m.sum()), 1) for m in cands]

    return score


def top_first_last(prior: np.ndarray, prompter, thresh=0.5, scorer=None):
    """PerSAM-style: highest-response point as positive, lowest as negative."""
    flat = prior.ravel()
    hi = np.unravel_index(int(flat.argmax()), prior.shape)
    lo = np.unravel_index(int(flat.argmin()), prior.shape)
    pts = [[hi[1], hi[0]], [lo[1], lo[0]]]
    return _pred(prompter, pts, [1, 0], scorer=scorer)[0]


def top_center_bbox(prior: np.ndarray, prompter, thresh=0.5, scorer=None):
    """ProtoSAM-style: top point + bbox centre + the bounding box itself."""
    binary = prior > thresh
    if not binary.any():
        binary = prior >= prior.max()
    ys, xs = np.nonzero(binary)
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)
    hi = np.unravel_index(int(prior.ravel().argmax()), prior.shape)
    cx, cy = int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2)
    return _pred(prompter, [[hi[1], hi[0]], [cx, cy]], [1, 1], box=box, scorer=scorer)[0]


def random_points(prior: np.ndarray, prompter, k=3, thresh=0.5, rng=None, scorer=None):
    """Sample k positive points from the thresholded prior (Table 3 non-EPE row)."""
    rng = rng or np.random.default_rng(0)
    binary = prior > thresh
    if not binary.any():
        binary = prior >= prior.max()
    ys, xs = np.nonzero(binary)
    idx = rng.choice(len(ys), size=min(k, len(ys)), replace=False)
    pts = [[int(xs[i]), int(ys[i])] for i in idx]
    return _pred(prompter, pts, [1] * len(pts), scorer=scorer)[0]


def oracle_points(gt: np.ndarray, prompter, k=3, rng=None):
    """Oracle: k prompt points drawn at random from the test ground truth."""
    rng = rng or np.random.default_rng(0)
    ys, xs = np.nonzero(gt)
    if len(ys) == 0:
        return np.zeros_like(gt, bool)
    idx = rng.choice(len(ys), size=min(k, len(ys)), replace=False)
    pts = [[int(xs[i]), int(ys[i])] for i in idx]
    return _pred(prompter, pts, [1] * len(pts))[0]


def run_prompting(strategy: str, prior: np.ndarray, prompter, cfg: EPEConfig, rng=None):
    """Dispatch a prompting strategy; returns ``(mask, trace_or_None)``."""
    sc = prior_scorer(prior, cfg.theta_t, cfg.theta_l)
    if strategy in ("epe", "epe_bbox"):
        c = EPEConfig(**{**cfg.__dict__, "picker": "edt" if strategy == "epe" else "bbox"})
        return epe_segment(prior, lambda p, l, s: _pred(prompter, p, l, scorer=s), c)
    if strategy == "top_first_last":
        return top_first_last(prior, prompter, cfg.theta_l, sc), None
    if strategy == "top_center_bbox":
        return top_center_bbox(prior, prompter, cfg.theta_l, sc), None
    if strategy == "random3":
        return random_points(prior, prompter, 3, cfg.theta_l, rng, sc), None
    raise ValueError(f"unknown prompting strategy: {strategy}")
