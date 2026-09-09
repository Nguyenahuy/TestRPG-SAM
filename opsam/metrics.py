"""Dice / IoU / ROC-AUC evaluation (paper Sec. 4.1)."""
from __future__ import annotations

import numpy as np


def iou_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if union == 0:
        return 1.0, 1.0
    return float(inter / union), float(2 * inter / denom) if denom else 0.0


def roc_auc(score: np.ndarray, gt: np.ndarray) -> float:
    """Rank-based AUC of a continuous prior against the binary ground truth."""
    y = gt.astype(bool).ravel()
    s = score.astype(np.float64).ravel()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties so the AUC is tie-corrected
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class Accumulator:
    def __init__(self):
        self.ious: list[float] = []
        self.dices: list[float] = []
        self.aucs: list[float] = []

    def add(self, iou: float, dice: float, auc: float | None = None):
        self.ious.append(iou)
        self.dices.append(dice)
        if auc is not None and not np.isnan(auc):
            self.aucs.append(auc)

    def summary(self) -> dict:
        out = {
            "n": len(self.ious),
            "IoU": 100 * float(np.mean(self.ious)) if self.ious else 0.0,
            "Dice": 100 * float(np.mean(self.dices)) if self.dices else 0.0,
        }
        if self.aucs:
            out["roc_auc"] = 100 * float(np.mean(self.aucs))
        return out
