"""RPG-SAM adds AUC-PR (Table 2) to the OP-SAM metric set."""
from __future__ import annotations

import numpy as np

from opsam.metrics import Accumulator as _Acc
from opsam.metrics import iou_dice, roc_auc  # noqa: F401  (re-exported)


def pr_auc(score: np.ndarray, gt: np.ndarray) -> float:
    """Average precision of a continuous heatmap against the binary ground truth."""
    y = gt.astype(bool).ravel()
    s = score.astype(np.float64).ravel()
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == y.size:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / n_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


class Accumulator(_Acc):
    """Adds the AUC-PR channel."""

    def __init__(self):
        super().__init__()
        self.prs: list[float] = []

    def add(self, iou, dice, auc=None, pr=None):
        super().add(iou, dice, auc)
        if pr is not None and not np.isnan(pr):
            self.prs.append(pr)

    def summary(self) -> dict:
        out = super().summary()
        if self.prs:
            out["auc_pr"] = float(np.mean(self.prs))
        return out
