"""Correlation-based Prior Generation (paper Sec. 3.1, Eq. 1-2)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .features import minmax


def sinkhorn(A: torch.Tensor, n_iter: int = 3, eps: float = 1e-8) -> torch.Tensor:
    """Alternating row/column normalisation -> doubly stochastic (symmetric) matrix."""
    A = A.clamp_min(0)
    for _ in range(n_iter):
        A = A / (A.sum(dim=1, keepdim=True) + eps)
        A = A / (A.sum(dim=0, keepdim=True) + eps)
    # end row-stochastic so the refinement stays a convex recombination
    return A / (A.sum(dim=1, keepdim=True) + eps)


def cross_correlation(fq: torch.Tensor, fs: torch.Tensor, mode: str = "softmax") -> torch.Tensor:
    """S_corr in R^{hw x hw} between query and support patch embeddings (Eq. 1).

    ``softmax`` reads Eq. 1 as scaled dot-product attention: the 1/sqrt(D) scaling
    is what makes the logits well-conditioned, and the row-stochastic result turns
    ``S_corr . m_r`` into the attention mass landing on polyp patches, i.e. a
    probability in [0, 1] that the paper's fixed 0.5/0.7 thresholds can act on.
    """
    d = fq.shape[-1]
    if mode == "cosine":
        return F.normalize(fq, dim=-1) @ F.normalize(fs, dim=-1).T
    scores = fq @ fs.T / d**0.5
    return scores.softmax(dim=-1) if mode == "softmax" else scores


def cpg_prior(
    fq: torch.Tensor,
    fs: torch.Tensor,
    ms_r: torch.Tensor,
    attn_q: torch.Tensor | None = None,
    rho: int = 2,
    corr_mode: str = "softmax",
    sinkhorn_iters: int = 3,
    self_refine: bool = True,
    affinity_pow: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """p_ori = [S_self]^rho . S_corr . m_r^s  (Eq. 2).

    Returns ``(p_refined, p_init)``, both flat ``(hw,)`` in [0, 1].
    """
    s_corr = cross_correlation(fq, fs, corr_mode)
    p = s_corr @ ms_r
    # A row-stochastic S_corr already yields a calibrated probability; the other
    # modes produce an arbitrary scale that has to be stretched into [0, 1].
    p_init = p if corr_mode == "softmax" else minmax(p / ms_r.sum().clamp_min(1.0))

    if not self_refine or attn_q is None or rho <= 0:
        return p_init, p_init

    a = attn_q.clamp_min(0)
    if affinity_pow != 1.0:
        a = a.pow(affinity_pow)
    s_self = sinkhorn(a, sinkhorn_iters)
    # Sinkhorn leaves S_self row-stochastic, so each product is a convex
    # recombination of the prior and the [0, 1] range is preserved.
    v = p_init
    for _ in range(rho):
        v = s_self @ v
    return v, p_init
