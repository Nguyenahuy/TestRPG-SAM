"""Reliability-Weighted Prototype Mining (paper Sec. 2.1, Eq. 1-4)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import slic


@dataclass
class RWPMConfig:
    n_segments: int = 10        # SLIC K
    compactness: float = 20.0   # SLIC m
    top_n: int = 10             # reverse-purity neighbourhood (unspecified in the paper)
    sim: str = "cosine"         # cosine | dot   -- Eq. 3 writes a bare dot product
    fg_scale: str = "kfg"       # kfg | one | inv -- reading of the K_fg factor in Eq. 3
    bg_suppress: bool = True    # ablation: drop the negative anchors
    reliability: bool = True    # ablation: W_k = 1 for every prototype
    softmax_temp: float = 1.0
    diffusion: str = "row"      # row | sinkhorn | none
    diffusion_iters: int = 2    # N applications of the affinity
    affinity_pow: float = 4.0   # measured; see README decision 2
    norm: str = "minmax"        # minmax | logit_minmax | none  -- see heatmap()


def _sim(a: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
    """Similarity of every row of ``a`` against ``b`` (a prototype or a second bank)."""
    if mode == "cosine":
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
    return a @ b.transpose(-1, -2)


def slic_grid(image: np.ndarray, grid: int, cfg: RWPMConfig) -> np.ndarray:
    """SLIC superpixels of the support image, pooled onto the DINOv2 patch grid."""
    img = cv2.resize(image, (grid * 14, grid * 14), interpolation=cv2.INTER_AREA)
    lab = slic(img, n_segments=cfg.n_segments, compactness=cfg.compactness, start_label=0)
    lab = cv2.resize(lab.astype(np.int32), (grid, grid), interpolation=cv2.INTER_NEAREST)
    return lab.reshape(-1)


def mask_grid(mask: np.ndarray, grid: int) -> np.ndarray:
    """Support mask on the patch grid (a patch is foreground if >=50% of it is)."""
    m = cv2.resize(mask.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
    return (m.reshape(-1) > 0.5)


def build_prototypes(fs: torch.Tensor, labels: np.ndarray, fg: np.ndarray):
    """Split every superpixel by m_s and average its features -> P_fg, P_bg."""
    dev = fs.device
    fg_t = torch.from_numpy(fg).to(dev)
    p_fg, p_bg = [], []
    for k in np.unique(labels):
        sel = torch.from_numpy(labels == k).to(dev)
        f = sel & fg_t
        b = sel & ~fg_t
        if int(f.sum()) > 0:
            p_fg.append(fs[f].mean(0))
        if int(b.sum()) > 0:
            p_bg.append(fs[b].mean(0))
    if not p_fg:                       # degenerate support: fall back to a global prototype
        p_fg = [fs[fg_t].mean(0)] if int(fg_t.sum()) else [fs.mean(0)]
    if not p_bg:
        p_bg = [fs.mean(0)]
    return torch.stack(p_fg), torch.stack(p_bg)


def contrast_factor(p_fg: torch.Tensor, fs: torch.Tensor, fg: np.ndarray, cfg: RWPMConfig):
    """Eq. 1: standardised gap between a prototype's support-foreground and
    support-background responses. Prototypes that also light up the background
    (reflections, mucus) score near zero."""
    g = _sim(fs, p_fg, cfg.sim)                      # (hw, K_fg)
    fg_t = torch.from_numpy(fg).to(g.device)
    num = g[fg_t].mean(0) - g[~fg_t].mean(0)
    return torch.relu(num / (g.std(dim=0) + 1e-8))


def reverse_purity(p_fg: torch.Tensor, fq: torch.Tensor, fs: torch.Tensor,
                   fg: np.ndarray, cfg: RWPMConfig):
    """Eq. 2: match a prototype into the query, average its top-n hits into a proxy,
    project the proxy back and measure how much of the return lands inside m_s."""
    n = min(cfg.top_n, fq.shape[0])
    g_q = _sim(fq, p_fg, cfg.sim)                    # (hw, K_fg)
    idx = g_q.topk(n, dim=0).indices                 # (n, K_fg)
    proxy = fq[idx].mean(0)                          # (K_fg, D)
    g_s = _sim(fs, proxy, cfg.sim)                   # (hw, K_fg)
    idx_s = g_s.topk(n, dim=0).indices
    fg_t = torch.from_numpy(fg).to(fq.device).float()
    purity = fg_t[idx_s].mean(0)                     # (K_fg,)
    p0 = fg_t.mean()
    return torch.relu((purity - p0) / (1.0 - p0 + 1e-8))


def self_affinity(fq: torch.Tensor, cfg: RWPMConfig) -> torch.Tensor:
    """Row-stochastic query self-affinity used by the diffusion of Eq. 4.

    Eq. 4 writes the raw Gram matrix ``(f_q f_q^T)^k``; left unnormalised its spectral
    radius blows the heatmap up, so the affinity is clamped, sharpened and
    row-normalised, which keeps each application a convex recombination.
    """
    a = torch.relu(_sim(fq, fq, "cosine"))
    if cfg.affinity_pow != 1.0:
        a = a.pow(cfg.affinity_pow)
    if cfg.diffusion == "sinkhorn":
        from opsam.cpg import sinkhorn
        return sinkhorn(a, 3)
    return a / (a.sum(dim=1, keepdim=True) + 1e-8)


def minmax(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


class SupportPrototypes:
    """Query-independent half of RWPM: prototypes plus their contrast factors."""

    def __init__(self, ex, image: np.ndarray, mask: np.ndarray, cfg: RWPMConfig):
        from PIL import Image

        self.cfg = cfg
        self.grid = ex.grid
        self.fs, _ = ex.forward(Image.fromarray(image), want_attn=False)
        self.labels = slic_grid(image, ex.grid, cfg)
        self.fg = mask_grid(mask, ex.grid)
        self.p_fg, self.p_bg = build_prototypes(self.fs, self.labels, self.fg)
        self.contrast = contrast_factor(self.p_fg, self.fs, self.fg, cfg)
        self.area_ratio = float(mask.mean())

    def weights(self, fq: torch.Tensor):
        """W_k = C_k . R_k, recomputed per query through the reverse-purity term."""
        if not self.cfg.reliability:
            return torch.ones_like(self.contrast), None, None
        r = reverse_purity(self.p_fg, fq, self.fs, self.fg, self.cfg)
        return self.contrast * r, self.contrast, r


def heatmap(sup: SupportPrototypes, fq: torch.Tensor, cfg: RWPMConfig):
    """Eq. 3 + Eq. 4: weighted foreground response minus background anchors,
    softmaxed over the query patches and then diffused."""
    w, c, r = sup.weights(fq)
    s_fg = _sim(fq, sup.p_fg, cfg.sim)               # (hw, K_fg)
    pos = (s_fg * w[None]).sum(1)
    k_fg = float(sup.p_fg.shape[0])
    scale = {"kfg": k_fg, "one": 1.0, "inv": 1.0 / k_fg}[cfg.fg_scale]
    logits = scale * pos
    if cfg.bg_suppress:
        logits = logits - _sim(fq, sup.p_bg, cfg.sim).sum(1)

    # Eq. 3 applies a Softmax over the query patches. A softmax of 1600 logits is far
    # too peaked for the paper's own [0.4, 0.7] scan band, so "logit_minmax" (rescale the
    # raw response instead) is kept as the alternative reading and measured against it.
    if cfg.norm == "logit_minmax":
        h = minmax(logits)
    else:
        h = torch.softmax(logits / cfg.softmax_temp, dim=0)
    if cfg.diffusion != "none" and cfg.diffusion_iters > 0:
        a = self_affinity(fq, cfg)
        for _ in range(cfg.diffusion_iters):
            h = a @ h
    if cfg.norm != "none":
        h = minmax(h)
    info = {"K_fg": int(k_fg), "K_bg": int(sup.p_bg.shape[0]),
            "W": [round(float(x), 4) for x in w],
            "C": [round(float(x), 4) for x in c] if c is not None else None,
            "R": [round(float(x), 4) for x in r] if r is not None else None}
    return h, info
