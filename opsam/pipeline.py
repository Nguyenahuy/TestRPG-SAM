"""OP-SAM end-to-end pipeline: CPG -> SPF -> EPE (paper Fig. 2)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from .cpg import cpg_prior
from .epe import EPEConfig
from .features import DinoV2Extractor, mask_to_feature_grid, minmax, upsample_prior
from .prompting import run_prompting
from .spf import fuse_priors, scale_lesion


@dataclass
class OPSAMConfig:
    rho: int = 2
    corr_mode: str = "softmax"
    sinkhorn_iters: int = 3
    self_refine: bool = True
    affinity_pow: float = 2.0
    scale_xl: float = 1.5
    scale_xs: float = 0.5
    tau: float = 0.5
    fusion: str = "spf"          # spf | avg | none
    use_aug: bool = True
    prompting: str = "epe"
    epe: EPEConfig = field(default_factory=EPEConfig)


class SupportBank:
    """Holds DINOv2 features and feature-grid masks for every support entry."""

    def __init__(self, extractor: DinoV2Extractor):
        self.ex = extractor
        self.feats: dict[str, torch.Tensor] = {}
        self.masks: dict[str, torch.Tensor] = {}

    def add(self, key: str, image: np.ndarray, mask: np.ndarray):
        feat, _ = self.ex.forward(Image.fromarray(image), want_attn=False)
        self.feats[key] = feat
        self.masks[key] = mask_to_feature_grid(torch.from_numpy(mask).float(), self.ex.grid).to(feat.device)

    @property
    def keys(self):
        return list(self.feats)


def build_support(ex: DinoV2Extractor, image: np.ndarray, mask: np.ndarray, cfg: OPSAMConfig) -> SupportBank:
    """Original support plus the zoomed-in / zoomed-out lesion augmentations (Sec. 3.2)."""
    bank = SupportBank(ex)
    bank.add("ori", image, mask)
    if cfg.use_aug:
        xl_img, xl_m = scale_lesion(image, mask, cfg.scale_xl)
        xs_img, xs_m = scale_lesion(image, mask, cfg.scale_xs)
        bank.add("xl", xl_img, xl_m)
        bank.add("xs", xs_img, xs_m)
    return bank


def build_support_multi(ex: DinoV2Extractor, pairs, cfg: OPSAMConfig) -> SupportBank:
    """Support bank from several annotated images (Table 5 k-shot rows)."""
    bank = SupportBank(ex)
    for i, (img, m) in enumerate(pairs):
        bank.add(f"s{i}", img, m)
    return bank


class OPSAM:
    def __init__(self, extractor: DinoV2Extractor, prompter, cfg: OPSAMConfig):
        self.ex = extractor
        self.prompter = prompter
        self.cfg = cfg

    def compute_prior(self, bank: SupportBank, query: np.ndarray):
        """CPG per support entry, then SPF fusion. Returns image-resolution prior."""
        fq, attn = self.ex.forward(Image.fromarray(query), want_attn=self.cfg.self_refine)
        priors, inits = {}, {}
        for k in bank.keys:
            p, p_init = cpg_prior(
                fq, bank.feats[k], bank.masks[k], attn,
                rho=self.cfg.rho, corr_mode=self.cfg.corr_mode,
                sinkhorn_iters=self.cfg.sinkhorn_iters, self_refine=self.cfg.self_refine,
                affinity_pow=self.cfg.affinity_pow,
            )
            priors[k], inits[k] = p, p_init

        info: dict = {}
        if len(priors) == 1 or self.cfg.fusion == "none":
            fused = priors[bank.keys[0]]
        elif self.cfg.fusion == "avg":
            fused = minmax(sum(priors.values()) / len(priors))
        else:
            fused, w, ciou = fuse_priors(priors, fq, bank.feats, bank.masks, self.cfg.tau)
            info = {"weights": w, "cIoU": ciou}

        h, w_ = query.shape[:2]
        prior_img = upsample_prior(fused, self.ex.grid, (h, w_)).cpu().numpy()
        init_img = upsample_prior(inits[bank.keys[0]], self.ex.grid, (h, w_)).cpu().numpy()
        return prior_img, init_img, info

    def segment(self, bank: SupportBank, query: np.ndarray, rng=None):
        prior, init, info = self.compute_prior(bank, query)
        self.prompter.set_image(query)
        mask, trace = run_prompting(self.cfg.prompting, prior, self.prompter, self.cfg.epe, rng)
        return mask, prior, init, trace, info
