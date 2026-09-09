"""Reference one-shot baselines: Oracle and PerSAM (both training-free)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .features import mask_to_feature_grid, minmax, upsample_prior
from .prompting import oracle_points


class OracleBaseline:
    """Prompts SAM2 with k points sampled from the test ground truth (paper Tab. 1)."""

    def __init__(self, prompter, k=3, seed=0):
        self.prompter = prompter
        self.k = k
        self.rng = np.random.default_rng(seed)

    def segment(self, image: np.ndarray, gt: np.ndarray):
        self.prompter.set_image(image)
        return oracle_points(gt, self.prompter, self.k, self.rng)


class PerSAM:
    """Training-free PerSAM: prototype cosine map -> pos/neg point prior -> cascaded refinement."""

    def __init__(self, extractor, prompter, refine_steps: int = 2):
        self.ex = extractor
        self.prompter = prompter
        self.refine_steps = refine_steps
        self.proto = None

    def set_support(self, image: np.ndarray, mask: np.ndarray):
        feat, _ = self.ex.forward(Image.fromarray(image), want_attn=False)
        m = mask_to_feature_grid(torch.from_numpy(mask).float(), self.ex.grid).to(feat.device)
        w = (m > 0.5).float()
        if w.sum() == 0:
            w = m
        self.proto = F.normalize((feat * w[:, None]).sum(0) / w.sum().clamp_min(1.0), dim=-1)

    def confidence_map(self, query: np.ndarray):
        fq, _ = self.ex.forward(Image.fromarray(query), want_attn=False)
        sim = F.normalize(fq, dim=-1) @ self.proto
        h, w = query.shape[:2]
        return upsample_prior(minmax(sim), self.ex.grid, (h, w)).cpu().numpy()

    def segment(self, query: np.ndarray):
        conf = self.confidence_map(query)
        hi = np.unravel_index(int(conf.ravel().argmax()), conf.shape)
        lo = np.unravel_index(int(conf.ravel().argmin()), conf.shape)
        pts = np.array([[hi[1], hi[0]], [lo[1], lo[0]]], np.float32)
        labels = np.array([1, 0], np.int32)

        self.prompter.set_image(query)
        mask, _, low_res = self.prompter.prompt(pts, labels)
        for _ in range(self.refine_steps):
            if not mask.any():
                break
            ys, xs = np.nonzero(mask)
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)
            mask, _, low_res = self.prompter.prompt(
                pts, labels, box=box, mask_input=low_res, multimask=False
            )
        return mask, conf
