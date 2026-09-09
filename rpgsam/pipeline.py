"""RPG-SAM end-to-end pipeline: RWPM -> GAS -> PIR (paper Fig. 2)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from opsam.features import upsample_prior

from .gas import GASConfig, select
from .pir import PIRConfig, pir_segment
from .rwpm import RWPMConfig, SupportPrototypes, heatmap


@dataclass
class RPGConfig:
    rwpm: RWPMConfig = field(default_factory=RWPMConfig)
    gas: GASConfig = field(default_factory=GASConfig)
    pir: PIRConfig = field(default_factory=PIRConfig)


class RPGSAM:
    def __init__(self, extractor, prompter, cfg: RPGConfig):
        self.ex = extractor
        self.prompter = prompter
        self.cfg = cfg

    def build_support(self, image: np.ndarray, mask: np.ndarray) -> SupportPrototypes:
        return SupportPrototypes(self.ex, image, mask, self.cfg.rwpm)

    def compute_heatmap(self, sup: SupportPrototypes, query: np.ndarray):
        fq, _ = self.ex.forward(Image.fromarray(query), want_attn=False)
        h, info = heatmap(sup, fq, self.cfg.rwpm)
        img = upsample_prior(h, self.ex.grid, query.shape[:2]).cpu().numpy()
        return img, info

    def compute_prior(self, sup: SupportPrototypes, query: np.ndarray):
        heat, info = self.compute_heatmap(sup, query)
        prior, gas_info = select(heat, self.cfg.gas, sup.area_ratio)
        return heat, prior, {**info, "gas": gas_info}

    def segment(self, sup: SupportPrototypes, query: np.ndarray):
        heat, prior, info = self.compute_prior(sup, query)
        self.prompter.set_image(query)

        def predict(points, labels, scorer):
            m, iou, _ = self.prompter.prompt(points, labels, scorer=scorer)
            return m, iou

        mask, trace = pir_segment(prior, predict, self.cfg.pir)
        return mask, heat, prior, trace, info
