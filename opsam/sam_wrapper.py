"""SAM2 image-predictor wrapper used as the mask generator."""
from __future__ import annotations

import numpy as np
import torch


class Sam2Prompter:
    """Thin wrapper exposing ``(points, labels) -> (bool mask, predicted IoU)``."""

    def __init__(self, ckpt: str, cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml", device="cuda"):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = torch.device(device)
        model = build_sam2(cfg, ckpt, device=self.device)
        self.predictor = SAM2ImagePredictor(model)

    def set_image(self, image: np.ndarray):
        self.predictor.set_image(image)

    @torch.inference_mode()
    def prompt(self, points: np.ndarray, labels: np.ndarray, box=None, mask_input=None,
               multimask: bool | None = None, scorer=None):
        """``scorer(list[mask]) -> list[float]`` picks among SAM's ambiguity candidates.

        SAM's own IoU head ranks the whole/part/subpart candidates poorly for polyps,
        so EPE ranks them by agreement with the semantic prior instead.
        """
        if multimask is None:
            multimask = box is None
        masks, ious, low_res = self.predictor.predict(
            point_coords=points.astype(np.float32) if points is not None else None,
            point_labels=labels.astype(np.int32) if labels is not None else None,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask,
        )
        cands = [m.astype(bool) for m in masks]
        scores = scorer(cands) if scorer is not None else list(ious)
        best = int(np.argmax(scores))
        return cands[best], float(ious[best]), low_res[best : best + 1]
