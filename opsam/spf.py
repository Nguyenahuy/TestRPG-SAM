"""Scale-cascaded Prior Fusion (paper Sec. 3.2, Eq. 3-5)."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .features import minmax


def scale_lesion(img: np.ndarray, mask: np.ndarray, scale: float, inpaint_radius: int = 3):
    """Cut the polyp out, resize it about its centroid, paste it back in place.

    The blank gap left behind by a shrunken lesion is filled by inpainting
    (paper Sec. 4.2). Returns the augmented ``(image, mask)`` pair.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0 or abs(scale - 1.0) < 1e-6:
        return img.copy(), mask.copy()

    h, w = mask.shape
    cy, cx = float(ys.mean()), float(xs.mean())
    aff = np.array([[scale, 0, cx * (1 - scale)], [0, scale, cy * (1 - scale)]], np.float32)

    warped = cv2.warpAffine(img, aff, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    new_mask = cv2.warpAffine(mask.astype(np.uint8), aff, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    out = img.copy()
    gap = (mask > 0) & (new_mask == 0)
    if gap.any():
        out[gap] = 0
        out = cv2.inpaint(out, gap.astype(np.uint8), inpaint_radius, cv2.INPAINT_TELEA)
    sel = new_mask > 0
    out[sel] = warped[sel]
    return out, new_mask.astype(np.uint8)


def confidence_iou(p_rev: torch.Tensor, ms_r: torch.Tensor, tau: float) -> torch.Tensor:
    """cIoU: like IoU but the intersection is weighted by the reverse-prior probability."""
    b = p_rev > tau
    gt = ms_r > 0.5
    inter = (p_rev * (b & gt).float()).sum()
    union = (b | gt).float().sum().clamp_min(1.0)
    return inter / union


def reverse_transfer(fq: torch.Tensor, fs: torch.Tensor, prior: torch.Tensor, tau: float) -> torch.Tensor:
    """Eq. 3: push the query prior back onto the support image to score its quality."""
    sel = prior > tau
    if sel.sum() == 0:
        return torch.zeros(fs.shape[0], device=fs.device)
    fq_sel = F.normalize(fq[sel], dim=-1)
    cos = fq_sel @ F.normalize(fs, dim=-1).T  # (n, hw)
    return minmax(cos.mean(dim=0))


def fuse_priors(
    priors: dict[str, torch.Tensor],
    fq: torch.Tensor,
    feats_s: dict[str, torch.Tensor],
    masks_s: dict[str, torch.Tensor],
    tau: float = 0.5,
):
    """Eq. 4-5: adaptively weight the ori/xl/xs priors by reverse-transfer cIoU."""
    ciou = {}
    for k, p in priors.items():
        p_rev = reverse_transfer(fq, feats_s[k], p, tau)
        ciou[k] = confidence_iou(p_rev, masks_s[k], tau)

    total = sum(ciou.values())
    if float(total) <= 0:
        w = {k: 1.0 / len(priors) for k in priors}
    else:
        w = {k: float(v / total) for k, v in ciou.items()}

    p_avg = sum(w[k] * priors[k] for k in priors)
    return minmax(p_avg), w, {k: float(v) for k, v in ciou.items()}
