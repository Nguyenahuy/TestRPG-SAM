"""DINOv2 feature + last-block self-attention extraction (paper Sec. 3.1)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_HUB_NAMES = {
    "vit_large": "dinov2_vitl14",
    "vit_base": "dinov2_vitb14",
    "vit_small": "dinov2_vits14",
    "vit_giant": "dinov2_vitg14",
}


class DinoV2Extractor:
    """Wraps a frozen DINOv2 ViT.

    Returns patch tokens ``f in R^{hw x D}`` and the head-averaged self-attention
    of the final block, restricted to patch-patch entries (``R^{hw x hw}``).
    """

    def __init__(self, arch="vit_large", img_size=560, device="cuda", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.img_size = img_size
        self.patch = 14
        self.grid = img_size // self.patch

        self.model = torch.hub.load("facebookresearch/dinov2", _HUB_NAMES[arch], verbose=False)
        self.model.eval().to(self.device, dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.dim = self.model.embed_dim
        self.n_prefix = 1 + getattr(self.model, "num_register_tokens", 0)

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self._attn = None
        self._hook_last_attention()

    def _hook_last_attention(self):
        """Recompute q@k^T inside the last block's attention from its qkv output.

        DINOv2 uses memory-efficient attention that never materialises the score
        matrix, so the scores are rebuilt from the captured qkv projection.
        """
        blk = self.model.blocks[-1]
        attn = blk.attn
        self.n_heads = attn.num_heads

        def hook(module, inp, out):
            B, N, C3 = out.shape
            C = C3 // 3
            h = self.n_heads
            qkv = out.reshape(B, N, 3, h, C // h).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            scores = (q @ k.transpose(-2, -1)) * (C // h) ** -0.5
            self._attn = scores.softmax(-1).mean(1)  # head-averaged, (B, N, N)

        attn.qkv.register_forward_hook(hook)

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self.tf(img.convert("RGB"))

    @torch.no_grad()
    def forward(self, img: Image.Image, want_attn=False):
        x = self.preprocess(img)[None].to(self.device, self.dtype)
        self._attn = None
        out = self.model.forward_features(x)
        feat = out["x_norm_patchtokens"][0].float()  # (hw, D)
        attn = None
        if want_attn:
            p = self.n_prefix
            attn = self._attn[0, p:, p:].float()  # (hw, hw)
        return feat, attn


def mask_to_feature_grid(mask: torch.Tensor, grid: int) -> torch.Tensor:
    """Resize an HxW binary mask to grid x grid and flatten to (hw,)."""
    m = mask[None, None].float()
    m = F.interpolate(m, size=(grid, grid), mode="bilinear", align_corners=False)
    return m.reshape(-1)


def upsample_prior(prior: torch.Tensor, grid: int, size) -> torch.Tensor:
    """Bilinearly upsample a flat feature-level prior back to image resolution."""
    p = prior.reshape(1, 1, grid, grid)
    p = F.interpolate(p, size=size, mode="bilinear", align_corners=False)
    return p[0, 0]


def minmax(x: torch.Tensor, eps=1e-8) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + eps)
