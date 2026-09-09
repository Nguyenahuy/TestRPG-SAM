"""Qualitative panels: heatmap, GAS scan, M_prior, PIR prompts, prediction (paper Fig. 3)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpgsam.paths import SAM2_CKPT, kvasir_root  # noqa: E402

from opsam.datasets import load_dataset  # noqa: E402
from opsam.features import DinoV2Extractor  # noqa: E402
from opsam.sam_wrapper import Sam2Prompter  # noqa: E402
from rpgsam.gas import refine  # noqa: E402
from rpgsam.metrics import iou_dice  # noqa: E402
from rpgsam.pipeline import RPGConfig, RPGSAM  # noqa: E402


def overlay(ax, img, mask, color=(1, 0.2, 0.2), alpha=0.45):
    ax.imshow(img)
    if mask is not None and mask.any():
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask.astype(bool)] = (*color, alpha)
        ax.imshow(rgba)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--support-index", type=int, default=850)
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--names", default="",
                    help="comma-separated image names to render instead of the first n; "
                         "A3 uses this to pick the failure cases B2 identified")
    ap.add_argument("--tag", default="rpgsam_panels")
    a = ap.parse_args()

    samples = load_dataset("kvasir", a.root or str(kvasir_root()))
    sup_s = samples[a.support_index]
    if a.names:
        want = [x.strip() for x in a.names.split(",") if x.strip()]
        by_name = {s.name: s for s in samples}
        missing = [w for w in want if w not in by_name]
        if missing:
            raise SystemExit(f"unknown image name(s): {missing}")
        queries = [by_name[w] for w in want]
    else:
        queries = [s for i, s in enumerate(samples) if i != a.support_index][: a.n]

    ex = DinoV2Extractor(img_size=560, device="cuda")
    sam = Sam2Prompter(str(SAM2_CKPT), device="cuda")
    model = RPGSAM(ex, sam, RPGConfig())
    sup_img, sup_mask = sup_s.load()
    sup = model.build_support(sup_img, sup_mask)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cols = ["image", "heatmap", "M_prior", "PIR prompts", "prediction", "ground truth"]
    fig, axes = plt.subplots(len(queries), len(cols), figsize=(3 * len(cols), 3 * len(queries)))
    for r, s in enumerate(queries):
        img, gt = s.load()
        mask, heat, prior, trace, info = model.segment(sup, img)
        iou = iou_dice(mask, gt)[0]
        ax = axes[r]
        ax[0].imshow(img); ax[0].axis("off")
        ax[0].set_ylabel(s.name[:10])
        ax[1].imshow(heat, cmap="jet", vmin=0, vmax=1); ax[1].axis("off")
        overlay(ax[2], img, prior, (0.2, 0.6, 1.0))
        ax[2].set_title(f"tau={info['gas']['tau']}", fontsize=8)
        overlay(ax[3], img, prior, (0.2, 0.6, 1.0), 0.25)
        for (x, y), lab in zip(trace.points, trace.labels):
            ax[3].plot(x, y, "*" if lab else "x", c="lime" if lab else "red", ms=11)
        ax[3].set_title(f"{trace.rounds} rounds ({trace.stopped})", fontsize=8)
        overlay(ax[4], img, mask, (1, 0.2, 0.2))
        ax[4].set_title(f"IoU {100 * iou:.1f}", fontsize=8)
        overlay(ax[5], img, gt, (0.2, 1, 0.2))
        if r == 0:
            for c, name in enumerate(cols):
                ax[c].set_title(f"{name}\n{ax[c].get_title()}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"{a.tag}.png", dpi=110)
    print("wrote", out / f"{a.tag}.png")

    # GAS candidate scan for the first query
    img, gt = queries[0].load()
    heat, prior, info = model.compute_prior(sup, img)
    scan = info["gas"]["scan"]
    fig, axes = plt.subplots(1, len(scan) + 1, figsize=(2.6 * (len(scan) + 1), 3))
    axes[0].imshow(heat, cmap="jet", vmin=0, vmax=1)
    axes[0].set_title("heatmap", fontsize=9); axes[0].axis("off")
    for i, c in enumerate(scan):
        m, _ = refine(heat > c["tau"], 0.2)
        overlay(axes[i + 1], img, m, (0.2, 0.6, 1.0))
        star = " *" if c["tau"] == info["gas"]["tau"] else ""
        axes[i + 1].set_title(f"tau={c['tau']} S={c['score']:.3f}{star}\n"
                              f"IoU {100 * iou_dice(m, gt)[0]:.1f}" if m is not None
                              else f"tau={c['tau']} empty", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "gas_scan.png", dpi=110)
    print("wrote", out / "gas_scan.png")


if __name__ == "__main__":
    main()
