"""Per-image ground-truth geometry, cached for the gap analyses.

Nothing here touches DINOv2 or SAM2 -- it is a pure CPU pass over the mask files, so
it runs in seconds and is the join key for every post-hoc experiment (B1, C2, Phase 4).

The solidity definition is deliberately *identical* to the one GAS scores candidates
with (``rpgsam.gas.solidity``, area-weighted over connected components), so that
"solidity of the GT" and "the S_geo GAS would award a perfect prediction" are on the
same scale.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpgsam.paths import kvasir_root  # noqa: E402

from opsam.datasets import load_dataset  # noqa: E402
from rpgsam.gas import solidity  # noqa: E402


def components(mask: np.ndarray) -> list[np.ndarray]:
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), 8)
    return [lab == i for i in range(1, n)]


def gt_geometry(mask: np.ndarray, a_ref_ratio: float) -> dict:
    """Everything C2/B1 stratify on, plus the S_geo GAS would award this exact shape."""
    total = float(mask.sum())
    hw = float(mask.size)
    comps = components(mask)
    if total <= 0 or not comps:
        return {"area_ratio": 0.0, "solidity": 0.0, "n_comp": 0,
                "s_geo_gt": 0.0, "scale_gt": 0.0, "elongation": 0.0}

    # area-weighted solidity, exactly rpgsam.gas.geo_score's first factor
    wsol = sum((float(c.sum()) / total) * solidity(c) for c in comps)
    scale = min(1.0, (total / hw) / a_ref_ratio) if a_ref_ratio > 0 else 1.0

    big = max(comps, key=lambda c: c.sum())
    cnts, _ = cv2.findContours(big.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elong = 0.0
    if cnts and len(max(cnts, key=cv2.contourArea)) >= 5:
        (_, (ma, mi), _) = cv2.fitEllipse(max(cnts, key=cv2.contourArea))
        elong = float(max(ma, mi) / max(min(ma, mi), 1e-6))

    return {"area_ratio": total / hw,
            "solidity": float(wsol),
            "n_comp": len(comps),
            "s_geo_gt": float(wsol * scale),   # the score GAS gives a perfect candidate
            "scale_gt": float(scale),
            "elongation": elong}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="kvasir")
    p.add_argument("--root", default=None)
    p.add_argument("--support-index", type=int, default=850,
                   help="A_ref comes from this image's polyp area, as GAS does it")
    p.add_argument("--out", default="outputs/analysis/gt_stats.json")
    a = p.parse_args()

    samples = load_dataset(a.dataset, a.root or str(kvasir_root()))
    sup = samples[a.support_index]
    _, sup_mask = sup.load()
    a_ref_ratio = float(sup_mask.mean())
    print(f"[support] {sup.name}  A_ref = {a_ref_ratio:.4f} of the image")

    rows = {}
    for s in tqdm(samples, ncols=88):
        _, gt = s.load()
        rows[s.name] = gt_geometry(gt, a_ref_ratio)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": a.dataset, "support": sup.name,
                               "a_ref_ratio": a_ref_ratio, "stats": rows}, indent=1))
    ar = np.array([r["area_ratio"] for r in rows.values()])
    so = np.array([r["solidity"] for r in rows.values()])
    print(f"[gt] n={len(rows)}  area% median={100*np.median(ar):.2f} "
          f"[{100*ar.min():.2f}, {100*ar.max():.2f}]  solidity median={np.median(so):.3f}")
    print(f"[gt] wrote {out}")


if __name__ == "__main__":
    main()
