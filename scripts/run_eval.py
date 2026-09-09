"""Main evaluation entry point for the RPG-SAM reproduction."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpgsam.paths import SAM2_CKPT, kvasir_root  # noqa: E402

from opsam.datasets import kvasir_hard, load_dataset  # noqa: E402
from opsam.features import DinoV2Extractor  # noqa: E402
from opsam.sam_wrapper import Sam2Prompter  # noqa: E402
from rpgsam.faults import FAULTS, corrupt_support  # noqa: E402
from rpgsam.gas import GEO_MODES, GASConfig  # noqa: E402
from rpgsam.metrics import Accumulator, iou_dice, pr_auc, roc_auc  # noqa: E402
from rpgsam.pipeline import RPGConfig, RPGSAM  # noqa: E402
from rpgsam.pir import PIRConfig  # noqa: E402
from rpgsam.rwpm import RWPMConfig  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="kvasir")
    p.add_argument("--root", default=None)
    p.add_argument("--subset", default="all", choices=["all", "hard"],
                   help="filters the whole dataset, so the support is drawn from the "
                        "subset too -- results are not comparable with a full-set run")
    p.add_argument("--query-subset", default="all", choices=["all", "hard"],
                   help="filters only the QUERIES, keeping the support image fixed. This "
                        "is what B3 wants: same support, harder queries, one variable")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--support-index", type=int, default=850,
                   help="850 = the support image the OP-SAM reproduction fixed (seed 0)")
    p.add_argument("--support-seed", type=int, default=-1,
                   help=">=0 draws the support image at random instead")
    p.add_argument("--out", default="outputs/run")
    p.add_argument("--tag", default="")
    p.add_argument("--img-size", type=int, default=560)
    p.add_argument("--save-per-image", action="store_true")

    g = p.add_argument_group("RWPM")
    g.add_argument("--slic-k", type=int, default=10)
    g.add_argument("--slic-m", type=float, default=20.0)
    g.add_argument("--top-n", type=int, default=10)
    g.add_argument("--sim", default="cosine", choices=["cosine", "dot"])
    g.add_argument("--fg-scale", default="kfg", choices=["kfg", "one", "inv"])
    g.add_argument("--no-bg-suppress", action="store_true")
    g.add_argument("--no-reliability", action="store_true", help="W_k = 1 (drops RWPM)")
    g.add_argument("--softmax-temp", type=float, default=1.0)
    g.add_argument("--diffusion", default="row", choices=["row", "sinkhorn", "none"])
    g.add_argument("--diffusion-iters", type=int, default=2)
    g.add_argument("--affinity-pow", type=float, default=4.0)

    g = p.add_argument_group("GAS")
    g.add_argument("--tau-min", type=float, default=0.4)
    g.add_argument("--tau-max", type=float, default=0.7)
    g.add_argument("--tau-stride", type=float, default=0.05)
    g.add_argument("--comp-ratio", type=float, default=0.2)
    g.add_argument("--aref", default="support", choices=["support", "frac", "none"])
    g.add_argument("--aref-frac", type=float, default=0.10)
    g.add_argument("--fixed-tau", type=float, default=None, help="disables the GAS scan")
    g.add_argument("--geo-mode", default="full", choices=list(GEO_MODES),
                   help="C3: which reading of Eq. 5 scores the candidates")

    g = p.add_argument_group("PIR")
    g.add_argument("--tau-cov", type=float, default=0.9)
    g.add_argument("--tau-iou", type=float, default=0.8)
    g.add_argument("--max-iter", type=int, default=5)
    g.add_argument("--cand-score", default="sam_iou", choices=["prior_iou", "sam_iou"])
    g.add_argument("--no-pir", action="store_true", help="single prompt, no refinement loop")
    p.add_argument("--prior-only", action="store_true", help="score M_prior, skip SAM2")

    g = p.add_argument_group("A1 fault injection")
    g.add_argument("--fault", default="none", choices=list(FAULTS),
                   help="degrade the support pair before RWPM sees it")
    g.add_argument("--fault-strength", type=float, default=1.0)
    g.add_argument("--fault-seed", type=int, default=0)
    return p.parse_args()


def build_cfg(a) -> RPGConfig:
    return RPGConfig(
        rwpm=RWPMConfig(n_segments=a.slic_k, compactness=a.slic_m, top_n=a.top_n,
                        sim=a.sim, fg_scale=a.fg_scale, bg_suppress=not a.no_bg_suppress,
                        reliability=not a.no_reliability, softmax_temp=a.softmax_temp,
                        diffusion=a.diffusion, diffusion_iters=a.diffusion_iters,
                        affinity_pow=a.affinity_pow),
        gas=GASConfig(tau_min=a.tau_min, tau_max=a.tau_max, stride=a.tau_stride,
                      comp_ratio=a.comp_ratio, aref_mode=a.aref, aref_frac=a.aref_frac,
                      fixed_tau=a.fixed_tau, geo_mode=a.geo_mode),
        pir=PIRConfig(tau_cov=a.tau_cov, tau_iou=a.tau_iou, max_iter=a.max_iter,
                      cand_score=a.cand_score, enabled=not a.no_pir),
    )


def recall_precision(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """B1 needs recall separately from IoU: a shrunken mask and a missed polyp look
    identical in mIoU but not in recall."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    g, p_ = float(gt.sum()), float(pred.sum())
    return (inter / g if g else 1.0), (inter / p_ if p_ else 0.0)


def main():
    a = parse_args()
    samples = load_dataset(a.dataset, a.root or str(kvasir_root()))
    if a.subset == "hard":
        samples = kvasir_hard(samples)
    sup_idx = (a.support_index if a.support_seed < 0
               else int(np.random.default_rng(a.support_seed).integers(len(samples))))
    if not 0 <= sup_idx < len(samples):
        raise SystemExit(f"--support-index {sup_idx} is out of range for "
                         f"{a.dataset}/{a.subset} ({len(samples)} samples); "
                         f"use --support-seed to draw one from this subset instead")
    support = samples[sup_idx]
    queries = [s for i, s in enumerate(samples) if i != sup_idx]
    if a.query_subset == "hard":
        keep = {s.name for s in kvasir_hard(samples)}
        queries = [s for s in queries if s.name in keep]
    if a.limit:
        queries = queries[: a.limit]
    print(f"[data] {a.dataset}/{a.subset} (queries: {a.query_subset}): "
          f"{len(samples)} samples, {len(queries)} queries")
    print(f"[support] idx={sup_idx} name={support.name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ex = DinoV2Extractor(img_size=a.img_size, device=device)
    prompter = None if a.prior_only else Sam2Prompter(str(SAM2_CKPT), device=device)
    model = RPGSAM(ex, prompter, build_cfg(a))

    sup_img, sup_mask = support.load()
    if a.fault != "none":
        before = float(sup_mask.mean())
        sup_img, sup_mask = corrupt_support(sup_img, sup_mask, a.fault,
                                            a.fault_strength, a.fault_seed)
        print(f"[fault] {a.fault} strength={a.fault_strength} "
              f"support area {100*before:.2f}% -> {100*float(sup_mask.mean()):.2f}%")
    sup = model.build_support(sup_img, sup_mask)
    print(f"[RWPM] K_fg={sup.p_fg.shape[0]} K_bg={sup.p_bg.shape[0]} "
          f"C={[round(float(c), 3) for c in sup.contrast]}")

    acc, prior_acc = Accumulator(), Accumulator()
    rows, taus, t0 = [], [], time.time()
    for s in tqdm(queries, ncols=88):
        img, gt = s.load()
        if a.prior_only:
            heat, prior, info = model.compute_prior(sup, img)
            mask, trace = prior, None
        else:
            mask, heat, prior, trace, info = model.segment(sup, img)
        iou, dice = iou_dice(mask, gt)
        auc, ap = roc_auc(heat, gt), pr_auc(heat, gt)
        acc.add(iou, dice, auc, ap)
        pi, pd = iou_dice(prior, gt)
        prior_acc.add(pi, pd, auc, ap)
        taus.append(info["gas"]["tau"])
        rec, prec = recall_precision(mask, gt)
        prec_rec, prior_prec = recall_precision(prior, gt)
        rows.append({"name": s.name, "iou": iou, "dice": dice, "auc": auc, "ap": ap,
                     "recall": rec, "precision": prec,
                     "prior_iou": pi, "prior_recall": prec_rec,
                     "prior_precision": prior_prec,
                     "gt_area": float(gt.mean()), "pred_area": float(mask.mean()),
                     "tau": info["gas"]["tau"], "s_geo": info["gas"]["S_geo"],
                     "rounds": trace.rounds if trace else None,
                     "negatives": trace.negatives if trace else None,
                     "stopped": trace.stopped if trace else None})

    summary = acc.summary()
    summary.update({
        "dataset": a.dataset, "subset": a.subset,
        "query_subset": a.query_subset, "method": "rpgsam",
        "support": support.name, "support_idx": sup_idx,
        "prior_only": a.prior_only, "seconds": round(time.time() - t0, 1),
        "tag": a.tag, "prior": prior_acc.summary(),
        "tau_mean": float(np.mean([t for t in taus if t is not None])),
        "tau_hist": {str(k): int(v) for k, v in
                     zip(*np.unique([round(t, 3) for t in taus if t is not None],
                                    return_counts=True))},
        "cfg": {"slic_k": a.slic_k, "slic_m": a.slic_m, "top_n": a.top_n, "sim": a.sim,
                "fg_scale": a.fg_scale, "bg_suppress": not a.no_bg_suppress,
                "reliability": not a.no_reliability, "diffusion": a.diffusion,
                "diffusion_iters": a.diffusion_iters, "affinity_pow": a.affinity_pow,
                "gas_range": [a.tau_min, a.tau_max, a.tau_stride], "fixed_tau": a.fixed_tau,
                "aref": a.aref, "tau_cov": a.tau_cov, "tau_iou": a.tau_iou,
                "max_iter": a.max_iter, "pir": not a.no_pir, "cand_score": a.cand_score,
                "geo_mode": a.geo_mode, "fault": a.fault,
                "fault_strength": a.fault_strength, "fault_seed": a.fault_seed},
    })
    summary["recall"] = 100 * float(np.mean([r["recall"] for r in rows]))
    summary["precision"] = 100 * float(np.mean([r["precision"] for r in rows]))
    summary["prior"]["recall"] = 100 * float(np.mean([r["prior_recall"] for r in rows]))
    summary["s_geo_mean"] = float(np.mean([r["s_geo"] for r in rows]))
    if rows and rows[0].get("rounds") is not None:
        summary["avg_rounds"] = float(np.mean([r["rounds"] for r in rows]))
        summary["avg_negatives"] = float(np.mean([r["negatives"] for r in rows]))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = a.tag or f"{a.dataset}_{a.subset}_rpgsam"
    payload = {"summary": summary}
    if a.save_per_image or len(rows) <= 1200:
        payload["per_image"] = rows
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
