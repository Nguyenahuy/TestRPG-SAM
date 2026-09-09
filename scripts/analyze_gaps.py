"""Post-hoc gap analysis: B1, B2, C2, A2 and the Phase-4 cross-tabulation.

Pure joins over per-image JSON already written by ``run_eval.py`` plus the cached GT
geometry -- no model is loaded, so the whole thing runs in a second on CPU and can be
re-run against any set of runs.

Every group statistic carries a percentile-bootstrap CI, because the size/solidity
strata are small enough that point estimates alone would not settle anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- helpers

def load_run(path: str) -> dict[str, dict]:
    d = json.loads(Path(path).read_text())
    if "per_image" not in d:
        raise SystemExit(f"{path} has no per_image block")
    return {r["name"]: r for r in d["per_image"]}


def boot_mean(x: np.ndarray, n: int = 4000):
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 4000):
    """CI of mean(a) - mean(b) for two independent groups."""
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")
    da = a[RNG.integers(0, len(a), size=(n, len(a)))].mean(axis=1)
    db = b[RNG.integers(0, len(b), size=(n, len(b)))].mean(axis=1)
    d = da - db
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def boot_corr(x: np.ndarray, y: np.ndarray, fn, n: int = 2000):
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    v = np.array([fn(x[i], y[i]) for i in idx])
    return fn(x, y), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def fmt(v, ci_lo=None, ci_hi=None, scale=100, d=2):
    if v is None or v != v:
        return "--"
    s = f"{v * scale:.{d}f}"
    if ci_lo is not None:
        s += f" [{ci_lo * scale:.{d}f}, {ci_hi * scale:.{d}f}]"
    return s


SIZE_BUCKETS = [("small (<5%)", lambda a: a < 0.05),
                ("medium (5-15%)", lambda a: 0.05 <= a <= 0.15),
                ("large (>15%)", lambda a: a > 0.15)]


def group_arr(rows, gt, key, pred):
    return np.array([r[key] for n, r in rows.items()
                     if n in gt and pred(gt[n]) and r.get(key) is not None])


# ---------------------------------------------------------------- experiments

def exp_B1(rows, gt, key="iou"):
    """Stratify by GT coverage: small <5%, medium 5-15%, large >15%."""
    out = {}
    for name, pred in SIZE_BUCKETS:
        v = np.array([r[key] for n, r in rows.items()
                      if n in gt and pred(gt[n]["area_ratio"]) and r.get(key) is not None])
        m, lo, hi = boot_mean(v)
        out[name] = {"n": len(v), "mean": m, "lo": lo, "hi": hi}
    return out


def exp_B2(runs, gt, thr=0.05):
    """Complete-miss rate: share of images whose prediction barely touches the GT."""
    out = {}
    for name, rows in runs.items():
        v = np.array([r["iou"] for r in rows.values() if r.get("iou") is not None])
        m, lo, hi = boot_mean((v < thr).astype(float))
        sm = np.array([r["iou"] for n, r in rows.items()
                       if n in gt and gt[n]["area_ratio"] < 0.05])
        ms, los, his = boot_mean((sm < thr).astype(float))
        out[name] = {"n": len(v), "rate": m, "lo": lo, "hi": hi,
                     "n_small": len(sm), "rate_small": ms,
                     "lo_small": los, "hi_small": his}
    return out


def exp_C2(rows, gt, key="iou"):
    """Quartiles of GT solidity -- the quantity S_geo's weighted-solidity term rewards."""
    names = [n for n in rows if n in gt]
    sol = np.array([gt[n]["solidity"] for n in names])
    q = np.percentile(sol, [25, 50, 75])
    quarts = [("Q1 (least convex)", lambda s: s <= q[0]),
              ("Q2", lambda s: q[0] < s <= q[1]),
              ("Q3", lambda s: q[1] < s <= q[2]),
              ("Q4 (most convex)", lambda s: s > q[2])]
    out, groups = {}, {}
    for lab, pred in quarts:
        sel = [n for n in names if pred(gt[n]["solidity"])]
        groups[lab] = sel
        v = np.array([rows[n][key] for n in sel if rows[n].get(key) is not None])
        m, lo, hi = boot_mean(v)
        out[lab] = {"n": len(sel), "mean": m, "lo": lo, "hi": hi,
                    "solidity": float(np.mean([gt[n]["solidity"] for n in sel])),
                    "s_geo_gt": float(np.mean([gt[n]["s_geo_gt"] for n in sel])),
                    "scale_gt": float(np.mean([gt[n]["scale_gt"] for n in sel])),
                    "area_ratio": float(np.mean([gt[n]["area_ratio"] for n in sel]))}
    a = np.array([rows[n][key] for n in groups["Q1 (least convex)"]])
    b = np.array([rows[n][key] for n in groups["Q4 (most convex)"]])
    d, dlo, dhi = boot_diff(b, a)
    out["_Q4_minus_Q1"] = {"diff": d, "lo": dlo, "hi": dhi}
    out["_quartiles"] = [float(x) for x in q]
    return out


def exp_A2(rows):
    """Correlation between the quality of M_prior and the quality of the final mask."""
    names = [n for n, r in rows.items() if r.get("prior_iou") is not None]
    p = np.array([rows[n]["prior_iou"] for n in names])
    f = np.array([rows[n]["iou"] for n in names])
    pr, plo, phi = boot_corr(p, f, pearson)
    sr, slo, shi = boot_corr(p, f, spearman)
    bins = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    rec = {}
    for lo_, hi_ in bins:
        sel = [n for n in names if lo_ <= rows[n]["prior_iou"] < hi_]
        if not sel:
            continue
        pi = np.array([rows[n]["prior_iou"] for n in sel])
        fi = np.array([rows[n]["iou"] for n in sel])
        # recovery = (final - prior) / (1 - prior): share of remaining headroom closed
        recov = (fi - pi) / np.maximum(1.0 - pi, 1e-6)
        m, l, h = boot_mean(recov)
        rec[f"[{lo_:.1f},{hi_:.1f})"] = {"n": len(sel), "prior": float(pi.mean()),
                                         "final": float(fi.mean()),
                                         "recovery": m, "lo": l, "hi": h}
    return {"pearson": pr, "pearson_lo": plo, "pearson_hi": phi,
            "spearman": sr, "spearman_lo": slo, "spearman_hi": shi,
            "n": len(names), "recovery_by_prior": rec}


def exp_A3(rows, gt, miss_thr=0.05):
    """A3, quantified: when the final mask misses the polyp, where did it go wrong?

    The plan asks this qualitatively ("visualise the hard cases"). The same question has
    a countable form: of the images the method misses completely, how many already had an
    empty M_prior? If PIR could rescue an upstream miss, that share would be well below
    100%.
    """
    names = [n for n, r in rows.items() if r.get("prior_iou") is not None]
    miss = [n for n in names if rows[n]["iou"] < miss_thr]
    prior_dead = [n for n in miss if rows[n]["prior_iou"] < miss_thr]
    d = np.array([rows[n]["iou"] - rows[n]["prior_iou"] for n in names])
    ruined = sorted((n for n in names
                     if rows[n]["prior_iou"] > 0.5
                     and rows[n]["iou"] < rows[n]["prior_iou"] - 0.25),
                    key=lambda n: rows[n]["iou"] - rows[n]["prior_iou"])
    return {"n": len(names), "n_miss": len(miss), "n_miss_prior_dead": len(prior_dead),
            "frac_miss_from_dead_prior": len(prior_dead) / max(len(miss), 1),
            "miss_gt_area_median": (float(np.median([gt[n]["area_ratio"] for n in miss
                                                     if n in gt])) if miss else float("nan")),
            "pir_helps": float((d > 0.01).mean()),
            "pir_hurts": float((d < -0.01).mean()),
            "pir_neutral": float((np.abs(d) <= 0.01).mean()),
            "mean_gain_when_helps": float(d[d > 0.01].mean()) if (d > 0.01).any() else 0.0,
            "mean_loss_when_hurts": float(d[d < -0.01].mean()) if (d < -0.01).any() else 0.0,
            "n_ruined": len(ruined),
            "examples_miss": miss[:8],
            "examples_ruined": ruined[:6]}


def exp_phase4(rows, gt):
    """2x2 of size x solidity, so 'small' and 'flat' are not confounded."""
    names = [n for n in rows if n in gt]
    sol_med = float(np.median([gt[n]["solidity"] for n in names]))
    out = {}
    for slab, spred in [("small (<5%)", lambda a: a < 0.05),
                        ("large (>=5%)", lambda a: a >= 0.05)]:
        for clab, cpred in [("flat (sol<=med)", lambda s: s <= sol_med),
                            ("convex (sol>med)", lambda s: s > sol_med)]:
            sel = [n for n in names
                   if spred(gt[n]["area_ratio"]) and cpred(gt[n]["solidity"])]
            v = np.array([rows[n]["iou"] for n in sel])
            pv = np.array([rows[n]["prior_iou"] for n in sel
                           if rows[n].get("prior_iou") is not None])
            m, lo, hi = boot_mean(v)
            pm, _, _ = boot_mean(pv)
            out[f"{slab} x {clab}"] = {"n": len(sel), "mean": m, "lo": lo,
                                       "hi": hi, "prior": pm}
    out["_solidity_median"] = sol_med
    return out


def exp_scale_consensus(gt):
    """How often does Scale Consensus damp a *perfect* candidate?

    ``scale = min(1, A/A_ref)``. Any GT smaller than the support polyp is penalised even
    when the candidate is exactly right, so this is a property of Eq. 5, not of a heatmap.
    """
    ar = np.array([g["area_ratio"] for g in gt.values()])
    sc = np.array([g["scale_gt"] for g in gt.values()])
    return {"n": len(ar),
            "frac_damped": float((sc < 0.999).mean()),
            "mean_scale_gt": float(sc.mean()),
            "median_area_ratio": float(np.median(ar)),
            "frac_scale_below_0.5": float((sc < 0.5).mean())}


# ---------------------------------------------------------------- rendering

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", default="outputs/analysis/gt_stats.json")
    p.add_argument("--rpgsam", default="outputs/tables/t1_rpgsam_kvasir.json")
    p.add_argument("--opsam", default="outputs/tables/t1_opsam.json")
    p.add_argument("--persam", default="outputs/tables/t1_persam.json")
    p.add_argument("--miss-thr", type=float, default=0.05)
    p.add_argument("--out", default="outputs/analysis/gaps.json")
    p.add_argument("--md", default="outputs/analysis/GAPS.md")
    a = p.parse_args()

    gtfile = json.loads(Path(a.gt).read_text())
    gt = gtfile["stats"]
    runs = {"RPG-SAM": load_run(a.rpgsam)}
    for lbl, path in [("OP-SAM", a.opsam), ("PerSAM", a.persam)]:
        if Path(path).exists():
            runs[lbl] = load_run(path)
    rpg = runs["RPG-SAM"]
    has_recall = "recall" in next(iter(rpg.values()))

    res = {
        "B1": {m: exp_B1(rows, gt) for m, rows in runs.items()},
        "B1_prior": exp_B1(rpg, gt, key="prior_iou"),
        "B1_recall": ({m: exp_B1(rows, gt, key="recall") for m, rows in runs.items()}
                      if has_recall else None),
        "B2": exp_B2(runs, gt, a.miss_thr),
        "C2": {m: exp_C2(rows, gt) for m, rows in runs.items()},
        "C2_prior": exp_C2(rpg, gt, key="prior_iou"),
        "A2": exp_A2(rpg),
        "A3": exp_A3(rpg, gt, a.miss_thr),
        "phase4": exp_phase4(rpg, gt),
        "scale_consensus": exp_scale_consensus(gt),
        "miss_thr": a.miss_thr,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))

    L = []
    L.append("# Gap analysis - post-hoc (no model re-run)\n")
    L.append(f"Kvasir-SEG, {len(rpg)} queries, support `{gtfile['support']}`, "
             f"A_ref = {100 * gtfile['a_ref_ratio']:.2f}% of the image. "
             "All intervals are 95% percentile bootstrap.\n")

    L.append("## B1 - IoU stratified by GT polyp coverage\n")
    L.append("| stratum | n | " + " | ".join(runs) + " | RPG-SAM M_prior |")
    L.append("|---|---|" + "---|" * (len(runs) + 1))
    for st, _ in SIZE_BUCKETS:
        cells = [fmt(res["B1"][m][st]["mean"], res["B1"][m][st]["lo"],
                     res["B1"][m][st]["hi"]) for m in runs]
        pr = res["B1_prior"][st]
        L.append(f"| {st} | {res['B1']['RPG-SAM'][st]['n']} | " + " | ".join(cells)
                 + f" | {fmt(pr['mean'], pr['lo'], pr['hi'])} |")
    d, lo, hi = boot_diff(group_arr(rpg, gt, "iou", lambda g: g["area_ratio"] > 0.15),
                          group_arr(rpg, gt, "iou", lambda g: g["area_ratio"] < 0.05))
    L.append(f"\nRPG-SAM large minus small: **{fmt(d, lo, hi)}** IoU points.")
    if has_recall:
        L.append("\nRecall by the same strata:\n")
        L.append("| stratum | " + " | ".join(runs) + " |")
        L.append("|---|" + "---|" * len(runs))
        for st, _ in SIZE_BUCKETS:
            cells = [fmt(res["B1_recall"][m][st]["mean"], res["B1_recall"][m][st]["lo"],
                         res["B1_recall"][m][st]["hi"]) for m in runs]
            L.append(f"| {st} | " + " | ".join(cells) + " |")
    L.append("")

    L.append(f"## B2 - complete-miss rate (IoU < {a.miss_thr})\n")
    L.append("| method | all images | small polyps only |")
    L.append("|---|---|---|")
    for m, v in res["B2"].items():
        L.append(f"| {m} | {fmt(v['rate'], v['lo'], v['hi'])}% (n={v['n']}) | "
                 f"{fmt(v['rate_small'], v['lo_small'], v['hi_small'])}% (n={v['n_small']}) |")
    L.append("")

    L.append("## C2 - IoU by quartile of GT solidity\n")
    q = res["C2"]["RPG-SAM"]["_quartiles"]
    L.append(f"Solidity quartile cuts: {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}\n")
    L.append("| quartile | n | solidity | S_geo(GT) | scale(GT) | GT area% | "
             + " | ".join(runs) + " | M_prior |")
    L.append("|---|---|---|---|---|---|" + "---|" * (len(runs) + 1))
    for lab in ["Q1 (least convex)", "Q2", "Q3", "Q4 (most convex)"]:
        r = res["C2"]["RPG-SAM"][lab]
        cells = [fmt(res["C2"][m][lab]["mean"], res["C2"][m][lab]["lo"],
                     res["C2"][m][lab]["hi"]) for m in runs]
        pr = res["C2_prior"][lab]
        L.append(f"| {lab} | {r['n']} | {r['solidity']:.3f} | {r['s_geo_gt']:.3f} | "
                 f"{r['scale_gt']:.3f} | {100 * r['area_ratio']:.1f} | " + " | ".join(cells)
                 + f" | {fmt(pr['mean'], pr['lo'], pr['hi'])} |")
    for m in runs:
        dd = res["C2"][m]["_Q4_minus_Q1"]
        L.append(f"\n{m} Q4 minus Q1: **{fmt(dd['diff'], dd['lo'], dd['hi'])}** IoU points.")
    L.append("")

    sc = res["scale_consensus"]
    L.append("### Scale Consensus applied to the ground truth itself\n")
    L.append(f"`scale = min(1, A/A_ref)` damps **{100 * sc['frac_damped']:.1f}%** of the "
             f"ground-truth masks (mean factor {sc['mean_scale_gt']:.3f}); "
             f"{100 * sc['frac_scale_below_0.5']:.1f}% are damped below 0.5. "
             "A perfect candidate is penalised on those images purely for being smaller "
             "than the support polyp.\n")

    L.append("## A2 - does the final mask track M_prior?\n")
    A = res["A2"]
    L.append(f"n = {A['n']}. Pearson r(IoU(M_prior,GT), IoU(final,GT)) = "
             f"**{A['pearson']:.3f}** [{A['pearson_lo']:.3f}, {A['pearson_hi']:.3f}], "
             f"Spearman rho = **{A['spearman']:.3f}** "
             f"[{A['spearman_lo']:.3f}, {A['spearman_hi']:.3f}].\n")
    L.append("Share of the remaining headroom `(final - prior) / (1 - prior)` that "
             "PIR+SAM2 closes, by how good the prior was:\n")
    L.append("| prior IoU band | n | mean prior | mean final | recovery |")
    L.append("|---|---|---|---|---|")
    for k, v in A["recovery_by_prior"].items():
        L.append(f"| {k} | {v['n']} | {fmt(v['prior'])} | {fmt(v['final'])} | "
                 f"{fmt(v['recovery'], v['lo'], v['hi'])}% |")
    L.append("")

    T = res["A3"]
    L.append("## A3 - where the failures actually come from\n")
    L.append(f"Of the {T['n_miss']} images RPG-SAM misses completely (IoU < {a.miss_thr}), "
             f"**{T['n_miss_prior_dead']} ({100 * T['frac_miss_from_dead_prior']:.1f}%)** "
             f"already had an empty `M_prior`. Their median GT covers "
             f"{100 * T['miss_gt_area_median']:.2f}% of the image.\n")
    L.append(f"Across all {T['n']} queries the PIR loop improves on `M_prior` in "
             f"{100 * T['pir_helps']:.1f}% of images (mean {T['mean_gain_when_helps']:+.3f} "
             f"IoU), **degrades it in {100 * T['pir_hurts']:.1f}%** "
             f"(mean {T['mean_loss_when_hurts']:+.3f}), and leaves it unchanged in "
             f"{100 * T['pir_neutral']:.1f}%. On {T['n_ruined']} images a prior above 0.5 "
             "IoU is driven more than 0.25 below itself by the loop's negative prompts.\n")
    L.append("Cases for the qualitative panels (`scripts/visualize.py --names ...`):\n")
    L.append(f"- upstream miss: `{','.join(T['examples_miss'][:4])}`")
    L.append(f"- loop destroyed a good prior: `{','.join(T['examples_ruined'][:4])}`\n")

    L.append("## Phase 4 - size x solidity, cross-controlled\n")
    L.append(f"Solidity median = {res['phase4']['_solidity_median']:.3f}\n")
    L.append("| cell | n | RPG-SAM IoU | M_prior IoU |")
    L.append("|---|---|---|---|")
    for k, v in res["phase4"].items():
        if k.startswith("_"):
            continue
        L.append(f"| {k} | {v['n']} | {fmt(v['mean'], v['lo'], v['hi'])} | {fmt(v['prior'])} |")
    L.append("")

    Path(a.md).write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n[wrote] {a.out}  {a.md}")


if __name__ == "__main__":
    main()
