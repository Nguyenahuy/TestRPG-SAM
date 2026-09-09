"""Turn the Phase-3 runs into the A1 / C3 / C4 / B3 verdict tables.

Reads whatever ``run_gaps.py`` has produced in ``outputs/gaps/`` -- missing groups are
skipped rather than fatal, so this can be run against a partial sweep. Each section ends
with the plan's own decision criterion evaluated against the numbers, so the verdict is
not left to be invented after the fact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(0)


def boot_mean(x, n=4000):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    m = x[RNG.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_paired_diff(a, b, n=4000):
    """Paired CI: the runs share the same query images, so pair them per image."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    m = d[RNG.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load(gaps: Path, tag: str):
    f = gaps / f"{tag}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return {"s": d["summary"], "p": {r["name"]: r for r in d.get("per_image", [])}}


def pct(v, lo=None, hi=None, d=2):
    if v is None or v != v:
        return "--"
    s = f"{100 * v:.{d}f}" if abs(v) <= 1.5 else f"{v:.{d}f}"
    if lo is not None:
        f = 100 if abs(v) <= 1.5 else 1
        s += f" [{f * lo:.{d}f}, {f * hi:.{d}f}]"
    return s


def subgroup(run, gt, pred, key="iou"):
    return [r[key] for n, r in run["p"].items()
            if n in gt and pred(gt[n]) and r.get(key) is not None]


def paired(run_a, run_b, gt, pred, key="iou"):
    names = [n for n in run_a["p"] if n in run_b["p"] and n in gt and pred(gt[n])]
    return ([run_a["p"][n][key] for n in names], [run_b["p"][n][key] for n in names])


# ---------------------------------------------------------------- sections

GEO_LABEL = {
    "full": "full Eq. 5 (paper)",
    "no_solidity": "(ii) drop Weighted Solidity",
    "no_scale": "(iii) drop Scale Consensus",
    "sym_scale": "(v) two-sided scale penalty",
    "perim": "(iv) perimeter convexity",
    "perim_sym": "(iv)+(v) both repairs",
}


def sec_C3(gaps, gt, L):
    modes = ["full", "no_solidity", "no_scale", "sym_scale", "perim", "perim_sym"]
    runs = {m: load(gaps, f"c3_geo_{m}") for m in modes}
    runs = {m: r for m, r in runs.items() if r}
    if not runs:
        return
    fixed = {t: load(gaps, f"c3_fixed_tau{t}") for t in ("040", "070")}
    aref = {f: load(gaps, f"c3_aref_frac{f}") for f in ("05", "10")}
    aref = {k: v for k, v in aref.items() if v}
    combo = load(gaps, "c3_sym_aref10")

    sol_med = float(np.median([x["solidity"] for x in gt.values()]))

    def flat(g):
        return g["solidity"] <= sol_med

    def conv(g):
        return g["solidity"] > sol_med

    def small(g):
        return g["area_ratio"] < 0.05

    L.append("## C3 - ablating the S_geo formula\n")
    L.append("Everything upstream and downstream is byte-identical; only the candidate "
             "score of Eq. 5 changes. `tau` is the mean selected threshold: the paper's "
             "scan band is [0.4, 0.7], so a mean near 0.40 means GAS is taking the bottom "
             "of the band on almost every image.\n")
    L.append("| variant | n | IoU | Dice | recall | mean tau | flat subgroup | convex subgroup | small (<5%) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m, r in runs.items():
        s = r["s"]
        fl = boot_mean(subgroup(r, gt, flat))
        cv = boot_mean(subgroup(r, gt, conv))
        sm = boot_mean(subgroup(r, gt, small))
        L.append(f"| {GEO_LABEL.get(m, m)} | {s['n']} | {s['IoU']:.2f} | {s['Dice']:.2f} | "
                 f"{s.get('recall', float('nan')):.2f} | {s.get('tau_mean', 0):.3f} | "
                 f"{pct(fl[0], fl[1], fl[2])} | {pct(cv[0], cv[1], cv[2])} | "
                 f"{pct(sm[0], sm[1], sm[2])} |")
    for f, r in aref.items():
        if r:
            s = r["s"]
            fl = boot_mean(subgroup(r, gt, flat))
            cv = boot_mean(subgroup(r, gt, conv))
            sm = boot_mean(subgroup(r, gt, small))
            L.append(f"| A_ref = {int(f)}% of image (not the support) | {s['n']} | "
                     f"{s['IoU']:.2f} | {s['Dice']:.2f} | "
                     f"{s.get('recall', float('nan')):.2f} | {s.get('tau_mean', 0):.3f} | "
                     f"{pct(fl[0], fl[1], fl[2])} | {pct(cv[0], cv[1], cv[2])} | "
                     f"{pct(sm[0], sm[1], sm[2])} |")
    if combo:
        s = combo["s"]
        fl = boot_mean(subgroup(combo, gt, flat))
        cv = boot_mean(subgroup(combo, gt, conv))
        sm = boot_mean(subgroup(combo, gt, small))
        L.append(f"| **repaired GAS: two-sided + A_ref = 10%** | {s['n']} | "
                 f"{s['IoU']:.2f} | {s['Dice']:.2f} | "
                 f"{s.get('recall', float('nan')):.2f} | {s.get('tau_mean', 0):.3f} | "
                 f"{pct(fl[0], fl[1], fl[2])} | {pct(cv[0], cv[1], cv[2])} | "
                 f"{pct(sm[0], sm[1], sm[2])} |")
    for t, r in fixed.items():
        if r:
            L.append(f"| fixed tau = 0.{t[1:]} (no scan) | {r['s']['n']} | "
                     f"{r['s']['IoU']:.2f} | {r['s']['Dice']:.2f} | "
                     f"{r['s'].get('recall', float('nan')):.2f} | 0.{t[1:]} | -- | -- | -- |")
    L.append("")

    base = runs.get("full")
    if base:
        L.append("Paired differences against the paper's Eq. 5, on the flat subgroup "
                 "(the group C3 predicts should benefit) and on the convex subgroup "
                 "(where a repair must not cost anything):\n")
        L.append("| variant | flat: delta IoU | convex: delta IoU | all: delta IoU |")
        L.append("|---|---|---|---|")
        for m, r in runs.items():
            if m == "full":
                continue
            row = []
            for pred in (flat, conv, lambda g: True):
                a, b = paired(r, base, gt, pred)
                row.append(pct(*boot_paired_diff(a, b)))
            L.append(f"| {GEO_LABEL.get(m, m)} | " + " | ".join(row) + " |")
        L.append("")
        L.append("**Criterion (plan C3):** variant (ii)/(iv) improves clearly on the flat "
                 "group without a large loss on the compact group.\n")


def sec_A1(gaps, gt, L):
    tags = [("a1_baseline", "baseline (full RWPM)"),
            ("a1_no_bg", "(i) no background suppression"),
            ("a1_no_reliability", "(ii) uniform W_k"),
            ("a1_no_bg_no_rel", "(i)+(ii)"),
            ("a1_fault_specular", "(iii) specular support"),
            ("a1_fault_blur", "(iii) blurred support"),
            ("a1_fault_erode", "(iii) eroded support mask"),
            ("a1_fault_dilate", "(iii) dilated support mask"),
            ("a1_fault_shift", "(iii) shifted support mask")]
    runs = [(lbl, load(gaps, t)) for t, lbl in tags]
    runs = [(lbl, r) for lbl, r in runs if r]
    if not runs:
        return

    L.append("## A1 - fault injection at RWPM, GAS and PIR untouched\n")
    L.append("`recovery = mean over images of (IoU_final - IoU_prior) / (1 - IoU_prior)`: "
             "the share of the headroom still left after RWPM that SAM2+PIR actually "
             "closes. It is the plan's \"% phuc hoi\", made per-image so it has a CI.\n")
    L.append("| RWPM variant | n | M_prior IoU | final IoU | delta | prior recall | recovery |")
    L.append("|---|---|---|---|---|---|---|")
    base_rec = None
    for lbl, r in runs:
        s = r["s"]
        pi = np.array([x["prior_iou"] for x in r["p"].values()])
        fi = np.array([x["iou"] for x in r["p"].values()])
        rec = boot_mean((fi - pi) / np.maximum(1 - pi, 1e-6))
        if base_rec is None:
            base_rec = rec[0]
        L.append(f"| {lbl} | {s['n']} | {s['prior']['IoU']:.2f} | {s['IoU']:.2f} | "
                 f"{s['IoU'] - s['prior']['IoU']:+.2f} | "
                 f"{s['prior'].get('recall', float('nan')):.2f} | "
                 f"{pct(rec[0], rec[1], rec[2])}% |")
    L.append("")
    L.append("**Criterion (plan A1):** recovery stays low (<30%) under the degraded "
             "variants, i.e. PIR does not undo an upstream error.\n")


def sec_C4(gaps, gt, L):
    files = sorted(gaps.glob("c4_sup_p*.json"))
    if not files:
        return
    rows = []
    for f in files:
        d = json.loads(f.read_text())["summary"]
        area = float(f.stem.split("_a")[-1])
        rows.append((area, d["support"], d["IoU"], d["Dice"],
                     d.get("recall", float("nan")), d["prior"]["IoU"], d["n"]))
    rows.sort()

    L.append("## C4 - sensitivity to the support image (and therefore to A_ref)\n")
    L.append("Supports chosen at evenly spaced percentiles of GT polyp coverage, so the "
             "sweep varies `A_ref` by construction: `A_ref` *is* the support polyp area.\n")
    L.append("| support polyp % of image | n | IoU | Dice | recall | M_prior IoU |")
    L.append("|---|---|---|---|---|---|")
    for area, name, iou, dice, rec, pi, n in rows:
        L.append(f"| {area:.2f} | {n} | {iou:.2f} | {dice:.2f} | {rec:.2f} | {pi:.2f} |")
    ious = np.array([r[2] for r in rows])
    areas = np.array([r[0] for r in rows])
    L.append("")
    L.append(f"mean {ious.mean():.2f}, sd {ious.std(ddof=1):.2f}, "
             f"range {ious.min():.2f}-{ious.max():.2f} "
             f"(spread {ious.max() - ious.min():.2f} IoU points).")
    if len(ious) > 2:
        r = float(np.corrcoef(np.log(areas), ious)[0, 1])
        L.append(f"Correlation between log(support polyp area) and final IoU: r = {r:.3f}.")
    L.append("\n**Criterion (plan C4):** large variance, especially for very small or "
             "very large support polyps.\n")


def sec_B3(gaps, gt, L):
    rows = [("same support, hard queries only", load(gaps, "b3_hardq_full")),
            ("same support, hard queries, M_prior only", load(gaps, "b3_hardq_prioronly")),
            ("support ALSO drawn from the hard subset", load(gaps, "b3_hardset_full"))]
    rows = [(lbl, r) for lbl, r in rows if r]
    if not rows:
        return
    L.append("## B3 - the extreme-size subset (Kvasir-H stand-in)\n")
    L.append("Coverage <= 3% or >= 50%. The first two rows keep the support image every "
             "other table uses and narrow only the *queries*, so 'harder queries' is the "
             "single variable. The third row changes the support as well, which is what a "
             "naive `--subset hard` does -- it is reported separately because its number "
             "is not comparable with Table 1.\n")
    L.append("| configuration | n | IoU | Dice | recall | M_prior IoU | AUC-PR |")
    L.append("|---|---|---|---|---|---|---|")
    for lbl, r in rows:
        s = r["s"]
        L.append(f"| {lbl} | {s['n']} | {s['IoU']:.2f} | {s['Dice']:.2f} | "
                 f"{s.get('recall', float('nan')):.2f} | {s['prior']['IoU']:.2f} | "
                 f"{s.get('auc_pr', float('nan')):.4f} |")
    L.append("")
    L.append("Piccolo and the real Kvasir-H split are behind registration, so this is the "
             "closest available stand-in and the OP-SAM published numbers (57.31 / 65.48) "
             "are **not** directly comparable to it.\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gaps", default="outputs/gaps")
    p.add_argument("--gt", default="outputs/analysis/gt_stats.json")
    p.add_argument("--md", default="outputs/analysis/GAP_REPORT.md")
    a = p.parse_args()

    gaps = Path(a.gaps)
    gt = json.loads(Path(a.gt).read_text())["stats"]

    L = ["# Phase 3 - the runs that needed a GPU\n",
         "Kvasir-SEG. All intervals are 95% percentile bootstrap; comparisons between two "
         "runs are paired per query image.\n"]
    sec_C3(gaps, gt, L)
    sec_A1(gaps, gt, L)
    sec_C4(gaps, gt, L)
    sec_B3(gaps, gt, L)

    Path(a.md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.md).write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n[wrote] {a.md}")


if __name__ == "__main__":
    main()
