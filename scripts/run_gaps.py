"""Driver for the GPU half of the gap study: A1, C3, C4 (and the B3 subset).

Every experiment is one ``run_eval.py`` invocation with a distinct tag, so a run is fully
described by its command line and lands in its own JSON. The driver is **resumable**: a
tag whose JSON already exists is skipped, so a Kaggle session that hits the time limit
can be restarted and will pick up where it stopped.

    python scripts/run_gaps.py --group all          # everything, ~4 GPU-hours
    python scripts/run_gaps.py --group C3 --limit 300
    python scripts/run_gaps.py --group all --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpgsam.paths import kvasir_root  # noqa: E402

from opsam.datasets import load_dataset, polyp_coverage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


# ---------------------------------------------------------------- experiment sets

def c3_runs(limit: int) -> list[tuple[str, list[str]]]:
    """C3 -- the S_geo ablation. Which term of Eq. 5 is doing the work, and does
    removing the convexity bias / adding the missing upper penalty help flat polyps?"""
    out = []
    for mode in ["full", "no_solidity", "no_scale", "sym_scale", "perim", "perim_sym"]:
        out.append((f"c3_geo_{mode}", ["--geo-mode", mode]))
    # C1 established that A_ref is the *support* polyp's area, so the reference scale is
    # whatever the single annotated image happened to contain. These ask what that costs.
    out.append(("c3_aref_frac05", ["--aref", "frac", "--aref-frac", "0.05"]))
    out.append(("c3_aref_frac10", ["--aref", "frac", "--aref-frac", "0.10"]))
    # The two-sided penalty peaks where A == A_ref, so it is only meaningful if A_ref is
    # a sensible reference scale. With A_ref = the support polyp (17.1% here vs an 11.4%
    # median GT) it would pull towards over-segmentation for a reason that has nothing to
    # do with the fix. This row is the honest test of a repaired GAS.
    out.append(("c3_sym_aref10", ["--geo-mode", "sym_scale",
                                  "--aref", "frac", "--aref-frac", "0.10"]))
    # the reference points GAS has to beat
    out.append(("c3_fixed_tau040", ["--fixed-tau", "0.4"]))
    out.append(("c3_fixed_tau070", ["--fixed-tau", "0.7"]))
    return out


def a1_runs(limit: int) -> list[tuple[str, list[str]]]:
    """A1 -- degrade RWPM in a controlled way, leave GAS+PIR untouched, and ask how
    much of the damage PIR undoes."""
    out = [("a1_baseline", [])]
    out.append(("a1_no_bg", ["--no-bg-suppress"]))                # (i)
    out.append(("a1_no_reliability", ["--no-reliability"]))       # (ii)
    out.append(("a1_no_bg_no_rel", ["--no-bg-suppress", "--no-reliability"]))
    for f in ["specular", "shift", "erode", "dilate", "blur"]:    # (iii)
        out.append((f"a1_fault_{f}", ["--fault", f]))
    # the same degradations scored *before* SAM2, so PIR's contribution is isolated
    out.append(("a1_baseline_prioronly", ["--prior-only"]))
    out.append(("a1_no_bg_prioronly", ["--no-bg-suppress", "--prior-only"]))
    out.append(("a1_fault_shift_prioronly", ["--fault", "shift", "--prior-only"]))
    out.append(("a1_fault_specular_prioronly", ["--fault", "specular", "--prior-only"]))
    return out


def c4_supports(root: str, n: int = 10) -> list[tuple[int, str, float]]:
    """C4 -- support images spanning the polyp-size range, chosen deterministically at
    evenly spaced percentiles of GT coverage rather than at random, so the *reason* a
    draw is bad is legible."""
    samples = load_dataset("kvasir", root)
    cov = np.array([polyp_coverage(s) for s in samples])
    order = np.argsort(cov)
    picks = np.linspace(0, len(order) - 1, n).round().astype(int)
    return [(int(order[p]), samples[order[p]].name, float(cov[order[p]])) for p in picks]


def c4_runs(root: str, n: int) -> list[tuple[str, list[str]]]:
    out = []
    for rank, (idx, name, cov) in enumerate(c4_supports(root, n)):
        out.append((f"c4_sup_p{rank:02d}_a{cov * 100:05.2f}", ["--support-index", str(idx)]))
    return out


def b3_runs(limit: int) -> list[tuple[str, list[str]]]:
    """B3 -- the extreme-size subset, the closest stand-in for Kvasir-H available here."""
    # --query-subset, not --subset: the support image stays the one every other run uses,
    # so "harder queries" is the only variable. Filtering the whole dataset instead would
    # also change the support and make the number incomparable with Table 1.
    return [("b3_hardq_full", ["--query-subset", "hard"]),
            ("b3_hardq_prioronly", ["--query-subset", "hard", "--prior-only"]),
            # kept for contrast: the support is drawn from the hard subset too
            ("b3_hardset_full", ["--subset", "hard", "--support-seed", "0"])]


GROUPS = {"C3": c3_runs, "A1": a1_runs, "B3": b3_runs}


def run_a3_panels(root: str, out: Path) -> None:
    """A3 -- render the two failure modes ``analyze_gaps.py`` picked out by name, rather
    than the first n images, so the panels show the cases the numbers point at."""
    gaps = ROOT / "outputs" / "analysis" / "gaps.json"
    if not gaps.exists():
        print("[A3] run scripts/analyze_gaps.py first; skipping panels")
        return
    a3 = json.loads(gaps.read_text()).get("A3", {})
    for key, tag in [("examples_miss", "a3_upstream_miss"),
                     ("examples_ruined", "a3_loop_ruined_prior")]:
        names = a3.get(key, [])[:4]
        if not names:
            continue
        cmd = [PY, str(ROOT / "scripts" / "visualize.py"), "--root", root,
               "--names", ",".join(names), "--tag", tag,
               "--out", str(ROOT / "outputs" / "figures")]
        print(f"\n=== A3 panels: {tag} ({len(names)} cases)", flush=True)
        subprocess.run(cmd, cwd=ROOT)


# ---------------------------------------------------------------- driver

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="all",
                   choices=["all", "A1", "A3", "B3", "C3", "C4"])
    p.add_argument("--limit", type=int, default=300,
                   help="queries per A1 run (0 = all 999)")
    p.add_argument("--c3-limit", type=int, default=0,
                   help="queries per C3 run; C3 is the headline ablation so it defaults "
                        "to the full query set")
    p.add_argument("--c4-n", type=int, default=10, help="number of C4 support draws")
    p.add_argument("--c4-limit", type=int, default=300)
    p.add_argument("--root", default=None)
    p.add_argument("--out", default="outputs/gaps")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="re-run tags that already exist")
    a = p.parse_args()

    root = a.root or str(kvasir_root())
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, list[str], int]] = []
    groups = ["A1", "B3", "C3", "C4", "A3"] if a.group == "all" else [a.group]
    if "A3" in groups:
        groups = [g for g in groups if g != "A3"]
        if not a.dry_run:
            run_a3_panels(root, out)
        if not groups:
            return
    for g in groups:
        if g == "C4":
            jobs += [(t, args, a.c4_limit) for t, args in c4_runs(root, a.c4_n)]
        elif g == "C3":
            jobs += [(t, args, a.c3_limit) for t, args in c3_runs(a.c3_limit)]
        elif g == "B3":
            jobs += [(t, args, 0) for t, args in b3_runs(0)]      # the subset is only 104
        else:
            jobs += [(t, args, a.limit) for t, args in GROUPS[g](a.limit)]

    todo = [j for j in jobs if a.force or not (out / f"{j[0]}.json").exists()]
    print(f"[plan] {len(jobs)} runs in group '{a.group}', {len(todo)} still to do "
          f"({len(jobs) - len(todo)} already on disk)")

    t_all = time.time()
    for i, (tag, extra, limit) in enumerate(todo, 1):
        cmd = [PY, str(ROOT / "scripts" / "run_eval.py"),
               "--root", root, "--out", str(out), "--tag", tag, "--save-per-image"]
        if limit:
            cmd += ["--limit", str(limit)]
        cmd += extra
        print(f"\n=== [{i}/{len(todo)}] {tag}\n    {' '.join(cmd[1:])}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"[FAIL] {tag} exited {r.returncode}", flush=True)
            continue
        d = json.loads((out / f"{tag}.json").read_text())["summary"]
        print(f"[done] {tag}: IoU {d['IoU']:.2f} Dice {d['Dice']:.2f} "
              f"prior {d['prior']['IoU']:.2f} ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n[all] {time.time() - t_all:.0f}s total")

    rows = []
    for tag, _, _ in jobs:
        f = out / f"{tag}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())["summary"]
        rows.append({"tag": tag, "n": d["n"], "IoU": round(d["IoU"], 2),
                     "Dice": round(d["Dice"], 2), "recall": round(d.get("recall", 0), 2),
                     "prior_IoU": round(d["prior"]["IoU"], 2),
                     "prior_recall": round(d["prior"].get("recall", 0), 2),
                     "s_geo": round(d.get("s_geo_mean", 0), 4),
                     "tau_mean": round(d.get("tau_mean", 0), 4),
                     "support": d["support"], "cfg": d["cfg"]})
    (out / "_index.json").write_text(json.dumps(rows, indent=1))
    print(f"[index] {out / '_index.json'} ({len(rows)} runs)")
    for r in rows:
        print(f"  {r['tag']:34s} n={r['n']:4d} IoU={r['IoU']:6.2f} "
              f"prior={r['prior_IoU']:6.2f} tau={r['tau_mean']:.3f}")


if __name__ == "__main__":
    main()
