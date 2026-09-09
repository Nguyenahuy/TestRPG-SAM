"""Render outputs/tables/*.json against the published RPG-SAM numbers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "tables"
OPSAM_TAB = ROOT / "outputs" / "tables"

# Kvasir column of the paper's Table 1
PAPER_T1 = {
    "SEGIC": (22.23, 32.81),
    "PerSAM": (26.01, 35.38),
    "OPSAM": (63.95, 71.43),
    "Matcher": (71.71, 78.91),
    "ProtoSAM": (73.09, 81.54),
    "RPG-SAM": (78.65, 85.65),
}

PAPER_T2 = [
    ("- - - -", "t2_row1_none", 63.68, 74.94, 0.8743),
    ("BG - - -", "t2_row2_bg", 67.89, 78.72, 0.8921),
    ("BG RWPM - -", "t2_row3_rwpm", 71.27, 81.05, 0.9068),
    ("BG RWPM GAS -", "t2_row4_gas", 75.25, 83.64, 0.9068),
    ("BG RWPM GAS PIR", "t2_row5_full", 78.65, 85.65, 0.9068),
]

PAPER_T3 = {
    "slic": [("K=5 m=5", "t3_slic_k5_m5", 77.34, 84.52),
             ("K=10 m=5", "t3_slic_k10_m5", 78.28, 85.27),
             ("K=10 m=20", "t3_slic_k10_m20", 78.65, 85.65),
             ("K=10 m=50", "t3_slic_k10_m50", 78.34, 85.41),
             ("K=20 m=5", "t3_slic_k20_m5", 76.08, 83.21)],
    "pir": [("cov=.8 iou=.7", "t3_pir_c0.8_i0.7", 77.98, 85.20),
            ("cov=.8 iou=.8", "t3_pir_c0.8_i0.8", 78.11, 85.32),
            ("cov=.8 iou=.9", "t3_pir_c0.8_i0.9", 77.79, 85.12),
            ("cov=.9 iou=.8", "t3_pir_c0.9_i0.8", 78.65, 85.65),
            ("cov=.95 iou=.8", "t3_pir_c0.95_i0.8", 78.66, 85.60)],
    "gas": [("0.3-0.6", "t3_gas_0.3_0.6", 75.05, 82.57),
            ("0.4-0.6", "t3_gas_0.4_0.6", 77.15, 84.48),
            ("0.4-0.7", "t3_gas_0.4_0.7", 78.65, 85.65),
            ("0.5-0.7", "t3_gas_0.5_0.7", 77.26, 84.57),
            ("0.5-0.8", "t3_gas_0.5_0.8", 77.12, 84.41)],
}


def load(tag: str, base: Path = TAB):
    f = base / f"{tag}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())["summary"]


def fmt(s, key="IoU"):
    return f"{s[key]:.2f}" if s else "--"


def row(label, s, paper=None, ap=False):
    if s is None:
        cells = ["--", "--"] + (["--"] if ap else [])
    else:
        cells = [f"{s['IoU']:.2f}", f"{s['Dice']:.2f}"] + (
            [f"{s.get('auc_pr', float('nan')):.4f}"] if ap else [])
    pcells = []
    if paper:
        pcells = [f"{paper[0]:.2f}", f"{paper[1]:.2f}"] + (
            [f"{paper[2]:.4f}"] if ap and len(paper) > 2 else [])
    return "| " + " | ".join([label] + pcells + cells) + " |"


def main():
    out = ["# RPG-SAM reproduction results", ""]
    t1 = load("t1_rpgsam_kvasir")
    hard = load("t1_rpgsam_kvasir_hard")
    op = load("t1_opsam", OPSAM_TAB)
    per = load("t1_persam", OPSAM_TAB)
    orc = load("t1_oracle", OPSAM_TAB)

    out += ["## Table 1 - Kvasir-SEG", "",
            f"Support image fixed to the one the OP-SAM reproduction used "
            f"(`{t1['support'] if t1 else '?'}`, index {t1['support_idx'] if t1 else '?'}); "
            f"the other {t1['n'] if t1 else '?'} images are the query set.", "",
            "| method | paper IoU | paper Dice | ours IoU | ours Dice |",
            "|---|---|---|---|---|"]
    for name, p in PAPER_T1.items():
        s = {"RPG-SAM": t1, "OPSAM": op, "PerSAM": per}.get(name)
        out.append(row(name, s, p))
    out.append(row("Oracle (3 GT points)", orc, None))
    if hard:
        out += ["", f"Kvasir-H (extreme polyp sizes, n={hard['n']}): "
                    f"{hard['IoU']:.2f} IoU / {hard['Dice']:.2f} Dice "
                    f"(not a table in the RPG-SAM paper; the OP-SAM reproduction "
                    f"reports 57.93 / 66.49 there)."]

    out += ["", "## Table 2 - incremental ablation", ""]
    lim = load("t2_row5_full")
    out += [f"Query subset: first {lim['n'] if lim else '?'} images.", "",
            "| modules | paper IoU | paper Dice | paper AUC-PR | ours IoU | ours Dice | ours AUC-PR |",
            "|---|---|---|---|---|---|---|"]
    for label, tag, i, d, ap in PAPER_T2:
        out.append(row(label, load(tag), (i, d, ap), ap=True))

    out += ["", "## Table 3 - hyper-parameters", ""]
    for group, rows in PAPER_T3.items():
        out += [f"### {group.upper()}", "",
                "| setting | paper IoU | paper Dice | ours IoU | ours Dice |",
                "|---|---|---|---|---|"]
        for label, tag, i, d in rows:
            out.append(row(label, load(tag), (i, d)))
        out.append("")

    # ---- what GAS is actually worth against a fixed threshold -----------------
    gas_rows = [
        ("fixed tau = 0.4, no PIR", "t2b_row3_rwpm"),
        ("fixed tau = 0.7, no PIR", "t2_row3_rwpm"),
        ("GAS, no PIR", "t2_row4_gas"),
        ("fixed tau = 0.4 + PIR", "t2b_row5_full_fixed"),
        ("fixed tau = 0.7 + PIR", "t2c_row5_full_fixed07"),
        ("GAS + PIR (full method)", "t2_row5_full"),
    ]
    out += ["", "## GAS against a fixed threshold", "",
            "Same heatmap, same PIR settings; only the binarisation rule changes.", "",
            "| binarisation (300 queries) | IoU | Dice |", "|---|---|---|"]
    for label, tag in gas_rows:
        out.append(row(label, load(tag)))
    full_rows = [("fixed tau = 0.4 + PIR", "t2b_kvasir_fixed04"),
                 ("fixed tau = 0.7 + PIR", "t2c_kvasir_fixed07"),
                 ("GAS + PIR (full method)", "t1_rpgsam_kvasir")]
    out += ["", "| binarisation (999 queries) | IoU | Dice |", "|---|---|---|"]
    for label, tag in full_rows:
        out.append(row(label, load(tag)))

    seeds = [load(f"t4_support_s{i}") for i in range(5)]
    if any(seeds):
        vals = [s["IoU"] for s in seeds if s]
        out += ["## Support-image sensitivity (extra)", "",
                "| seed | support | IoU | Dice |", "|---|---|---|---|"]
        for i, s in enumerate(seeds):
            if s:
                out.append(f"| {i} | {s['support'][:16]} | {s['IoU']:.2f} | {s['Dice']:.2f} |")
        if vals:
            import statistics
            out += ["", f"mean {statistics.mean(vals):.2f}, "
                        f"sd {statistics.pstdev(vals):.2f}, "
                        f"range {min(vals):.2f}-{max(vals):.2f}"]

    dest = ROOT / "outputs" / "RESULTS.md"
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    sys.exit(main())
