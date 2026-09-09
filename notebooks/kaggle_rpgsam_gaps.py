"""Kaggle driver for the RPG-SAM gap study.

Plain .py so it diffs cleanly in git; `scripts/make_notebook.py` converts it to
`kaggle_rpgsam_gaps.ipynb`. Each `# %%` block is one notebook cell.

A1, C3 and the A3 panels already ran on a local GPU and their JSON ships inside the repo,
so the resumable driver skips them. What is left for Kaggle is C4 (10 runs) and B3 (3),
about 55 GPU-minutes, plus the optional full-999 re-run of C3.
"""

# %% [markdown]
# # RPG-SAM: testing three claimed research gaps
#
# Gap **A** error propagation RWPM -> GAS -> PIR, gap **B** false negatives / the ceiling
# imposed by `M_prior`, gap **C** the geometric assumptions in `S_geo`.
#
# | phase | experiments | where |
# |---|---|---|
# | 1-2 | B1, B2, C2, A2, A3, Phase 4 | CPU, runs here in ~1 min |
# | 3 | A1 (13 runs), C3 (11), A3 panels | **already done**, JSON ships in the repo |
# | 3 | **C4 (10 runs), B3 (3)** | **this notebook, ~55 min** |
# | 3 | C3 at the full 999 queries (optional) | ~2.5 h, tightens the CIs ~1.8x |

# %%
# ---- 1. Repo + dependencies -------------------------------------------------
# Replace with your fork.
REPO = "https://github.com/YOUR-USERNAME/rpg-sam-gaps.git"

import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO_DIR = WORK / "rpg-sam-gaps"

if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(REPO_DIR)], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))
print("cwd:", os.getcwd())

# %%
# SAM2 and scikit-image are the only things Kaggle's image lacks.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "git+https://github.com/facebookresearch/sam2.git", "scikit-image"],
               check=True)

# %%
# SAM2 resolves its model config through hydra, from a search path registered by the
# installed package. Check that now rather than an hour into the sweep.
import importlib

importlib.invalidate_caches()
try:
    from sam2.build_sam import build_sam2  # noqa: F401
    print("sam2 import OK")
except Exception as e:  # pragma: no cover - environment probe
    print("sam2 setup problem:", type(e).__name__, e)
    print("Fallback: clone and install editable, then restart the kernel:")
    print("  !git clone https://github.com/facebookresearch/sam2.git /kaggle/working/sam2")
    print("  !pip install -q -e /kaggle/working/sam2")

# %%
# ---- 2. Weights -------------------------------------------------------------
CKPT = WORK / "sam2.1_hiera_large.pt"
if not CKPT.exists():
    subprocess.run(
        ["wget", "-q", "-O", str(CKPT),
         "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"],
        check=True)
print("SAM2 checkpoint:", CKPT, f"{CKPT.stat().st_size / 1e6:.0f} MB")
# DINOv2 comes from torch.hub on first use, so Internet must be ON.

# %%
# ---- 3. Point the code at the data + weights --------------------------------
# Add Kvasir-SEG as a Kaggle input dataset; the tree must contain images/ and masks/.
KVASIR = None
for cand in sorted(Path("/kaggle/input").glob("*")):
    for sub in [cand, *cand.glob("*"), *cand.glob("*/*")]:
        if (sub / "images").is_dir() and (sub / "masks").is_dir():
            KVASIR = sub
            break
    if KVASIR:
        break
assert KVASIR is not None, "Kvasir-SEG not found under /kaggle/input"

os.environ["RPGSAM_DATA"] = str(KVASIR.parent)
os.environ["RPGSAM_CKPT"] = str(CKPT)
print("data:", KVASIR, "|", len(list((KVASIR / "images").iterdir())), "images")

# %%
# ---- 4. Smoke test (~1 min) -------------------------------------------------
# Confirms DINOv2 + SAM2 + the data all load before committing GPU time.
subprocess.run([sys.executable, "scripts/run_eval.py", "--limit", "8",
                "--out", "outputs/smoke", "--tag", "smoke"], check=True)

# %%
# ---- 5. Phase 1-2: post-hoc gap analysis (CPU, ~1 min) ----------------------
# Reads the per-image JSON committed in outputs/tables/, so B1, B2, C2, A2, A3 and the
# Phase-4 cross-tab reproduce without touching the GPU.
subprocess.run([sys.executable, "scripts/gt_stats.py"], check=True)
subprocess.run([sys.executable, "scripts/analyze_gaps.py"], check=True)
print(Path("outputs/analysis/GAPS.md").read_text())

# %%
# ---- 6. What still needs a GPU ---------------------------------------------
# The driver skips any tag whose JSON is already on disk, so this only runs what the
# local machine did not. Check the plan before spending the quota.
subprocess.run([sys.executable, "scripts/run_gaps.py", "--group", "all", "--dry-run"],
               check=True)

# %%
# ---- 7. C4: support-image / A_ref sensitivity (10 runs, ~50 min) ------------
# Supports picked at evenly spaced percentiles of GT polyp coverage (0.47% -> 81.18%).
# Because A_ref *is* the support polyp's area, this doubles as an A_ref sweep -- which
# C3 identified as the parameter that actually drives GAS.
subprocess.run([sys.executable, "scripts/run_gaps.py", "--group", "C4"], check=True)

# %%
# ---- 8. B3: the extreme-size subset (3 runs, ~5 min) -----------------------
subprocess.run([sys.executable, "scripts/run_gaps.py", "--group", "B3"], check=True)

# %%
# ---- 9. Report --------------------------------------------------------------
subprocess.run([sys.executable, "scripts/report_gaps.py"], check=True)
print(Path("outputs/analysis/GAP_REPORT.md").read_text())

# %%
# ---- 10. Keep the results ---------------------------------------------------
# /kaggle/working is what "Save Version" persists.
subprocess.run(["zip", "-qr", str(WORK / "rpgsam_gap_outputs.zip"), "outputs"], check=True)
print("saved:", WORK / "rpgsam_gap_outputs.zip")

# %% [markdown]
# ## Optional: re-run C3 at the full 999 queries
#
# The local C3 sweep used 300 queries and every paired confidence interval crossed zero,
# so it could not separate the `S_geo` variants. 999 queries tightens the intervals by
# about 1.8x. Costs ~2.5 GPU-hours, so run it in its own session.
#
# ```python
# subprocess.run([sys.executable, "scripts/run_gaps.py",
#                 "--group", "C3", "--c3-limit", "0", "--force"])
# ```
#
# `--force` is required: without it the 300-query JSON already on disk would be kept.
# Point `--out` somewhere else if you want to keep both.
