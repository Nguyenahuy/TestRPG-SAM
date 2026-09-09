# Gap analysis - post-hoc (no model re-run)

Kvasir-SEG, 999 queries, support `cju85rkbnlo1c08503uxcpax1`, A_ref = 17.11% of the image. All intervals are 95% percentile bootstrap.

## B1 - IoU stratified by GT polyp coverage

| stratum | n | RPG-SAM | OP-SAM | PerSAM | RPG-SAM M_prior |
|---|---|---|---|---|---|
| small (<5%) | 200 | 64.33 [59.77, 69.02] | 63.02 [58.64, 67.26] | 52.76 [47.01, 58.44] | 59.74 [55.48, 63.80] |
| medium (5-15%) | 423 | 80.15 [78.01, 82.18] | 78.40 [76.14, 80.42] | 69.96 [66.75, 72.97] | 77.91 [76.04, 79.68] |
| large (>15%) | 376 | 78.80 [76.59, 80.89] | 77.11 [75.01, 79.22] | 65.73 [62.82, 68.69] | 74.64 [72.71, 76.56] |

RPG-SAM large minus small: **14.46 [9.40, 19.81]** IoU points.

## B2 - complete-miss rate (IoU < 0.05)

| method | all images | small polyps only |
|---|---|---|
| RPG-SAM | 3.00 [2.00, 4.00]% (n=999) | 12.50 [8.00, 17.50]% (n=200) |
| OP-SAM | 2.80 [1.80, 3.90]% (n=999) | 10.50 [6.50, 14.50]% (n=200) |
| PerSAM | 6.41 [4.90, 8.01]% (n=999) | 28.00 [22.00, 34.50]% (n=200) |

## C2 - IoU by quartile of GT solidity

Solidity quartile cuts: 0.948 / 0.982 / 0.996

| quartile | n | solidity | S_geo(GT) | scale(GT) | GT area% | RPG-SAM | OP-SAM | PerSAM | M_prior |
|---|---|---|---|---|---|---|---|---|---|
| Q1 (least convex) | 250 | 0.892 | 0.730 | 0.818 | 22.3 | 70.38 [67.34, 73.44] | 67.80 [64.53, 70.81] | 56.18 [52.19, 60.02] | 67.02 [64.17, 69.65] |
| Q2 | 250 | 0.967 | 0.701 | 0.725 | 18.0 | 78.88 [75.91, 81.66] | 78.58 [75.74, 81.31] | 65.14 [61.23, 68.96] | 75.44 [72.69, 77.97] |
| Q3 | 249 | 0.990 | 0.636 | 0.643 | 13.6 | 80.89 [77.96, 83.68] | 79.01 [76.03, 81.78] | 73.05 [69.06, 76.83] | 77.09 [74.16, 79.77] |
| Q4 (most convex) | 250 | 1.001 | 0.407 | 0.407 | 7.7 | 75.76 [72.23, 79.08] | 73.96 [70.50, 77.29] | 65.34 [60.80, 69.71] | 72.64 [69.48, 75.89] |

RPG-SAM Q4 minus Q1: **5.39 [0.79, 9.98]** IoU points.

OP-SAM Q4 minus Q1: **6.15 [1.55, 10.68]** IoU points.

PerSAM Q4 minus Q1: **9.15 [3.29, 14.96]** IoU points.

### Scale Consensus applied to the ground truth itself

`scale = min(1, A/A_ref)` damps **67.6%** of the ground-truth masks (mean factor 0.649); 37.1% are damped below 0.5. A perfect candidate is penalised on those images purely for being smaller than the support polyp.

## A2 - does the final mask track M_prior?

n = 999. Pearson r(IoU(M_prior,GT), IoU(final,GT)) = **0.923** [0.903, 0.940], Spearman rho = **0.807** [0.777, 0.833].

Share of the remaining headroom `(final - prior) / (1 - prior)` that PIR+SAM2 closes, by how good the prior was:

| prior IoU band | n | mean prior | mean final | recovery |
|---|---|---|---|---|
| [0.0,0.1) | 33 | 3.39 | 2.29 | -1.18 [-2.08, -0.44]% |
| [0.1,0.3) | 57 | 20.36 | 19.19 | -1.25 [-4.14, 1.68]% |
| [0.3,0.5) | 53 | 39.98 | 48.04 | 14.16 [5.71, 23.25]% |
| [0.5,0.7) | 136 | 60.85 | 67.36 | 16.77 [10.90, 22.61]% |
| [0.7,0.9) | 528 | 82.64 | 86.60 | 23.28 [19.68, 26.69]% |
| [0.9,1.0) | 192 | 92.02 | 92.68 | 7.25 [1.05, 13.22]% |

## A3 - where the failures actually come from

Of the 30 images RPG-SAM misses completely (IoU < 0.05), **21 (70.0%)** already had an empty `M_prior`. Their median GT covers 2.26% of the image.

Across all 999 queries the PIR loop improves on `M_prior` in 63.7% of images (mean +0.080 IoU), **degrades it in 21.8%** (mean -0.077), and leaves it unchanged in 14.5%. On 8 images a prior above 0.5 IoU is driven more than 0.25 below itself by the loop's negative prompts.

Cases for the qualitative panels (`scripts/visualize.py --names ...`):

- upstream miss: `cju0tl3uz8blh0993wxvn7ly3,cju18gzrq18zw0878wbf4ftw6,cju1egx9pvz2n0988eoy8jp23,cju1fm3id6gl50801r3fok20c`
- loop destroyed a good prior: `cju5b9oyda4yr0850g9viziyv,cju33belnbyhm0878yxl42233,cju2nbdpmlmcj0993s1cht0dz,cju7d2q1k27nf08715zshsckt`

## Phase 4 - size x solidity, cross-controlled

Solidity median = 0.982

| cell | n | RPG-SAM IoU | M_prior IoU |
|---|---|---|---|
| small (<5%) x flat (sol<=med) | 45 | 64.40 [54.39, 73.77] | 60.34 |
| small (<5%) x convex (sol>med) | 155 | 64.32 [58.75, 69.63] | 59.56 |
| large (>=5%) x flat (sol<=med) | 455 | 75.64 [73.55, 77.73] | 72.31 |
| large (>=5%) x convex (sol>med) | 344 | 84.63 [82.69, 86.45] | 81.75 |
