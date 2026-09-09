# Phase 3 - the runs that needed a GPU

Kvasir-SEG. All intervals are 95% percentile bootstrap; comparisons between two runs are paired per query image.

## C3 - ablating the S_geo formula

Everything upstream and downstream is byte-identical; only the candidate score of Eq. 5 changes. `tau` is the mean selected threshold: the paper's scan band is [0.4, 0.7], so a mean near 0.40 means GAS is taking the bottom of the band on almost every image.

| variant | n | IoU | Dice | recall | mean tau | flat subgroup | convex subgroup | small (<5%) |
|---|---|---|---|---|---|---|---|---|
| full Eq. 5 (paper) | 300 | 75.32 | 82.33 | 87.85 | 0.410 | 73.92 [70.14, 77.70] | 76.86 [72.01, 81.35] | 60.81 [52.47, 68.58] |
| (ii) drop Weighted Solidity | 300 | 75.15 | 82.14 | 88.14 | 0.400 | 74.05 [70.11, 77.83] | 76.37 [71.33, 80.83] | 60.84 [52.54, 68.93] |
| (iii) drop Scale Consensus | 300 | 76.24 | 83.11 | 85.09 | 0.459 | 74.11 [70.15, 77.89] | 78.57 [74.01, 82.78] | 65.04 [57.21, 72.39] |
| (v) two-sided scale penalty | 300 | 75.10 | 82.63 | 84.78 | 0.463 | 73.07 [69.54, 76.33] | 77.34 [72.61, 81.56] | 63.00 [55.11, 70.26] |
| (iv) perimeter convexity | 300 | 75.85 | 82.99 | 86.96 | 0.422 | 74.44 [70.85, 77.92] | 77.39 [72.65, 81.66] | 61.42 [53.60, 69.24] |
| (iv)+(v) both repairs | 300 | 75.73 | 83.09 | 85.44 | 0.454 | 74.11 [70.89, 77.38] | 77.50 [72.97, 81.88] | 62.86 [55.33, 70.36] |
| A_ref = 5% of image (not the support) | 300 | 77.04 | 83.92 | 86.79 | 0.435 | 75.68 [71.98, 79.20] | 78.52 [73.81, 82.63] | 65.63 [57.61, 73.12] |
| A_ref = 10% of image (not the support) | 300 | 76.31 | 83.27 | 87.22 | 0.422 | 74.85 [71.09, 78.35] | 77.91 [73.25, 82.07] | 61.97 [54.02, 69.62] |
| **repaired GAS: two-sided + A_ref = 10%** | 300 | 73.96 | 81.79 | 80.94 | 0.529 | 71.50 [67.77, 74.93] | 76.66 [72.14, 81.04] | 63.89 [56.00, 70.98] |
| fixed tau = 0.40 (no scan) | 300 | 75.15 | 82.14 | 88.14 | 0.40 | -- | -- | -- |
| fixed tau = 0.70 (no scan) | 300 | 65.45 | 74.84 | 69.52 | 0.70 | -- | -- | -- |

Paired differences against the paper's Eq. 5, on the flat subgroup (the group C3 predicts should benefit) and on the convex subgroup (where a repair must not cost anything):

| variant | flat: delta IoU | convex: delta IoU | all: delta IoU |
|---|---|---|---|
| (ii) drop Weighted Solidity | 0.13 [-0.24, 0.49] | -0.50 [-1.18, -0.01] | -0.17 [-0.54, 0.14] |
| (iii) drop Scale Consensus | 0.19 [-1.88, 2.47] | 1.71 [-0.78, 4.48] | 0.91 [-0.76, 2.61] |
| (v) two-sided scale penalty | -0.85 [-3.11, 1.48] | 0.48 [-0.58, 1.97] | -0.22 [-1.55, 1.22] |
| (iv) perimeter convexity | 0.52 [-1.13, 2.19] | 0.53 [-0.23, 1.80] | 0.52 [-0.44, 1.58] |
| (iv)+(v) both repairs | 0.19 [-1.76, 2.23] | 0.63 [-0.34, 2.04] | 0.40 [-0.75, 1.66] |

**Criterion (plan C3):** variant (ii)/(iv) improves clearly on the flat group without a large loss on the compact group.

## A1 - fault injection at RWPM, GAS and PIR untouched

`recovery = mean over images of (IoU_final - IoU_prior) / (1 - IoU_prior)`: the share of the headroom still left after RWPM that SAM2+PIR actually closes. It is the plan's "% phuc hoi", made per-image so it has a CI.

| RWPM variant | n | M_prior IoU | final IoU | delta | prior recall | recovery |
|---|---|---|---|---|---|---|
| baseline (full RWPM) | 300 | 71.41 | 75.32 | +3.91 | 86.52 | 19.32 [14.85, 23.48]% |
| (i) no background suppression | 300 | 68.56 | 73.33 | +4.77 | 88.55 | 22.69 [19.06, 26.15]% |
| (ii) uniform W_k | 300 | 69.31 | 72.91 | +3.60 | 90.50 | 17.66 [13.68, 21.59]% |
| (i)+(ii) | 300 | 63.66 | 67.09 | +3.43 | 91.84 | 17.13 [13.62, 20.75]% |
| (iii) specular support | 300 | 70.19 | 74.39 | +4.20 | 86.82 | 20.87 [16.67, 24.96]% |
| (iii) blurred support | 300 | 69.66 | 76.49 | +6.83 | 84.86 | 25.35 [21.15, 29.44]% |
| (iii) eroded support mask | 300 | 65.74 | 69.12 | +3.38 | 90.00 | 13.52 [9.16, 17.64]% |
| (iii) dilated support mask | 300 | 69.53 | 73.48 | +3.95 | 89.00 | 19.39 [15.12, 23.50]% |
| (iii) shifted support mask | 300 | 64.61 | 70.03 | +5.42 | 85.10 | 19.29 [15.34, 23.15]% |

**Criterion (plan A1):** recovery stays low (<30%) under the degraded variants, i.e. PIR does not undo an upstream error.
