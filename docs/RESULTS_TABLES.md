
**Headline metrics (seed 1 / seed 2 / seed 3; temperature-scaled with the frozen source-validation T)**

| Set | Model | Accuracy | Macro-F1 | ECE (15 bins) | NLL |
|---|---|---|---|---|---|
| PlantVillage test | Scratch CNN | 0.987 / 0.988 / 0.987 | 0.982 / 0.983 / 0.982 | 0.004 / 0.004 / 0.005 | 0.056 / 0.050 / 0.049 |
| PlantVillage test | ResNet-18 | 0.994 / 0.996 / 0.995 | 0.992 / 0.995 / 0.995 | 0.003 / 0.003 / 0.004 | 0.031 / 0.039 / 0.037 |
| PlantDoc | Scratch CNN | 0.152 / 0.154 / 0.143 | 0.118 / 0.115 / 0.107 | 0.669 / 0.649 / 0.696 | 9.774 / 9.044 / 10.390 |
| PlantDoc | ResNet-18 | 0.167 / 0.167 / 0.149 | 0.128 / 0.134 / 0.111 | 0.676 / 0.744 / 0.734 | 8.766 / 14.243 / 11.593 |

**PlantDoc 95% cluster-bootstrap intervals, per seed (10,000 resamples; clusters = near-duplicate groups)**

| Model | Accuracy [95% CI] seeds 1/2/3 | Macro-F1 [95% CI] seeds 1/2/3 |
|---|---|---|
| Scratch CNN | 0.152 [0.137, 0.169], 0.154 [0.139, 0.170], 0.143 [0.128, 0.159] | 0.118 [0.105, 0.131], 0.115 [0.103, 0.127], 0.107 [0.095, 0.119] |
| ResNet-18 | 0.167 [0.150, 0.183], 0.167 [0.150, 0.183], 0.149 [0.134, 0.165] | 0.128 [0.114, 0.141], 0.134 [0.120, 0.147], 0.111 [0.098, 0.124] |

**Calibration on PlantDoc (seeds 1/2/3). Brier = sum over classes, range 0–2**

| Model | ECE before scaling | ECE after scaling | Mean confidence − accuracy | Brier |
|---|---|---|---|---|
| Scratch CNN | 0.596 / 0.582 / 0.610 | 0.669 / 0.649 / 0.696 | 0.669 / 0.649 / 0.696 | 1.454 / 1.426 / 1.490 |
| ResNet-18 | 0.646 / 0.645 / 0.666 | 0.676 / 0.744 / 0.734 | 0.676 / 0.744 / 0.734 | 1.443 / 1.544 / 1.537 |

**Abstention on PlantDoc at the frozen thresholds (seeds 1/2/3)**

| Model | Rule | Coverage | Selective risk (error rate among answered) |
|---|---|---|---|
| Scratch CNN | R-cov (90% coverage on val_cal) | 0.188 / 0.179 / 0.181 | 0.779 / 0.728 / 0.764 |
| Scratch CNN | R-risk (≤1% risk on val_cal) | 0.972 / 0.886 / 0.990 | 0.845 / 0.837 / 0.857 |
| Scratch CNN | Accept everything | 1.000 ×3 | 0.848 / 0.846 / 0.857 |
| ResNet-18 | R-cov (90% coverage on val_cal) | 0.083 / 0.079 / 0.106 | 0.669 / 0.633 / 0.673 |
| ResNet-18 | R-risk (≤1% risk on val_cal) | 0.909 / 0.914 / 0.958 | 0.821 / 0.830 / 0.850 |
| ResNet-18 | Accept everything | 1.000 ×3 | 0.833 / 0.833 / 0.851 |

**Can confidence separate right from wrong answers? (seeds 1/2/3)**

| Model | AUROC error detection, PlantDoc | AUROC, PlantVillage test | AURC PlantDoc | AURC if rejection were random |
|---|---|---|---|---|
| Scratch CNN | 0.593 / 0.614 / 0.605 | 0.983 / 0.988 / 0.985 | 0.794 / 0.773 / 0.795 | 0.848 / 0.846 / 0.857 |
| ResNet-18 | 0.626 / 0.611 / 0.619 | 0.965 / 0.970 / 0.965 | 0.759 / 0.761 / 0.779 | 0.833 / 0.833 / 0.851 |

**ResNet-18 minus CNN on PlantDoc, paired bootstrap, per seed (positive = ResNet-18 higher)**

| Metric | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|
| accuracy | +0.015 [-0.002, +0.031] | +0.012 [-0.004, +0.028] | +0.005 [-0.011, +0.021] |
| macro_f1 | +0.010 [-0.004, +0.024] | +0.019 [+0.004, +0.033] | +0.004 [-0.010, +0.018] |
| ece15_width | +0.007 [-0.012, +0.026] | +0.095 [+0.076, +0.114] | +0.038 [+0.019, +0.056] |
| brier_sum | -0.011 [-0.038, +0.017] | +0.117 [+0.087, +0.147] | +0.047 [+0.018, +0.076] |
| aurc | -0.035 [-0.059, -0.011] | -0.012 [-0.038, +0.014] | -0.016 [-0.039, +0.008] |

**Accuracy gap, PlantVillage test minus PlantDoc, percentage points**

| Model | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|
| Scratch CNN | 83.5 pp [81.8, 85.1] | 83.3 pp [81.7, 85.0] | 84.3 pp [82.7, 85.9] |
| ResNet-18 | 82.7 pp [81.0, 84.4] | 82.9 pp [81.3, 84.6] | 84.6 pp [83.0, 86.2] |

Parameters: CNN 3,395,447; ResNet-18 11,188,311. Learning rates chosen on validation loss only: CNN 0.0005, ResNet-18 0.001.
