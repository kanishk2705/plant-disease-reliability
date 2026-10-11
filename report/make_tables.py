"""Markdown tables from results/results.json: every seed shown, plus the range. No ensembling."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1]
res = json.load(open(R / "results/results.json")); M = res["models"]
N = {"cnn": "Scratch CNN", "resnet18": "ResNet-18"}
f = lambda x, d=3: "n/a" if x is None else f"{x:.{d}f}"
def rng(v, d=3): return f"{min(v):.{d}f}–{max(v):.{d}f}"
out = []
def table(title, rows, cols):
    out.append(f"\n**{title}**\n"); out.append("| " + " | ".join(cols) + " |"); out.append("|" + "---|" * len(cols))
    for r in rows: out.append("| " + " | ".join(r) + " |")
def vals(m, part, key, kind="cal"): return [M[m]["seeds"][s][part][kind][key] for s in ("1", "2", "3")]
rows = []
for part, lab in (("source_test", "PlantVillage test"), ("external", "PlantDoc")):
    for m in ("cnn", "resnet18"):
        rows.append([lab, N[m]] + [f(vals(m, part, k)[0]) + " / " + f(vals(m, part, k)[1]) + " / " + f(vals(m, part, k)[2]) for k in ("accuracy", "macro_f1", "ece15_width", "nll")])
table("Headline metrics (seed 1 / seed 2 / seed 3; temperature-scaled with the frozen source-validation T)", rows, ["Set", "Model", "Accuracy", "Macro-F1", "ECE (15 bins)", "NLL"])
rows = []
for m in ("cnn", "resnet18"):
    ci = [M[m]["seeds"][s]["external"]["ci"] for s in ("1", "2", "3")]
    rows.append([N[m]] + [", ".join(f"{c[k]['point']:.3f} [{c[k]['lo']:.3f}, {c[k]['hi']:.3f}]" for c in ci) for k in ("accuracy", "macro_f1")])
table("PlantDoc 95% cluster-bootstrap intervals, per seed (10,000 resamples; clusters = near-duplicate groups)", rows, ["Model", "Accuracy [95% CI] seeds 1/2/3", "Macro-F1 [95% CI] seeds 1/2/3"])
rows = []
for m in ("cnn", "resnet18"):
    rows.append([N[m], " / ".join(f(v) for v in vals(m, "external", "ece15_width", "uncal")), " / ".join(f(v) for v in vals(m, "external", "ece15_width")),
                 " / ".join(f(v) for v in vals(m, "external", "overconfidence")), " / ".join(f(v) for v in vals(m, "external", "brier_sum"))])
table("Calibration on PlantDoc (seeds 1/2/3). Brier = sum over classes, range 0–2", rows, ["Model", "ECE before scaling", "ECE after scaling", "Mean confidence − accuracy", "Brier"])
rows = []
for m in ("cnn", "resnet18"):
    for rule, lab in (("r_cov", "R-cov (90% coverage on val_cal)"), ("r_risk", "R-risk (≤1% risk on val_cal)")):
        s = [M[m]["seeds"][k]["external"]["selective"][rule] for k in ("1", "2", "3")]
        rows.append([N[m], lab, " / ".join(f(x["coverage"]) for x in s), " / ".join(f(x["risk"]) for x in s)])
    rows.append([N[m], "Accept everything", "1.000 ×3", " / ".join(f(1 - a) for a in vals(m, "external", "accuracy"))])
table("Abstention on PlantDoc at the frozen thresholds (seeds 1/2/3)", rows, ["Model", "Rule", "Coverage", "Selective risk (error rate among answered)"])
rows = []
for m in ("cnn", "resnet18"):
    rows.append([N[m], " / ".join(f(v) for v in vals(m, "external", "auroc_error")), " / ".join(f(v) for v in vals(m, "source_test", "auroc_error")),
                 " / ".join(f(v) for v in vals(m, "external", "aurc")), " / ".join(f(v) for v in vals(m, "external", "aurc_random"))])
table("Can confidence separate right from wrong answers? (seeds 1/2/3)", rows, ["Model", "AUROC error detection, PlantDoc", "AUROC, PlantVillage test", "AURC PlantDoc", "AURC if rejection were random"])
rows = []
for k in ("accuracy", "macro_f1", "ece15_width", "brier_sum", "aurc"):
    cells = []
    for s in ("1", "2", "3"):
        p = res["paired_resnet18_minus_cnn"][s]["external"][k]; cells.append(f"{p['point']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]")
    rows.append([k] + cells)
table("ResNet-18 minus CNN on PlantDoc, paired bootstrap, per seed (positive = ResNet-18 higher)", rows, ["Metric", "Seed 1", "Seed 2", "Seed 3"])
rows = []
for m in ("cnn", "resnet18"):
    g = [M[m]["seeds"][s]["gaps_source_minus_external"]["accuracy"] for s in ("1", "2", "3")]
    rows.append([N[m]] + [f"{x['point']*100:.1f} pp [{x['lo']*100:.1f}, {x['hi']*100:.1f}]" for x in g])
table("Accuracy gap, PlantVillage test minus PlantDoc, percentage points", rows, ["Model", "Seed 1", "Seed 2", "Seed 3"])
out.append(f"\nParameters: CNN {M['cnn']['params']:,}; ResNet-18 {M['resnet18']['params']:,}. Learning rates chosen on validation loss only: CNN {M['cnn']['seeds']['1']['lr']}, ResNet-18 {M['resnet18']['seeds']['1']['lr']}.")
(R / "docs/RESULTS_TABLES.md").write_text("\n".join(out) + "\n"); print("\n".join(out))
