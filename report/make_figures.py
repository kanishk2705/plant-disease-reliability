"""Figures from results/results.json (kept outside the frozen code paths on purpose)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
res = json.load(open(R / "results/results.json"))
OUT = R / "docs/figures"; OUT.mkdir(parents=True, exist_ok=True)
COL = {"cnn": "#c8553d", "resnet18": "#2a6f97"}
NAME = {"cnn": "Scratch CNN (3.4M params)", "resnet18": "ResNet-18 (11.2M, ImageNet init)"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140,
                     "axes.grid": True, "grid.alpha": .25})
M = res["models"]; short = res["classes"]


def seedv(m, part, key, kind="cal"):
    return [M[m]["seeds"][s][part][kind][key] for s in sorted(M[m]["seeds"])]


# 1: in-distribution vs external accuracy / macro-F1 (every seed shown)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, key, title in zip(ax, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1")):
    for i, m in enumerate(("cnn", "resnet18")):
        for part, x in (("source_test", 0), ("external", 1)):
            v = seedv(m, part, key)
            a.scatter([x + (i - .5) * .12] * len(v), v, color=COL[m], s=40, label=NAME[m] if x == 0 else None, zorder=3)
        a.plot([(i - .5) * .12, 1 + (i - .5) * .12], [np.mean(seedv(m, "source_test", key)), np.mean(seedv(m, "external", key))],
               color=COL[m], alpha=.5)
    a.set_xticks([0, 1]); a.set_xticklabels(["PlantVillage test\n(same distribution)", "PlantDoc\n(external)"]); a.set_ylim(0, 1.02)
    a.set_title(title); a.axhline(1 / 23, color="grey", ls=":", lw=1); a.text(1.45, 1 / 23 + .015, "chance (1/23)", ha="right", color="grey", fontsize=9)
ax[0].legend(loc="center left", fontsize=9, frameon=False)
fig.suptitle("The distribution-shift gap (3 seeds each, all points shown)", y=1.02); fig.tight_layout(); fig.savefig(OUT / "fig1_gap.png", bbox_inches="tight"); plt.close(fig)

# 2: reliability diagrams (seed 1)
fig, ax = plt.subplots(2, 2, figsize=(9, 8.2), sharex=True, sharey=True)
for r_, part in enumerate(("source_test", "external")):
    for c_, m in enumerate(("cnn", "resnet18")):
        a = ax[r_][c_]; e = M[m]["seeds"]["1"]
        bins = e["plots"][part]["reliability_bins"]; k = len(bins)
        xs = (np.arange(k) + .5) / k
        acc = [b["accuracy"] if b["count"] else np.nan for b in bins]
        a.bar(xs, acc, width=1 / k, color=COL[m], alpha=.75, edgecolor="white")
        a.plot([0, 1], [0, 1], "k--", lw=1)
        a.set_title(f'{NAME[m].split(" (")[0]} · {"PlantVillage test" if part=="source_test" else "PlantDoc"}\nECE {e[part]["cal"]["ece15_width"]:.3f} · acc {e[part]["cal"]["accuracy"]:.3f}', fontsize=10)
        if r_ == 1: a.set_xlabel("confidence")
        if c_ == 0: a.set_ylabel("accuracy")
fig.suptitle("Reliability diagrams after source-only temperature scaling (seed 1, 15 equal-width bins)", y=1.0); fig.tight_layout()
fig.savefig(OUT / "fig2_reliability.png", bbox_inches="tight"); plt.close(fig)

# 3: risk-coverage on external (seed 1) with frozen thresholds
fig, ax = plt.subplots(figsize=(6.6, 4.6))
for m in ("cnn", "resnet18"):
    e = M[m]["seeds"]["1"]; rc = e["plots"]["external"]["rc_curve"]
    ax.plot(rc["coverage"], rc["risk"], color=COL[m], label=NAME[m])
    for rule, mk in (("r_cov", "o"), ("r_risk", "s")):
        s = e["external"]["selective"][rule]
        if s: ax.scatter([s["coverage"]], [s["risk"]], color=COL[m], marker=mk, s=55, zorder=4, edgecolor="k")
base = 1 - np.mean([e["external"]["cal"]["accuracy"] for e in M["cnn"]["seeds"].values()])
ax.axhline(base, color="grey", ls=":", lw=1); ax.text(.02, base + .012, "accept-all risk (random rejection)", color="grey", fontsize=9)
ax.scatter([], [], marker="o", color="w", edgecolor="k", label="frozen R-cov (90% on val_cal)"); ax.scatter([], [], marker="s", color="w", edgecolor="k", label="frozen R-risk (1% on val_cal)")
ax.set_xlabel("coverage (fraction of PlantDoc images answered)"); ax.set_ylabel("selective risk (error rate among answered)")
ax.set_ylim(0, 1); ax.set_xlim(0, 1.01); ax.set_title("Abstention on PlantDoc (seed 1)"); ax.legend(fontsize=8.5, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig3_risk_coverage.png"); plt.close(fig)

# 4: per-class external F1 (seed 1)
pc = {m: M[m]["seeds"]["1"]["plots"]["external"]["per_class"] for m in M}
order = np.argsort(pc["resnet18"]["f1"])
fig, ax = plt.subplots(figsize=(8.5, 7))
y = np.arange(len(order))
for off, m in ((-.2, "cnn"), (.2, "resnet18")):
    ax.barh(y + off, np.array(pc[m]["f1"])[order], height=.38, color=COL[m], label=NAME[m])
ax.set_yticks(y); ax.set_yticklabels([short[i].replace("___", " – ").replace("_", " ") for i in order], fontsize=8)
ax.set_xlabel("F1 on PlantDoc (seed 1)"); ax.legend(frameon=False, fontsize=9); ax.set_title("Per-class external F1 (exploratory)")
fig.tight_layout(); fig.savefig(OUT / "fig4_per_class_f1.png"); plt.close(fig)

# 5: effect of source-only temperature scaling on external ECE
fig, ax = plt.subplots(figsize=(6, 4))
for i, m in enumerate(("cnn", "resnet18")):
    u = seedv(m, "external", "ece15_width", "uncal"); c = seedv(m, "external", "ece15_width", "cal")
    for a_, b_ in zip(u, c):
        ax.plot([i - .15, i + .15], [a_, b_], color=COL[m], marker="o")
ax.set_xticks([0, 1]); ax.set_xticklabels(["CNN", "ResNet-18"]); ax.set_ylabel("external ECE (lower is better)")
ax.set_title("Temperature scaling fitted on PlantVillage\ndid not fix PlantDoc calibration (left: raw, right: scaled)", fontsize=10); ax.set_ylim(0, .9)
fig.tight_layout(); fig.savefig(OUT / "fig5_temperature.png"); plt.close(fig)
print("figures written")
