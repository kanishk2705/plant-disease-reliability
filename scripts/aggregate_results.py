"""Collect final_report.json files into one compact results JSON (per-seed values and ranges, no ensembling).

  python scripts/aggregate_results.py --work /content/work --repo-root /content/repo --out /content/work/results.json
Also computes the paired ResNet-18 minus CNN comparison per seed on the same examples.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from plantrel import report

KEYS = ["accuracy", "macro_f1", "ece15_width", "ece15_mass", "brier_sum", "nll", "overconfidence", "aurc", "aurc_random", "aurc_oracle", "auroc_error"]


def r4(o):
    if isinstance(o, float): return round(o, 4) if np.isfinite(o) else None
    if isinstance(o, dict): return {k: r4(v) for k, v in o.items()}
    if isinstance(o, list): return [r4(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", default="/content/work"); ap.add_argument("--repo-root", default=str(R))
    ap.add_argument("--out", required=True); ap.add_argument("--n-boot", type=int, default=10000); a = ap.parse_args()
    runs = Path(a.work) / "runs"; repo = Path(a.repo_root)
    ds = json.load(open(repo / "configs/frozen/dataset_decisions_v1.json")); K = len(ds["class_list"])
    src = pd.read_csv(repo / "data_manifests/source_manifest.csv"); ext = pd.read_csv(repo / "data_manifests/external_manifest.csv")
    ecl = pd.factorize(ext["cluster"].to_numpy())[0]
    res = {"classes": ds["class_list"], "models": {}, "paired_resnet18_minus_cnn": {}}
    for m in ("cnn", "resnet18"):
        d = {"params": None, "seeds": {}}
        for s in (1, 2, 3):
            rd = runs / f"final_{m}_s{s}"; f = rd / "final_report.json"
            if not f.exists(): continue
            fr = json.load(open(f)); cfg = json.load(open(rd / "config.json")); d["params"] = cfg["params"]
            cal = json.load(open(rd / "calibration.json"))
            ent = {"T": fr["T"], "taus": fr["taus"], "lr": cfg["lr"], "epochs_run": sum(1 for _ in open(rd / "log.jsonl")),
                   "val_cal_diagnostic": {k: cal["val_cal_diagnostic"]["cal"][k] for k in ("nll", "ece15_width", "accuracy")},
                   "val_cal_nll_uncal": cal["val_cal_diagnostic"]["uncal"]["nll"]}
            for part in ("source_test", "external"):
                p = fr[part]["point"]
                ent[part] = {"n": p["n"], "uncal": {k: p["uncal"][k] for k in KEYS}, "cal": {k: p["cal"][k] for k in KEYS},
                             "selective": p["cal"]["selective"], "ci": fr[part]["ci"],
                             "high_conf_errors_ge_0.9": fr[part]["plots"]["high_conf_errors_ge_0.9"]}
            ent["gaps_source_minus_external"] = fr["gaps_source_minus_external"]
            if s == 1:
                ent["plots"] = {p: {k: fr[p]["plots"][k] for k in ("reliability_bins", "rc_curve", "per_class", "confusion", "conf_hist_correct", "conf_hist_wrong")}
                                for p in ("source_test", "external")}
            d["seeds"][str(s)] = ent
        res["models"][m] = d
    for s in (1, 2, 3):
        pa, pb = runs / f"final_resnet18_s{s}", runs / f"final_cnn_s{s}"
        if not ((pa / "external_logits.npz").exists() and (pb / "external_logits.npz").exists()): continue
        fa, fb = json.load(open(pa / "final_report.json")), json.load(open(pb / "final_report.json"))
        ea, eb = np.load(pa / "external_logits.npz"), np.load(pb / "external_logits.npz")
        za, zb = np.load(pa / "source_logits.npz"), np.load(pb / "source_logits.npz")
        tcl = pd.factorize(src["gid"].to_numpy()[za["test_index"]])[0]
        res["paired_resnet18_minus_cnn"][str(s)] = {
            "external": report.paired_models(ea["logits"], fa["T"], eb["logits"], fb["T"], ea["labels"], ecl, K, a.n_boot, 10 + s),
            "source_test": report.paired_models(za["test_logits"], fa["T"], zb["test_logits"], fb["T"], za["test_labels"], tcl, K, a.n_boot, 20 + s)}
    Path(a.out).write_text(json.dumps(r4(res), separators=(",", ":")))
    print("wrote", a.out, Path(a.out).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
