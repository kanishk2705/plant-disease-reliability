"""External evaluation of ONE frozen run. Refuses to touch external data unless the run freeze verifies.

  python scripts/final_eval.py --run-id final_cnn_s1 --work /content/work --repo-root /content/repo --n-boot 10000
  python scripts/final_eval.py --run-id final_cnn_s1 --work ... --source-only   (no freeze needed, no external data)

Order enforced here: verify freeze -> load model -> compute external logits -> metrics.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from plantrel import report
from plantrel.evaluation import bundle
from plantrel.freeze import FreezeVerificationError, verify_run_freeze
from plantrel.hashing import sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True); ap.add_argument("--work", default="/content/work")
    ap.add_argument("--repo-root", default=str(R)); ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--source-only", action="store_true")
    a = ap.parse_args()
    work, repo = Path(a.work), Path(a.repo_root)
    rd = work / "runs" / a.run_id
    ds = repo / "configs/frozen/dataset_decisions_v1.json"
    ds_obj = json.load(open(ds)); classes = ds_obj["class_list"]; K = len(classes)
    src = pd.read_csv(repo / "data_manifests/source_manifest.csv"); ext = pd.read_csv(repo / "data_manifests/external_manifest.csv")
    z = np.load(rd / "source_logits.npz"); cal = json.load(open(rd / "calibration.json"))
    T, taus = cal["calibration"]["T"], {"r_cov": cal["calibration"]["tau_r_cov"], "r_risk": cal["calibration"]["tau_r_risk"]}
    t_idx = z["test_index"]; test_cl = pd.factorize(src["gid"].to_numpy()[t_idx])[0]
    out = {"run_id": a.run_id, "T": T, "taus": taus, "code_commit_head": None}
    if not a.source_only:
        fz = repo / "configs/frozen" / f"{a.run_id}.json"
        try:
            verify_run_freeze(fz, expected_sha256=(fz.with_suffix(".sha256").read_text().split()[0]), repo_root=repo,
                              checkpoint_path=rd / "best.pt", source_manifest_path=repo / "data_manifests/source_manifest.csv",
                              external_manifest_path=repo / "data_manifests/external_manifest.csv", dataset_freeze_path=ds)
        except FreezeVerificationError as e:
            print("REFUSED:", e); sys.exit(1)
        frozen = json.load(open(fz)); assert frozen["T"] == T and frozen["tau"]["r_cov"] == taus["r_cov"] and frozen["tau"]["r_risk"] == taus["r_risk"]
        out["code_commit_frozen"] = frozen["code_commit"]; print("run freeze verified", flush=True)
    out["source_test"] = {"point": bundle(z["test_logits"], z["test_labels"], T, classes, taus),
                          "ci": report.bootstrap_set(z["test_logits"], z["test_labels"], test_cl, T, K, taus["r_cov"], a.n_boot, 1),
                          "plots": report.plot_data(z["test_logits"], z["test_labels"], T, K)}
    if not a.source_only:
        import torch
        from plantrel.checkpoint import load_checkpoint
        from plantrel.data import normalise, to_float
        from plantrel.models import build_model
        from plantrel.trainer import predict_logits
        cfg = json.load(open(rd / "config.json")); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg["model"], K, pretrained=False).to(dev)
        load_checkpoint(rd / "best.pt", expected_config_hash=cfg["config_hash"], model=model, restore_rng=False)
        earr = np.load(work / "cache/external.npy", mmap_mode="r"); ey = np.load(work / "cache/external_labels.npy")
        lg = predict_logits(model, earr, ey, np.arange(len(ey)), dev)
        np.savez_compressed(rd / "external_logits.npz", logits=lg, labels=ey)
        ecl = pd.factorize(ext["cluster"].to_numpy())[0]
        out["external"] = {"point": bundle(lg, ey, T, classes, taus), "ci": report.bootstrap_set(lg, ey, ecl, T, K, taus["r_cov"], a.n_boot, 2),
                           "plots": report.plot_data(lg, ey, T, K)}
        out["gaps_source_minus_external"] = report.gaps(z["test_logits"], z["test_labels"], test_cl, lg, ey, ecl, T, K, a.n_boot, 3)
    (rd / ("source_report.json" if a.source_only else "final_report.json")).write_text(json.dumps(out))
    s = out["source_test"]["point"]["cal"]
    print(json.dumps({"run": a.run_id, "source_acc": round(s["accuracy"], 4), "source_f1": round(s["macro_f1"], 4),
                      **({"ext_acc": round(out["external"]["point"]["cal"]["accuracy"], 4),
                          "ext_f1": round(out["external"]["point"]["cal"]["macro_f1"], 4)} if "external" in out else {})}))


if __name__ == "__main__":
    main()
