"""Source-only calibration for finished runs (reads val_cal, val_sel, test only; never external data).

Writes <run_dir>/calibration.json with T, thresholds (R-cov 90%, R-risk 1%), checkpoint hash,
config hash and the SOURCE-TEST metric bundle. val_cal numbers are development diagnostics.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from plantrel.evaluation import calibrate, bundle
from plantrel.hashing import sha256_file


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", default="/content/work"); ap.add_argument("--pattern", default="final_*")
    a = ap.parse_args()
    classes = json.load(open(R / "configs/frozen/dataset_decisions_v1.json"))["class_list"]
    for rd in sorted((Path(a.work) / "runs").glob(a.pattern)):
        if not (rd / "summary.json").exists():
            print("incomplete:", rd.name); continue
        z = np.load(rd / "source_logits.npz")
        cal = calibrate(z["val_cal_logits"], z["val_cal_labels"])
        taus = {"r_cov": cal["tau_r_cov"], "r_risk": cal["tau_r_risk"]}
        out = {"run_id": rd.name, "calibration": cal, "checkpoint_sha256": sha256_file(rd / "best.pt"),
               "config_hash": json.load(open(rd / "config.json"))["config_hash"],
               "val_cal_diagnostic": bundle(z["val_cal_logits"], z["val_cal_labels"], cal["T"], classes, taus),
               "val_sel": bundle(z["val_sel_logits"], z["val_sel_labels"], cal["T"], classes, taus),
               "source_test": bundle(z["test_logits"], z["test_labels"], cal["T"], classes, taus)}
        (rd / "calibration.json").write_text(json.dumps(out, indent=1))
        st = out["source_test"]
        print(json.dumps({"run": rd.name, "T": cal["T"], "tau": taus, "ckpt": out["checkpoint_sha256"],
                          "cfg": out["config_hash"], "test_acc": round(st["cal"]["accuracy"], 4),
                          "test_f1": round(st["cal"]["macro_f1"], 4), "test_ece": round(st["cal"]["ece15_width"], 4)}))


if __name__ == "__main__":
    main()
