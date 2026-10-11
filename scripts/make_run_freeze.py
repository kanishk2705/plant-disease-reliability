"""Write configs/frozen/<run_id>.json (+ .sha256) from a calibration summary line printed by calibrate_runs.py.

usage: python scripts/make_run_freeze.py CALIBRATION_JSON_LINES_FILE --code-commit <40-hex>
Run AFTER all code is committed; the freeze records the code commit it is bound to.
"""
import argparse, json, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from plantrel.freeze import write_freeze_file
from plantrel.hashing import sha256_file


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("lines"); ap.add_argument("--code-commit", required=True)
    ap.add_argument("--environment", required=True, help="JSON file describing the Colab environment")
    a = ap.parse_args()
    ds = json.load(open(R / "configs/frozen/dataset_decisions_v1.json"))
    env = json.load(open(a.environment))
    for line in Path(a.lines).read_text().splitlines():
        if not line.strip().startswith("{"):
            continue
        r = json.loads(line)
        seed = int(r["run"].rsplit("_s", 1)[1])
        obj = {"schema_version": 1, "run_id": r["run"], "seed": seed, "code_commit": a.code_commit, "T": r["T"],
               "tau": {"r_cov": r["tau"]["r_cov"], "r_risk": r["tau"]["r_risk"]}, "checkpoint_sha256": r["ckpt"],
               "class_list": ds["class_list"], "eval_class_lists": ds["eval_class_lists"], "config_hash": r["cfg"],
               "manifest_hashes": {"source": ds["split_manifest_sha256"], "external": ds["external_manifest_sha256"]},
               "dataset_freeze_sha256": sha256_file(R / "configs/frozen/dataset_decisions_v1.json"), "environment": env}
        print(r["run"], write_freeze_file(R / "configs/frozen" / f"{r['run']}.json", obj))


if __name__ == "__main__":
    main()
