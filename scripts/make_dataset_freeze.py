"""Create configs/frozen/dataset_decisions_v1.json (+ .sha256) from the Phase 3/4 outputs.

Source/external data decisions only; no model output is used. Marked provisional where the
user-side items (PlantVillage licence, pathology sign-off) remain open.
"""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plantrel.freeze import write_freeze_file
from plantrel.hashing import sha256_file

R = Path(__file__).resolve().parents[1]
dec = json.load(open(R / "configs/proposed/dataset_decisions_DRAFT.json"))
rep = json.load(open(R / "docs/audit/phase4_report.json"))
classes = rep["classes"]
src, ext = R / "data_manifests/source_manifest.csv", R / "data_manifests/external_manifest.csv"
assert sha256_file(src) == rep["sha256"]["source_manifest.csv"] and sha256_file(ext) == rep["sha256"]["external_manifest.csv"]
e = pd.read_csv(ext)
obj = {
    "schema_version": 1, "protocol_version": "1.0",
    "status": "PROVISIONAL: PlantVillage licence unresolved; class-alignment pathology evidence not source-verified by the user.",
    "class_list": classes,
    "eval_class_lists": {"source_test": classes, "external": classes},
    "class_alignment": dec["primary_class_set"],
    "crop_policy": {"unit": dec["external_unit"], "transform": dec["evaluation_transform"], "stored_size": 224},
    "min_counts": dec["min_counts"],
    "dedup_settings": {"external_cluster_rule": dec["external_cluster_rule"], "plantvillage_group_merge": dec["plantvillage_group_merge"],
                       "cross_dataset_rule": dec["cross_dataset_rule"], "split_seed": rep["seed"]},
    "distributions": {"source_counts": rep["counts"], "source_groups": rep["groups"], "external_n": rep["external_n"],
                      "external_per_class": {c: int((e["canonical"] == c).sum()) for c in classes}},
    "evaluation_transform": dec["evaluation_transform"],
    "licence_findings": {"plantvillage": dec["datasets"]["plantvillage"], "plantdoc": dec["datasets"]["plantdoc"]},
    "leakage_checks": {k: rep[k] for k in ("groups_in_multiple_splits", "exact_dup_hash_across_splits", "pv_sha_in_external", "ext_clusters_dup")},
    "split_manifest_sha256": rep["sha256"]["source_manifest.csv"],
    "external_manifest_sha256": rep["sha256"]["external_manifest.csv"],
}
print(write_freeze_file(R / "configs/frozen/dataset_decisions_v1.json", obj))
