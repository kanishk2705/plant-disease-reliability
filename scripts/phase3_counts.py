#!/usr/bin/env python3
"""Apply the proposed alignment and the protocol 4.4 minimum-count rule to the audit manifests.

Uses only labels, identifiers, hashes and counts. No model output is involved.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plantrel import audit  # noqa: E402

MIN_EXT, MIN_PV_IMAGES, MIN_PV_GROUPS = 15, 200, 20
EXT_NEAR, PV_NEAR = 6, 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit-dir", required=True)
    p.add_argument("--analysis-dir", required=True)
    p.add_argument("--alignment", required=True)
    p.add_argument("--det-labels-dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    al = json.load(open(a.alignment))["rows"]
    pv = pd.read_csv(Path(a.audit_dir) / "pv_manifest.csv")
    pc = pd.read_csv(Path(a.analysis_dir) / "pdcrop_annotated.csv")
    R = {}

    # ---- external units: whole images of the cropped release
    n0 = len(pc)
    ph = pc.phash.astype("uint64").to_numpy()
    edges = [(i, j) for i, j, _ in audit.near_pairs(ph, EXT_NEAR)]
    sha_first = {}
    for i, s in enumerate(pc.sha256):
        if s in sha_first:
            edges.append((sha_first[s], i))
        else:
            sha_first[s] = i
    pc["cluster"] = audit.connected_components(len(pc), edges)
    pc["cluster_n"] = pc.groupby("cluster")["file"].transform("size")
    pc["cluster_labels"] = pc.groupby("cluster")["class"].transform("nunique")
    R["ext_start_images"] = n0
    R["ext_clusters"] = int(pc.cluster.nunique())
    R["ext_clusters_with_more_than_one_image"] = int((pc.groupby("cluster").size() > 1).sum())
    R["ext_clusters_conflicting_label"] = int((pc.groupby("cluster")["class"].nunique() > 1).sum())
    pc["excl_conflict"] = pc.cluster_labels > 1
    pc["excl_multiclass_image"] = pc.multi_class_image & ~pc.excl_conflict
    keep = pc[~pc.excl_conflict & ~pc.excl_multiclass_image].copy()
    R["ext_excluded_conflicting_label_cluster_images"] = int(pc.excl_conflict.sum())
    R["ext_excluded_multiclass_images_additional"] = int(pc.excl_multiclass_image.sum())
    keep = keep.sort_values(["cluster", "split", "file"]).drop_duplicates("cluster", keep="first")
    R["ext_after_cleaning_one_per_cluster"] = len(keep)
    R["ext_removed_as_redundant_duplicates"] = int(len(pc) - pc.excl_conflict.sum() - pc.excl_multiclass_image.sum() - len(keep))

    # ---- PlantVillage class counts and groups, with PV-side group merge proposal (same class, pHash <= 2)
    pv["fallback"] = pv.group.astype(str).str.startswith("fallback_")
    pph = pv.phash.astype("uint64").to_numpy()
    pe = [(i, j) for i, j, d in audit.near_pairs(pph, PV_NEAR) if pv["class"].iat[i] == pv["class"].iat[j]]
    # link by existing group id as well
    gid = {g: k for k, g in enumerate(pv.group.unique())}
    base = pv.group.map(gid).to_numpy()
    first = {}
    for i, g in enumerate(base):
        if g in first:
            pe.append((first[g], i))
        else:
            first[g] = i
    pv["merged_group"] = audit.connected_components(len(pv), pe)
    R["pv_groups_before_merge"] = int(pv.group.nunique())
    R["pv_groups_after_phash_le2_same_class_merge"] = int(pv.merged_group.nunique())
    R["pv_same_class_phash_le2_pairs"] = int(len([1 for i, j, d in audit.near_pairs(pph, PV_NEAR) if pv["class"].iat[i] == pv["class"].iat[j]]))
    pvt = pv.groupby("class").agg(pv_images=("file", "size"), pv_groups=("group", "nunique"), pv_groups_merged=("merged_group", "nunique"),
                                  pv_fallback_images=("fallback", "sum")).reset_index()

    # ---- apply alignment
    rows = []
    ext_counts = keep.groupby("class").size()
    ext_raw = pc.groupby("class").size()
    ext_clusters_raw = pc.groupby("class").cluster.nunique()
    for r in al:
        pvrow = pvt[pvt["class"] == r["plantvillage"]].iloc[0]
        e = int(ext_counts.get(r["plantdoc"], 0))
        ok_ext, ok_pv = e >= MIN_EXT, (pvrow.pv_images >= MIN_PV_IMAGES and pvrow.pv_groups_merged >= MIN_PV_GROUPS)
        rows.append({**r, "ext_raw_images": int(ext_raw.get(r["plantdoc"], 0)), "ext_after_cleaning": e,
                     "pv_images": int(pvrow.pv_images), "pv_groups": int(pvrow.pv_groups), "pv_groups_merged": int(pvrow.pv_groups_merged),
                     "pass_ext_min15": bool(ok_ext), "pass_pv_min200_20": bool(ok_pv),
                     "primary_set": bool(r["tier"] == "A" and ok_ext and ok_pv),
                     "sensitivity_set": bool(r["tier"] in ("A", "B") and ok_ext and ok_pv)})
    T = pd.DataFrame(rows)
    T.to_csv(out / "alignment_counts.csv", index=False)
    prim, sens = T[T.primary_set], T[T.sensitivity_set]
    R["primary_K"] = int(len(prim))
    R["sensitivity_K"] = int(len(sens))
    R["primary_external_images"] = int(prim.ext_after_cleaning.sum())
    R["primary_pv_images"] = int(prim.pv_images.sum())
    R["sensitivity_external_images"] = int(sens.ext_after_cleaning.sum())
    R["primary_classes"] = prim.plantvillage.tolist()
    R["tier_A_failing_min_counts"] = T[(T.tier == "A") & ~T.sensitivity_set].plantvillage.tolist()
    R["tier_B"] = T[T.tier == "B"].plantvillage.tolist()
    R["pv_classes_without_plantdoc_counterpart"] = sorted(set(pv["class"]) - set(T.plantvillage))
    R["min_ext_in_primary"] = int(prim.ext_after_cleaning.min())
    R["min_pv_in_primary"] = int(prim.pv_images.min())
    # leakage-relevant: how many primary-class PV images lack a published leaf id
    pp = pv[pv["class"].isin(prim.plantvillage)]
    R["primary_pv_images_fallback_group"] = int(pp.fallback.sum())
    keep.to_csv(out / "external_clean_manifest.csv", index=False)
    pv[["class", "file", "group", "merged_group", "sha256"]].to_csv(out / "pv_groups_manifest.csv", index=False)
    pvt.to_csv(out / "pv_class_table.csv", index=False)
    json.dump(R, open(out / "counts_summary.json", "w"), indent=2)
    print(json.dumps(R, indent=1))
    print(T[["plantdoc", "tier", "ext_raw_images", "ext_after_cleaning", "pv_images", "pv_groups", "pv_groups_merged", "primary_set"]].to_string())


if __name__ == "__main__":
    main()
