#!/usr/bin/env python3
"""Phase 3 analysis of the inventory manifests: duplicates, groups, cross-dataset overlap, counts.

Reads the CSVs written by phase3_audit.py and writes tables and a JSON summary. No model is involved.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plantrel import audit  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit-dir", required=True)
    p.add_argument("--det-labels-dir", required=True, help="dir with train_labels.csv and test_labels.csv")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ad, out = Path(a.audit_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pv = pd.read_csv(ad / "pv_manifest.csv")
    pc = pd.read_csv(ad / "pdcrop_manifest.csv")
    S = {}

    # ---------------- PlantVillage
    S["pv_n"] = len(pv)
    S["pv_classes"] = int(pv["class"].nunique())
    S["pv_unreadable"] = int((pv.error.fillna("") != "").sum())
    S["pv_size_counts"] = {f"{w}x{h}": int(n) for (w, h), n in pv.groupby(["w", "h"]).size().items()}
    S["pv_modes"] = pv["mode"].value_counts().to_dict()
    pv["fallback"] = pv.group.astype(str).str.startswith("fallback_")
    S["pv_fallback_images"] = int(pv.fallback.sum())
    S["pv_groups"] = int(pv.group.nunique())
    S["pv_groups_nonfallback"] = int(pv.loc[~pv.fallback, "group"].nunique())
    gs = pv.groupby("group").size()
    S["pv_group_size"] = {"min": int(gs.min()), "median": float(gs.median()), "max": int(gs.max()), "singletons": int((gs == 1).sum())}
    cls_tab = pv.groupby("class").agg(images=("file", "size"), groups=("group", "nunique"), fallback=("fallback", "sum")).reset_index()
    cls_tab.to_csv(out / "pv_class_table.csv", index=False)
    # groups spanning more than one class (suspicious)
    S["pv_groups_spanning_classes"] = int((pv.groupby("group")["class"].nunique() > 1).sum())
    # exact duplicates
    dup = pv[pv.duplicated("sha256", keep=False)]
    S["pv_exact_dup_files"] = int(len(dup))
    S["pv_exact_dup_groups"] = int(dup.sha256.nunique())
    S["pv_exact_dup_groups_conflicting_class"] = int((dup.groupby("sha256")["class"].nunique() > 1).sum())
    S["pv_exact_dup_groups_spanning_leaf_ids"] = int((dup.groupby("sha256")["group"].nunique() > 1).sum())
    # near duplicates (perceptual hash)
    ph = pv.phash.astype("uint64").to_numpy()
    near = {}
    pairs = audit.near_pairs(ph, 6)
    pdist = np.array([d for _, _, d in pairs]) if pairs else np.array([])
    for t in (0, 2, 4, 6):
        near[str(t)] = int((pdist <= t).sum())
    S["pv_near_pairs_by_threshold"] = near
    gi = pv.group.to_numpy()
    cl = pv["class"].to_numpy()
    sel = [(i, j, d) for i, j, d in pairs if d <= 4]
    S["pv_near_pairs_le4_crossing_leaf_id"] = int(sum(1 for i, j, _ in sel if gi[i] != gi[j]))
    S["pv_near_pairs_le4_crossing_class"] = int(sum(1 for i, j, _ in sel if cl[i] != cl[j]))
    S["pv_near_pairs_le6_crossing_leaf_id"] = int(sum(1 for i, j, _ in pairs if gi[i] != gi[j]))
    S["pv_near_pairs_le6_crossing_class"] = int(sum(1 for i, j, _ in pairs if cl[i] != cl[j]))

    # ---------------- PlantDoc cropped release (whole images with one label)
    S["pc_n"] = len(pc)
    S["pc_classes"] = int(pc["class"].nunique())
    S["pc_by_split"] = pc.groupby("split").size().to_dict()
    S["pc_unreadable"] = int((pc.error.fillna("") != "").sum())
    ar = (pc.w / pc.h)
    S["pc_aspect"] = {"min": float(ar.min()), "p10": float(ar.quantile(.1)), "median": float(ar.median()), "p90": float(ar.quantile(.9)), "max": float(ar.max()),
                      "within_0.8_1.25": float(((ar >= .8) & (ar <= 1.25)).mean())}
    S["pc_size"] = {"min_side_min": int(pc[["w", "h"]].min(axis=1).min()), "min_side_p5": float(pc[["w", "h"]].min(axis=1).quantile(.05)),
                    "min_side_median": float(pc[["w", "h"]].min(axis=1).median())}
    S["pc_images_min_side_lt_64"] = int((pc[["w", "h"]].min(axis=1) < 64).sum())
    S["pc_images_min_side_lt_128"] = int((pc[["w", "h"]].min(axis=1) < 128).sum())
    # exact duplicate groups inside PlantDoc crop release
    g = pc.groupby("sha256").agg(n=("file", "size"), classes=("class", "nunique"), splits=("split", "nunique"))
    S["pc_exact_dup_groups"] = int((g.n > 1).sum())
    S["pc_exact_dup_groups_conflicting_label"] = int((g.classes > 1).sum())
    S["pc_exact_dup_groups_across_train_test"] = int((g.splits > 1).sum())
    # near duplicates inside PlantDoc
    pph = pc.phash.astype("uint64").to_numpy()
    ppairs = audit.near_pairs(pph, 6)
    pd_dist = np.array([d for _, _, d in ppairs]) if ppairs else np.array([])
    S["pc_near_pairs_by_threshold"] = {str(t): int((pd_dist <= t).sum()) for t in (0, 2, 4, 6)}

    # detection labels: images with multiple box classes
    lab = pd.concat([pd.read_csv(Path(a.det_labels_dir) / "train_labels.csv"), pd.read_csv(Path(a.det_labels_dir) / "test_labels.csv")])
    multi = lab.groupby("filename")["class"].nunique()
    multi_files = set(multi[multi > 1].index)
    pc["multi_class_image"] = pc.file.isin(multi_files)
    S["pc_images_with_multiple_box_classes"] = int(pc.multi_class_image.sum())

    # ---------------- cross-dataset overlap PlantVillage vs PlantDoc
    shared = set(pv.sha256) & set(pc.sha256)
    S["cross_exact_shared_sha256"] = len(shared)
    best, arg = audit.cross_nearest(pph, ph)
    pc["pv_nearest_dist"] = best
    pc["pv_nearest_file"] = pv.file.to_numpy()[arg]
    pc["pv_nearest_class"] = pv["class"].to_numpy()[arg]
    S["cross_nearest_dist_hist"] = {str(k): int((best == k).sum()) for k in range(0, 13)}
    S["cross_nearest_le6"] = int((best <= 6).sum())
    S["cross_nearest_le10"] = int((best <= 10).sum())
    pc.sort_values("pv_nearest_dist")[["split", "class", "file", "pv_nearest_dist", "pv_nearest_class", "pv_nearest_file"]].head(40).to_csv(out / "cross_nearest_top40.csv", index=False)
    pc.to_csv(out / "pdcrop_annotated.csv", index=False)
    json.dump(S, open(out / "summary.json", "w"), indent=2, default=str)
    print(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    main()
