"""Phase 4: leaf-group-aware source splits, external manifest, leakage checks.

Inputs (outputs/audit/*.csv from Phase 3) and the DRAFT dataset decisions.
Split seed 42. Fractions train .70 / val_sel .075 / val_cal .075 / test .15,
applied per class at the level of merged groups (a group is never split).
Writes outputs/splits/*.csv and docs/audit/phase4_report.json.
"""
import json, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
FR = {"train": .70, "val_sel": .075, "val_cal": .075, "test": .15}
ORDER = list(FR)

def assign_class(groups: pd.Series, rng) -> dict:
    sizes = groups.value_counts()
    ids = sorted(sizes.index, key=str)
    perm = rng.permutation(len(ids))
    ids = [ids[i] for i in perm]
    ids.sort(key=lambda g: -sizes[g])  # stable: big groups first
    n = int(sizes.sum()); target = {s: FR[s] * n for s in ORDER}; got = {s: 0 for s in ORDER}
    out = {}
    for g in ids:
        s = max(ORDER, key=lambda k: (target[k] - got[k]) / target[k])
        out[g] = s; got[s] += sizes[g]
    return out

def main():
    pv = pd.read_csv(ROOT/"outputs/audit/pv_groups_manifest.csv")
    ext = pd.read_csv(ROOT/"outputs/audit/external_clean_manifest.csv")
    dec = json.load(open(ROOT/"configs/proposed/dataset_decisions_DRAFT.json"))
    prim = {c["plantdoc_label"]: c["canonical"] for c in dec["primary_class_set"]}
    classes = sorted(prim.values())
    pvk = pv[pv["class"].isin(classes)].copy()
    pvk["gid"] = pvk["class"] + ":::" + pvk["merged_group"].astype(str)
    rng = np.random.default_rng(SEED)
    split = {}
    for c in classes:
        split.update(assign_class(pvk.loc[pvk["class"] == c, "gid"], rng))
    pvk["split"] = pvk["gid"].map(split)
    pvk["label"] = pvk["class"].map({c: i for i, c in enumerate(classes)})
    ek = ext[ext["class"].isin(prim)].copy()
    ek["canonical"] = ek["class"].map(prim); ek["label"] = ek["canonical"].map({c: i for i, c in enumerate(classes)})
    outd = ROOT/"outputs/splits"; outd.mkdir(parents=True, exist_ok=True)
    pvk[["class","file","gid","sha256","split","label"]].to_csv(outd/"source_manifest.csv", index=False)
    ek[["canonical","class","file","cluster","sha256","label","w","h","phash","pv_nearest_dist"]].to_csv(outd/"external_manifest.csv", index=False)
    rep = {"seed": SEED, "K": len(classes), "classes": classes}
    g2s = pvk.groupby("gid")["split"].nunique(); rep["groups_in_multiple_splits"] = int((g2s > 1).sum())
    h = pvk.groupby("sha256")["split"].nunique(); rep["exact_dup_hash_across_splits"] = int((h > 1).sum())
    rep["pv_sha_in_external"] = int(pvk["sha256"].isin(set(ek["sha256"])).sum())
    rep["ext_clusters_dup"] = int(ek["cluster"].duplicated().sum())
    rep["counts"] = {s: int((pvk.split == s).sum()) for s in ORDER}
    rep["fractions"] = {s: round(rep["counts"][s]/len(pvk), 4) for s in ORDER}
    rep["classes_missing_from_any_split"] = [c for c in classes if pvk[pvk["class"]==c]["split"].nunique() < 4]
    rep["per_class_min_per_split"] = {s: int(pvk[pvk.split==s].groupby("class").size().min()) for s in ORDER}
    rep["groups"] = {s: int(pvk[pvk.split==s]["gid"].nunique()) for s in ORDER}
    rep["external_n"] = len(ek); rep["external_min_per_class"] = int(ek.groupby("label").size().min())
    rep["external_classes_covered"] = int(ek["label"].nunique())
    rep["pv_total"] = len(pvk)
    rep["sha256"] = {f: hashlib.sha256((outd/f).read_bytes()).hexdigest() for f in ("source_manifest.csv","external_manifest.csv")}
    (ROOT/"docs/audit/phase4_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "classes"}, indent=1))
    ok = rep["groups_in_multiple_splits"] == 0 and rep["exact_dup_hash_across_splits"] == 0 \
        and rep["pv_sha_in_external"] == 0 and rep["ext_clusters_dup"] == 0 and not rep["classes_missing_from_any_split"]
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
