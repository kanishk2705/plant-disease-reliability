"""Fetch the two datasets at the pinned revisions, verify every file against the manifest SHA-256,
and build the uint8 caches + label/split arrays. Run once per fresh runtime.

usage: python scripts/colab_prepare.py --work /content/work
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from plantrel.data import build_cache
from plantrel.hashing import sha256_file


def sh(*a, cwd=None):
    subprocess.run(a, cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", default="/content/work"); a = ap.parse_args()
    W = Path(a.work); W.mkdir(parents=True, exist_ok=True)
    dec = json.load(open(R / "configs/frozen/dataset_decisions_v1.json"))
    pv_rev = dec["licence_findings"]["plantvillage"]["revision"]; pd_rev = dec["licence_findings"]["plantdoc"]["revision"]
    classes = dec["class_list"]
    pv_dir, pd_dir = W / "pv", W / "pd"
    if not pv_dir.exists():
        sh("git", "clone", "-q", "--filter=blob:none", "--no-checkout", "https://github.com/spMohanty/PlantVillage-Dataset", str(pv_dir))
        sh("git", "sparse-checkout", "init", "--no-cone", cwd=pv_dir)
        sh("git", "sparse-checkout", "set", *[f"/raw/color/{c}/" for c in classes], cwd=pv_dir)
        sh("git", "checkout", "-q", pv_rev, cwd=pv_dir)
    if not pd_dir.exists():
        sh("git", "clone", "-q", "https://github.com/pratikkayal/PlantDoc-Dataset", str(pd_dir))
        sh("git", "checkout", "-q", pd_rev, cwd=pd_dir)
    for d, rev in ((pv_dir, pv_rev), (pd_dir, pd_rev)):
        got = subprocess.run(["git", "rev-parse", "HEAD"], cwd=d, capture_output=True, text=True).stdout.strip()
        assert got == rev, f"{d}: revision {got} != pinned {rev}"
    src = pd.read_csv(R / "data_manifests/source_manifest.csv"); ext = pd.read_csv(R / "data_manifests/external_manifest.csv")
    sp = [str(pv_dir / "raw/color" / c / f) for c, f in zip(src["class"], src["file"])]
    ep = [str(pd_dir / s / c / f) for s, c, f in zip(ext["split"], ext["class"], ext["file"])]
    bad = [p for p, h in zip(sp + ep, list(src["sha256"]) + list(ext["sha256"])) if sha256_file(p) != h]
    assert not bad, f"{len(bad)} files differ from manifest SHA-256, e.g. {bad[:3]}"
    print("all", len(sp) + len(ep), "files match the manifests", flush=True)
    C = W / "cache"; C.mkdir(exist_ok=True)
    build_cache(sp, C / "source.npy", workers=2); build_cache(ep, C / "external.npy", workers=2)
    np.save(C / "source_labels.npy", src["label"].to_numpy(np.int64)); np.save(C / "source_splits.npy", src["split"].to_numpy(str))
    np.save(C / "external_labels.npy", ext["label"].to_numpy(np.int64))
    print("caches built:", {p.name: p.stat().st_size // 2**20 for p in C.iterdir()})


if __name__ == "__main__":
    main()
