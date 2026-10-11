#!/usr/bin/env python3
"""Phase 3 dataset inventory: counts, sizes, hashes and perceptual hashes. No model is involved.

Writes per-image manifests (CSV) to --out (kept out of Git) and prints a JSON summary.
"""
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image  # noqa: E402

from plantrel import audit  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def inspect(path):
    try:
        with Image.open(path) as im:
            im.load()
            return {"w": im.width, "h": im.height, "mode": im.mode, "phash": audit.phash64(im), "error": ""}
    except Exception as exc:  # corrupt files are listed, not hidden
        return {"w": -1, "h": -1, "mode": "", "phash": -1, "error": type(exc).__name__}


def work(item):
    path, meta = item
    r = inspect(path)
    r["sha256"] = audit.sha256_of(path)
    return {**meta, **r}


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pv", required=True, help="PlantVillage repo checkout (contains raw/color and leaf_grouping)")
    p.add_argument("--pd-crop", required=True, help="PlantDoc cropped classification repo checkout")
    p.add_argument("--pd-det", required=True, help="PlantDoc object-detection repo checkout")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    out = Path(a.out)
    leaf_map = json.load(open(Path(a.pv) / "leaf_grouping" / "leaf-map.json"))

    pv_items = []
    root = Path(a.pv) / "raw" / "color"
    for cls in sorted(os.listdir(root)):
        for fn in sorted(os.listdir(root / cls)):
            pv_items.append((str(root / cls / fn), {"dataset": "PV", "split": "", "class": cls, "file": fn,
                                                     "group": audit.pv_leaf_id(cls, fn, leaf_map)}))
    pd_items = []
    for split in ("train", "test"):
        for cls in sorted(os.listdir(Path(a.pd_crop) / split)):
            for fn in sorted(os.listdir(Path(a.pd_crop) / split / cls)):
                pd_items.append((str(Path(a.pd_crop) / split / cls / fn),
                                 {"dataset": "PDcrop", "split": split, "class": cls, "file": fn, "group": fn}))
    det_items = []
    for split in ("TRAIN", "TEST"):
        for fn in sorted(os.listdir(Path(a.pd_det) / split)):
            if fn.endswith(".xml"):
                continue
            det_items.append((str(Path(a.pd_det) / split / fn), {"dataset": "PDdet", "split": split, "class": "",
                                                                  "file": fn, "group": fn}))
    print(f"PV {len(pv_items)}, PDcrop {len(pd_items)}, PDdet {len(det_items)} files", file=sys.stderr, flush=True)
    with Pool(a.workers) as pool:
        for name, items in (("pdcrop", pd_items), ("pddet", det_items), ("pv", pv_items)):
            rows = pool.map(work, items, chunksize=64)
            write_csv(rows, out / f"{name}_manifest.csv")
            print(f"wrote {name}: {len(rows)} rows, {sum(1 for r in rows if r['error'])} unreadable", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
