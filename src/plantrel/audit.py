"""Dataset-audit helpers (Phase 3). Pure functions over files, labels and hashes; no model is involved.

* ``pv_leaf_id`` replicates the leaf-id derivation in the PlantVillage repository's Hugging Face
  loader (``plant_village.py``), so the grouping we split on is the one the dataset authors publish.
* ``phash64`` is a DCT perceptual hash (32x32 grayscale, low-frequency 8x8 block, median threshold).
* ``near_pairs`` finds all pairs within a Hamming distance <= 7 exactly, using 8 byte-blocks
  (pigeonhole: two 64-bit hashes differing in <= 7 bits agree in at least one byte).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def pv_identifier(file_name: str) -> str:
    ident = file_name.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    for ext in (".jpg", ".JPG", ".png", ".PNG"):
        ident = ident.replace(ext, "")
    return ident.strip()


def pv_leaf_id(class_name: str, file_name: str, leaf_map: dict) -> str:
    ident = pv_identifier(file_name)
    key = ident.lower().strip()
    if key in leaf_map:
        sugg = leaf_map[key]
        if len(sugg) == 1:
            return sugg[0]
        for s in sugg:
            if class_name in s:
                return s
        return f"fallback_{ident}"
    return f"fallback_{ident}"


def phash64(img: Image.Image) -> int:
    g = img.convert("L").resize((32, 32), Image.LANCZOS)
    a = dctn(np.asarray(g, dtype=np.float64), type=2, norm="ortho")[:8, :8].flatten()
    med = np.median(a[1:])
    bits = (a > med).astype(np.uint64)
    return int(sum(int(b) << i for i, b in enumerate(bits)))


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _bytes(h: np.ndarray) -> np.ndarray:
    return h.astype("<u8").view(np.uint8).reshape(-1, 8)


def near_pairs(hashes, max_dist: int) -> list[tuple[int, int, int]]:
    """All index pairs (i<j) with Hamming distance <= max_dist (max_dist <= 7), exact."""
    if not 0 <= max_dist <= 7:
        raise ValueError("max_dist must be in 0..7")
    h = np.asarray(hashes, dtype=np.uint64)
    by = _bytes(h)
    seen: set[tuple[int, int]] = set()
    out = []
    for blk in range(8):
        order = np.argsort(by[:, blk], kind="stable")
        keys = by[order, blk]
        bounds = np.flatnonzero(np.diff(keys)) + 1
        for grp in np.split(order, bounds):
            if grp.size < 2:
                continue
            sub = h[grp]
            x = sub[:, None] ^ sub[None, :]
            d = _POP8[x.astype("<u8").view(np.uint8).reshape(*x.shape, 8)].sum(axis=-1)
            ii, jj = np.nonzero(np.triu(d <= max_dist, k=1))
            for a, b in zip(ii.tolist(), jj.tolist()):
                i, j = int(grp[a]), int(grp[b])
                if i > j:
                    i, j = j, i
                if (i, j) not in seen:
                    seen.add((i, j))
                    out.append((i, j, int(d[a, b])))
    return sorted(out)


def cross_nearest(a_hashes, b_hashes, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """For each hash in A, the minimum Hamming distance to any hash in B and the index of that match."""
    a = np.asarray(a_hashes, dtype=np.uint64)
    b = np.asarray(b_hashes, dtype=np.uint64)
    best = np.empty(a.size, dtype=np.int16)
    arg = np.empty(a.size, dtype=np.int64)
    for s in range(0, a.size, chunk):
        x = a[s:s + chunk, None] ^ b[None, :]
        d = _POP8[x.astype("<u8").view(np.uint8).reshape(*x.shape, 8)].sum(axis=-1)
        arg[s:s + chunk] = d.argmin(axis=1)
        best[s:s + chunk] = d.min(axis=1)
    return best, arg


def connected_components(n: int, edges) -> np.ndarray:
    """Union-find labels 0..k-1 for n nodes joined by (i, j, ...) edges."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        ra, rb = find(e[0]), find(e[1])
        if ra != rb:
            parent[rb] = ra
    roots = {}
    return np.array([roots.setdefault(find(i), len(roots)) for i in range(n)])
