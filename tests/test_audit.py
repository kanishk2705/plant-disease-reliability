import numpy as np
from PIL import Image

from plantrel import audit


def test_pv_leaf_id_follows_loader_rules():
    lm = {"rs_hl 5672": ["Apple___healthy:::12.0"], "multi 1": ["A___x:::1.0", "B___y:::2.0"]}
    assert audit.pv_leaf_id("Apple___healthy", "0055dd26-23a7___RS_HL 5672.JPG", lm) == "Apple___healthy:::12.0"
    assert audit.pv_leaf_id("B___y", "u___multi 1.JPG", lm) == "B___y:::2.0"
    assert audit.pv_leaf_id("C___z", "u___multi 1.JPG", lm) == "fallback_multi 1"
    assert audit.pv_leaf_id("A___x", "u___unseen 9.JPG", lm) == "fallback_unseen 9"
    assert audit.pv_identifier("u___RS_HL 5672 copy.JPG") == "RS_HL 5672"


def test_phash_identical_and_different():
    rng = np.random.default_rng(0)
    a = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    b = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    assert audit.phash64(a) == audit.phash64(a.copy())
    assert audit.hamming(audit.phash64(a), audit.phash64(b)) > 8


def test_near_pairs_exact_against_bruteforce():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 2**63, 40, dtype=np.uint64)
    hs = list(base)
    for k in range(0, 40, 3):  # add near copies with 1..6 flipped bits
        v = int(base[k])
        for bit in rng.choice(64, size=1 + k % 6, replace=False):
            v ^= 1 << int(bit)
        hs.append(np.uint64(v))
    hs = np.array(hs, dtype=np.uint64)
    for t in (0, 3, 7):
        got = {(i, j): d for i, j, d in audit.near_pairs(hs, t)}
        want = {(i, j): audit.hamming(int(hs[i]), int(hs[j])) for i in range(len(hs)) for j in range(i + 1, len(hs))
                if audit.hamming(int(hs[i]), int(hs[j])) <= t}
        assert got == want


def test_cross_nearest_and_components():
    a = np.array([0b1111, 0b0], dtype=np.uint64)
    b = np.array([0b1110, 0xFFFF], dtype=np.uint64)
    d, idx = audit.cross_nearest(a, b)
    assert d.tolist() == [1, 3] and idx.tolist() == [0, 0]
    labels = audit.connected_components(5, [(0, 1), (1, 2), (3, 4)])
    assert labels[0] == labels[1] == labels[2] != labels[3] == labels[4]
