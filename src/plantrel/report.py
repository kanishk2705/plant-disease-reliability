"""Per-run evaluation on the source test and the external set, with cluster-bootstrap intervals.

Pure functions of (logits, labels, clusters, frozen T and thresholds). No fitting happens here.
"""
from __future__ import annotations

import numpy as np

from . import metrics as M
from .bootstrap import cluster_bootstrap, gap_bootstrap, paired_bootstrap
from .evaluation import bundle


def _conf_ok(p, y):
    return M.confidence_and_correct(p, y)


def make_stats(n_classes: int, tau: float | None):
    cls = list(range(n_classes))
    stats = {
        "accuracy": lambda p, y: M.accuracy(y, p.argmax(1)),
        "macro_f1": lambda p, y: M.macro_f1(y, p.argmax(1), cls),
        "ece15_width": lambda p, y: M.ece(p, y, 15, "equal_width"),
        "brier_sum": lambda p, y: M.brier_score(p, y),
        "nll": lambda p, y: M.nll(p, y),
        "aurc": lambda p, y: M.aurc(*_conf_ok(p, y)),
        "auroc_error": lambda p, y: M.auroc_error_detection(*_conf_ok(p, y)),
    }
    if tau is not None:
        def sel_risk(p, y):
            c, ok = _conf_ok(p, y); return M.selective_metrics(c, ok, tau).risk
        def sel_cov(p, y):
            c, ok = _conf_ok(p, y); return M.selective_metrics(c, ok, tau).coverage
        stats["selective_risk"], stats["coverage"] = sel_risk, sel_cov
    return stats


def ci(res) -> dict:
    return {"point": res.point, "lo": res.lo, "hi": res.hi, "n_undefined": res.n_undefined, "n_clusters": res.n_clusters}


def bootstrap_set(logits, y, clusters, T, n_classes, tau, n_boot, seed):
    p = M.softmax(logits, T)
    out = {}
    for name, f in make_stats(n_classes, tau).items():
        out[name] = ci(cluster_bootstrap({"p": p, "y": y}, clusters, f, n_boot=n_boot, seed=seed))
    return out


def plot_data(logits, y, T, n_classes, n_points=60):
    p = M.softmax(logits, T)
    conf, ok = _conf_ok(p, y)
    bins = M.calibration_bins(conf, ok, 15, "equal_width")
    cur = M.risk_coverage_curve(conf, ok)
    sel = np.unique(np.linspace(0, len(cur["coverage"]) - 1, n_points).astype(int))
    pred = p.argmax(1)
    pc = M.per_class_metrics(y, pred, list(range(n_classes)))
    return {"reliability_bins": bins,
            "rc_curve": {"coverage": cur["coverage"][sel].tolist(), "risk": cur["risk"][sel].tolist()},
            "per_class": pc, "confusion": M.confusion_matrix(y, pred, list(range(n_classes))).tolist(),
            "conf_hist_edges": np.linspace(0, 1, 11).tolist(),
            "conf_hist_correct": np.histogram(conf[ok], bins=np.linspace(0, 1, 11))[0].tolist(),
            "conf_hist_wrong": np.histogram(conf[~ok], bins=np.linspace(0, 1, 11))[0].tolist(),
            "high_conf_errors_ge_0.9": int(((conf >= 0.9) & ~ok).sum()), "n": int(len(y))}


def gaps(src_logits, src_y, src_cl, ext_logits, ext_y, ext_cl, T, n_classes, n_boot, seed):
    ps, pe = M.softmax(src_logits, T), M.softmax(ext_logits, T)
    st = make_stats(n_classes, None)
    out = {}
    for name in ("accuracy", "macro_f1", "ece15_width", "brier_sum", "nll"):
        out[name] = ci(gap_bootstrap({"p": ps, "y": src_y}, src_cl, {"p": pe, "y": ext_y}, ext_cl, st[name], n_boot=n_boot, seed=seed))
    return out


def paired_models(logits_a, T_a, logits_b, T_b, y, clusters, n_classes, n_boot, seed):
    """Statistic(A) - statistic(B) on the same examples (A = ResNet-18, B = CNN by convention)."""
    pa, pb = M.softmax(logits_a, T_a), M.softmax(logits_b, T_b)
    st = make_stats(n_classes, None)
    out = {}
    for name in ("accuracy", "macro_f1", "ece15_width", "brier_sum", "nll", "aurc"):
        out[name] = ci(paired_bootstrap({"p": pa, "y": y}, {"p": pb, "y": y}, clusters, st[name], n_boot=n_boot, seed=seed))
    return out


def to_jsonable(o):
    """json.dumps default hook: numpy scalars/arrays to plain Python (NaN/inf become None)."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        o = o.item()
        return None if isinstance(o, float) and not np.isfinite(o) else o
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")
