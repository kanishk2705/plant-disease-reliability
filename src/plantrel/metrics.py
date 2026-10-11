"""Metrics specified in protocol sections 6 and 7.

Conventions (all stated in the protocol):
* Macro-F1 uses an explicit, fixed class list and ``zero_division=0``: precision is 0 when a
  class has no predictions, recall is 0 when it has no true examples, F1 is 0 when
  precision + recall = 0. The denominator is always the list length.
* Brier score is the SUM over classes of squared error, averaged over examples (range [0, 2]).
* ECE uses the maximum predicted probability as confidence and the L1 norm. Equal-width bins
  are (lo, hi] (the first bin also contains 0). Equal-mass bins split the sorted examples into
  nearly equal groups; ties are ordered by (confidence, correctness) so the result does not
  depend on input order.
* Selective classification accepts an example when confidence >= tau. Selective risk is
  undefined (NaN) when nothing is accepted; it is never reported as zero.
* AURC is sum_j (coverage_j - coverage_{j-1}) * risk_j over distinct confidence thresholds,
  with coverage_0 = 0. Without ties this is the mean selective risk over k = 1..N accepted.
* AUROC for error detection: positive = incorrect prediction, score = 1 - max probability,
  ties get average ranks. NaN if the labels are all one class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import rankdata

# ---------------------------------------------------------------- classification


def _labels(x, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return a


def accuracy(y_true, y_pred) -> float:
    t, p = _labels(y_true, "y_true"), _labels(y_pred, "y_pred")
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if t.size == 0:
        raise ValueError("accuracy is undefined for an empty set")
    return float(np.mean(t == p))


def _index_map(classes: Sequence) -> dict:
    classes = list(classes)
    if len(classes) == 0:
        raise ValueError("class list is empty")
    if len(set(classes)) != len(classes):
        raise ValueError("class list contains duplicates")
    return {c: i for i, c in enumerate(classes)}


def confusion_matrix(y_true, y_pred, classes: Sequence, normalize: bool = False) -> np.ndarray:
    """Rows = true class, columns = predicted class, in the order of ``classes``.

    Labels outside ``classes`` raise ValueError. With ``normalize=True`` rows are divided by
    their sum; rows with no true examples are NaN (not zero).
    """
    idx = _index_map(classes)
    t, p = _labels(y_true, "y_true"), _labels(y_pred, "y_pred")
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same length")
    try:
        ti = np.fromiter((idx[v] for v in t.tolist()), dtype=np.int64, count=t.size)
        pi = np.fromiter((idx[v] for v in p.tolist()), dtype=np.int64, count=p.size)
    except KeyError as exc:
        raise ValueError(f"label {exc.args[0]!r} is not in the class list") from None
    k = len(idx)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (ti, pi), 1)
    if not normalize:
        return cm
    sums = cm.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = cm / sums
    out[sums[:, 0] == 0] = np.nan
    return out


def per_class_metrics(y_true, y_pred, classes: Sequence) -> dict:
    """Precision, recall, F1 and support for every class in the fixed list (zero_division=0)."""
    cm = confusion_matrix(y_true, y_pred, classes)
    tp = np.diag(cm).astype(float)
    pred_n = cm.sum(axis=0).astype(float)
    true_n = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_n, out=np.zeros_like(tp), where=pred_n > 0)
    recall = np.divide(tp, true_n, out=np.zeros_like(tp), where=true_n > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {"classes": list(classes), "precision": precision, "recall": recall, "f1": f1,
            "support": true_n.astype(np.int64), "n_predicted": pred_n.astype(np.int64)}


def macro_f1(y_true, y_pred, classes: Sequence) -> float:
    """Unweighted mean of per-class F1 over exactly ``classes`` (never adds or drops a class)."""
    return float(np.mean(per_class_metrics(y_true, y_pred, classes)["f1"]))


# ---------------------------------------------------------------- probabilities


def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("logits must be 2-D (N, K)")
    if not temperature > 0:
        raise ValueError("temperature must be > 0")
    z = z / temperature
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))


def _probs(probs, labels=None, atol: float = 1e-5):
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        raise ValueError("probs must be a non-empty 2-D array (N, K)")
    if not np.all(np.isfinite(p)) or p.min() < 0:
        raise ValueError("probs must be finite and non-negative")
    if not np.allclose(p.sum(axis=1), 1.0, atol=atol):
        raise ValueError("each probability row must sum to 1")
    if labels is None:
        return p, None
    y = np.asarray(labels)
    if y.shape != (p.shape[0],) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels must be integer class indices, one per row")
    if y.min() < 0 or y.max() >= p.shape[1]:
        raise ValueError("label index out of range")
    return p, y


def confidence_and_correct(probs, labels) -> tuple[np.ndarray, np.ndarray]:
    p, y = _probs(probs, labels)
    return p.max(axis=1), (p.argmax(axis=1) == y)


def brier_score(probs, labels) -> float:
    """Multiclass Brier, SUM over classes, mean over examples. Range [0, 2]."""
    p, y = _probs(probs, labels)
    onehot = np.zeros_like(p)
    onehot[np.arange(p.shape[0]), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def nll(probs, labels, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood; the true-class probability is floored at ``eps``."""
    p, y = _probs(probs, labels)
    return float(-np.mean(np.log(np.maximum(p[np.arange(p.shape[0]), y], eps))))


def nll_from_logits(logits, labels, temperature: float = 1.0) -> float:
    z = np.asarray(logits, dtype=np.float64) / temperature
    y = np.asarray(labels)
    logp = z - logsumexp(z, axis=1, keepdims=True)
    return float(-np.mean(logp[np.arange(z.shape[0]), y]))


def fit_temperature(logits, labels, bounds: tuple[float, float] = (0.05, 20.0)) -> float:
    """Single scalar T minimising NLL (optimised over log T within ``bounds``)."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    res = minimize_scalar(lambda u: nll_from_logits(z, y, float(np.exp(u))), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-8})
    return float(np.exp(res.x))


# ---------------------------------------------------------------- ECE


def calibration_bins(conf, correct, n_bins: int = 15, scheme: str = "equal_width") -> list[dict]:
    """Per-bin count, mean confidence and accuracy (empty bins have count 0 and NaN means)."""
    c = np.asarray(conf, dtype=np.float64)
    ok = np.asarray(correct, dtype=bool)
    if c.ndim != 1 or c.shape != ok.shape or c.size == 0:
        raise ValueError("conf and correct must be non-empty 1-D arrays of the same length")
    if c.min() < 0 or c.max() > 1 or not np.all(np.isfinite(c)):
        raise ValueError("confidence must lie in [0, 1]")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    groups: list[np.ndarray]
    if scheme == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        which = np.searchsorted(edges[1:-1], c, side="left")  # (lo, hi]; 0 falls in bin 0
        groups = [np.flatnonzero(which == b) for b in range(n_bins)]
    elif scheme == "equal_mass":
        order = np.lexsort((ok, c))  # sort by confidence, then correctness: input-order independent
        groups = [g for g in np.array_split(order, n_bins)]
    else:
        raise ValueError("scheme must be 'equal_width' or 'equal_mass'")
    out = []
    for g in groups:
        if g.size == 0:
            out.append({"count": 0, "mean_confidence": float("nan"), "accuracy": float("nan")})
        else:
            out.append({"count": int(g.size), "mean_confidence": float(c[g].mean()), "accuracy": float(ok[g].mean())})
    return out


def ece_from_confidence(conf, correct, n_bins: int = 15, scheme: str = "equal_width") -> float:
    bins = calibration_bins(conf, correct, n_bins, scheme)
    n = sum(b["count"] for b in bins)
    return float(sum(b["count"] / n * abs(b["accuracy"] - b["mean_confidence"]) for b in bins if b["count"] > 0))


def ece(probs, labels, n_bins: int = 15, scheme: str = "equal_width") -> float:
    conf, correct = confidence_and_correct(probs, labels)
    return ece_from_confidence(conf, correct, n_bins, scheme)


def signed_overconfidence(probs, labels) -> float:
    """Mean confidence minus accuracy (positive = overconfident)."""
    conf, correct = confidence_and_correct(probs, labels)
    return float(conf.mean() - correct.mean())


# ---------------------------------------------------------------- selective classification


@dataclass(frozen=True)
class SelectiveResult:
    threshold: float
    n_total: int
    n_accepted: int
    n_rejected: int
    n_errors_accepted: int
    coverage: float
    risk: float  # NaN when nothing is accepted: undefined, never zero


def _conf_correct(conf, correct):
    c = np.asarray(conf, dtype=np.float64)
    ok = np.asarray(correct, dtype=bool)
    if c.ndim != 1 or c.shape != ok.shape or c.size == 0:
        raise ValueError("conf and correct must be non-empty 1-D arrays of the same length")
    if not np.all(np.isfinite(c)):
        raise ValueError("confidence must be finite")
    return c, ok


def selective_metrics(conf, correct, tau: float) -> SelectiveResult:
    c, ok = _conf_correct(conf, correct)
    accept = c >= tau
    n_acc = int(accept.sum())
    n_err = int((~ok[accept]).sum())
    risk = float("nan") if n_acc == 0 else n_err / n_acc
    return SelectiveResult(float(tau), int(c.size), n_acc, int(c.size - n_acc), n_err, n_acc / c.size, risk)


def risk_coverage_curve(conf, correct) -> dict:
    """One point per distinct confidence value, accepting everything with conf >= threshold.

    Returned in order of increasing coverage (decreasing threshold).
    """
    c, ok = _conf_correct(conf, correct)
    order = np.argsort(-c, kind="stable")
    cs, errs = c[order], (~ok[order]).astype(np.int64)
    last_of_group = np.r_[cs[1:] != cs[:-1], True]  # last index of each tied run
    n_acc = np.arange(1, c.size + 1)[last_of_group]
    n_err = np.cumsum(errs)[last_of_group]
    return {"threshold": cs[last_of_group], "coverage": n_acc / c.size, "risk": n_err / n_acc,
            "n_accepted": n_acc, "n_errors": n_err}


def aurc(conf, correct) -> float:
    cur = risk_coverage_curve(conf, correct)
    cov = np.r_[0.0, cur["coverage"]]
    return float(np.sum(np.diff(cov) * cur["risk"]))


def aurc_random(correct) -> float:
    """Random-rejection reference: constant risk equal to the full-coverage error rate."""
    ok = np.asarray(correct, dtype=bool)
    if ok.size == 0:
        raise ValueError("empty input")
    return float(np.mean(~ok))


def aurc_oracle(correct) -> float:
    """Oracle reference: all correct examples are accepted before any error."""
    ok = np.asarray(correct, dtype=bool)
    n = ok.size
    if n == 0:
        raise ValueError("empty input")
    n_correct = int(ok.sum())
    k = np.arange(1, n + 1)
    return float(np.mean(np.maximum(0, k - n_correct) / k))


def threshold_for_coverage(conf, target_coverage: float) -> float:
    """R-cov: the largest observed confidence value whose coverage is >= ``target_coverage``."""
    if not 0 < target_coverage <= 1:
        raise ValueError("target_coverage must be in (0, 1]")
    cur = risk_coverage_curve(conf, np.ones(np.asarray(conf).size, dtype=bool))
    ok = cur["coverage"] >= target_coverage - 1e-12
    return float(cur["threshold"][np.flatnonzero(ok)[0]])


def threshold_for_risk(conf, correct, max_risk: float) -> float | None:
    """R-risk: the smallest observed confidence value whose selective risk is <= ``max_risk``.

    Returns None when no threshold achieves it.
    """
    cur = risk_coverage_curve(conf, correct)
    ok = cur["risk"] <= max_risk + 1e-12
    if not ok.any():
        return None
    return float(cur["threshold"][np.flatnonzero(ok)[-1]])


def auroc_error_detection(conf, correct) -> float:
    """AUROC of ``1 - confidence`` for detecting incorrect predictions (positive = incorrect).

    Mann-Whitney form with average ranks for ties. NaN when all predictions are correct or all
    are incorrect.
    """
    c, ok = _conf_correct(conf, correct)
    positive = ~ok
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    score = 1.0 - c
    ranks = rankdata(score, method="average")
    u = ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))
