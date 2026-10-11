"""Cluster bootstrap (protocol section 7).

Clusters, not individual rows, are resampled with replacement: ``leaf_id`` for the source test
and the external cluster id (source image or duplicate-group component) for the external set.
Intervals are percentile intervals. Replicates in which the statistic is undefined (NaN) are
excluded from the interval and counted in ``n_undefined``; they are never turned into zeros.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

Data = Mapping[str, np.ndarray]
Statistic = Callable[..., float]


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    lo: float
    hi: float
    alpha: float
    n_boot: int
    n_clusters: int
    n_undefined: int
    replicates: np.ndarray


def _prepare(data: Data, clusters) -> tuple[dict, np.ndarray]:
    arrays = {k: np.asarray(v) for k, v in data.items()}
    cl = np.asarray(clusters)
    if cl.ndim != 1 or cl.size == 0:
        raise ValueError("clusters must be a non-empty 1-D array")
    for k, v in arrays.items():
        if v.shape[0] != cl.shape[0]:
            raise ValueError(f"array {k!r} length differs from clusters")
    return arrays, cl


def _members(clusters: np.ndarray) -> list[np.ndarray]:
    _, inverse = np.unique(clusters, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    return np.split(order, np.cumsum(counts)[:-1])


def _draw(members: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    picks = rng.integers(0, len(members), size=len(members))
    return np.concatenate([members[i] for i in picks])


def _subset(arrays: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in arrays.items()}


def _interval(point: float, reps: np.ndarray, alpha: float, n_clusters: int, n_boot: int) -> BootstrapResult:
    finite = reps[np.isfinite(reps)]
    if finite.size == 0:
        lo = hi = float("nan")
    else:
        lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(float(point), float(lo), float(hi), alpha, n_boot, n_clusters,
                           int(reps.size - finite.size), reps)


def _stat(statistic: Statistic, arrays: dict) -> float:
    return float(statistic(**arrays))


def cluster_bootstrap(data: Data, clusters, statistic: Statistic, *, n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> BootstrapResult:
    """Percentile interval for ``statistic(**data)`` resampling whole clusters."""
    arrays, cl = _prepare(data, clusters)
    members = _members(cl)
    rng = np.random.default_rng(seed)
    point = _stat(statistic, arrays)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        reps[b] = _stat(statistic, _subset(arrays, _draw(members, rng)))
    return _interval(point, reps, alpha, len(members), n_boot)


def paired_bootstrap(data_a: Data, data_b: Data, clusters, statistic: Statistic, *, n_boot: int = 2000,
                     seed: int = 0, alpha: float = 0.05) -> BootstrapResult:
    """Interval for statistic(A) - statistic(B) on the SAME examples, resampling the same clusters."""
    arrays_a, cl = _prepare(data_a, clusters)
    arrays_b, _ = _prepare(data_b, clusters)
    members = _members(cl)
    rng = np.random.default_rng(seed)
    point = _stat(statistic, arrays_a) - _stat(statistic, arrays_b)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = _draw(members, rng)
        reps[b] = _stat(statistic, _subset(arrays_a, idx)) - _stat(statistic, _subset(arrays_b, idx))
    return _interval(point, reps, alpha, len(members), n_boot)


def gap_bootstrap(src_data: Data, src_clusters, ext_data: Data, ext_clusters, statistic_src: Statistic,
                  statistic_ext: Statistic | None = None, *, n_boot: int = 2000, seed: int = 0,
                  alpha: float = 0.05) -> BootstrapResult:
    """Interval for statistic(source) - statistic(external); the two sets are resampled independently."""
    s_arrays, s_cl = _prepare(src_data, src_clusters)
    e_arrays, e_cl = _prepare(ext_data, ext_clusters)
    stat_e = statistic_ext or statistic_src
    s_members, e_members = _members(s_cl), _members(e_cl)
    rng = np.random.default_rng(seed)
    point = _stat(statistic_src, s_arrays) - _stat(stat_e, e_arrays)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        si = _draw(s_members, rng)
        ei = _draw(e_members, rng)
        reps[b] = _stat(statistic_src, _subset(s_arrays, si)) - _stat(stat_e, _subset(e_arrays, ei))
    return _interval(point, reps, alpha, len(s_members) + len(e_members), n_boot)
