import numpy as np
import pytest

from plantrel import bootstrap as bs
from plantrel import metrics as m


def mean_x(x):
    return float(np.mean(x))


def n_rows(x):
    return float(len(x))


def test_clusters_are_resampled_whole_not_row_by_row():
    # clusters of sizes 3 and 1 -> a replicate has 2, 4 or 6 rows; row-level resampling would always give 4
    x = np.arange(4.0)
    r = bs.cluster_bootstrap({"x": x}, [0, 0, 0, 1], n_rows, n_boot=600, seed=1)
    assert set(np.unique(r.replicates)) == {2.0, 4.0, 6.0}
    assert r.n_clusters == 2 and r.point == 4.0


def test_deterministic_given_seed_and_different_for_other_seed():
    x = np.random.default_rng(0).normal(size=40)
    cl = np.repeat(np.arange(10), 4)
    a = bs.cluster_bootstrap({"x": x}, cl, mean_x, n_boot=200, seed=5)
    b = bs.cluster_bootstrap({"x": x}, cl, mean_x, n_boot=200, seed=5)
    c = bs.cluster_bootstrap({"x": x}, cl, mean_x, n_boot=200, seed=6)
    assert np.array_equal(a.replicates, b.replicates) and not np.array_equal(a.replicates, c.replicates)


def test_clustered_interval_is_wider_than_row_level_for_correlated_clusters():
    # every row in a cluster has the same value: rows are not independent evidence
    vals = np.repeat(np.random.default_rng(2).normal(size=15), 10)
    cl = np.repeat(np.arange(15), 10)
    clustered = bs.cluster_bootstrap({"x": vals}, cl, mean_x, n_boot=800, seed=0)
    per_row = bs.cluster_bootstrap({"x": vals}, np.arange(150), mean_x, n_boot=800, seed=0)
    assert (clustered.hi - clustered.lo) > 1.5 * (per_row.hi - per_row.lo)


def test_interval_contains_point_for_symmetric_case():
    x = np.random.default_rng(0).normal(size=200)
    r = bs.cluster_bootstrap({"x": x}, np.arange(200), mean_x, n_boot=500, seed=0)
    assert r.lo < r.point < r.hi


def test_validation_of_lengths():
    with pytest.raises(ValueError):
        bs.cluster_bootstrap({"x": np.arange(3.0)}, [0, 1], mean_x, n_boot=5)


def test_paired_bootstrap_uses_identical_resamples():
    x = np.random.default_rng(0).normal(size=30)
    cl = np.repeat(np.arange(10), 3)
    r = bs.paired_bootstrap({"x": x + 1.0}, {"x": x}, cl, mean_x, n_boot=300, seed=0)
    assert np.allclose(r.replicates, 1.0) and r.point == pytest.approx(1.0)
    assert r.lo == pytest.approx(1.0) and r.hi == pytest.approx(1.0)


def test_paired_model_comparison_on_accuracy():
    n = 60
    y = np.tile([0, 1, 2], n // 3)
    good = {"y_true": y, "y_pred": y.copy()}
    bad_pred = y.copy()
    bad_pred[::2] = (bad_pred[::2] + 1) % 3
    bad = {"y_true": y, "y_pred": bad_pred}
    r = bs.paired_bootstrap(good, bad, np.repeat(np.arange(20), 3), m.accuracy, n_boot=300, seed=1)
    assert r.point == pytest.approx(0.5) and r.lo > 0


def test_gap_bootstrap_independent_sets_and_point_estimate():
    src = {"x": np.full(20, 1.0)}
    ext = {"x": np.repeat([0.0] * 5 + [0.5] * 5, 2)}  # whole clusters are 0 or 0.5
    r = bs.gap_bootstrap(src, np.arange(20), ext, np.repeat(np.arange(10), 2), mean_x, n_boot=300, seed=0)
    assert r.point == pytest.approx(1.0 - 0.25) and r.n_clusters == 30
    assert r.lo < r.point < r.hi


def test_gap_uses_separate_statistics_for_source_and_external():
    r = bs.gap_bootstrap({"x": np.ones(4)}, np.arange(4), {"x": np.ones(4)}, np.arange(4), mean_x,
                         lambda x: 0.25 * float(np.mean(x)), n_boot=20, seed=0)
    assert r.point == pytest.approx(0.75)


def test_macro_f1_class_list_is_fixed_across_replicates():
    # one perfectly predicted example per class, each its own cluster. A replicate containing d distinct
    # classes must score d/3 (absent classes count as F1 = 0 in a denominator that stays 3).
    y = np.array([0, 1, 2])
    stat = lambda y_true, y_pred: m.macro_f1(y_true, y_pred, [0, 1, 2])
    r = bs.cluster_bootstrap({"y_true": y, "y_pred": y}, [0, 1, 2], stat, n_boot=400, seed=0)
    vals = set(np.round(r.replicates * 3).astype(int).tolist())
    assert vals <= {1, 2, 3} and {1, 2, 3} <= vals
    assert r.replicates.max() == pytest.approx(1.0) and r.replicates.min() == pytest.approx(1 / 3)


def test_undefined_replicates_are_counted_not_zeroed():
    # cluster 0 has confidence .9 (correct), cluster 1 has .1 (wrong); tau=.5 accepts nothing when only
    # cluster 1 is drawn -> selective risk undefined in about a quarter of replicates
    data = {"conf": np.array([0.9, 0.1]), "correct": np.array([True, False])}
    stat = lambda conf, correct: m.selective_metrics(conf, correct, 0.5).risk
    r = bs.cluster_bootstrap(data, [0, 1], stat, n_boot=400, seed=0)
    assert r.n_undefined > 0
    assert np.isnan(r.replicates).sum() == r.n_undefined
    assert r.lo == 0.0 and r.hi == 0.0  # all defined replicates have risk 0


def test_all_replicates_undefined_gives_nan_interval():
    data = {"conf": np.array([0.1, 0.1]), "correct": np.array([True, False])}
    stat = lambda conf, correct: m.selective_metrics(conf, correct, 0.5).risk
    r = bs.cluster_bootstrap(data, [0, 1], stat, n_boot=20, seed=0)
    assert r.n_undefined == 20 and np.isnan(r.lo) and np.isnan(r.hi)


def test_selective_metrics_bootstrap_with_related_observations():
    # six crops from three source images (two each); resampling must move whole images
    conf = np.array([0.9, 0.9, 0.6, 0.6, 0.3, 0.3])
    correct = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    cl = np.array([0, 0, 1, 1, 2, 2])
    r = bs.cluster_bootstrap({"conf": conf, "correct": correct}, cl, lambda conf, correct: m.aurc(conf, correct),
                             n_boot=200, seed=0)
    assert r.n_clusters == 3 and np.isfinite(r.lo)
