import numpy as np
from plantrel import evaluation as E, report, metrics as M


def _data(n=600, k=4, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n); lg = rng.normal(size=(n, k)) * 2; lg[np.arange(n), y] += 2.5
    return lg, y, rng.integers(0, 80, n)


def test_calibrate_uses_only_given_data_and_reduces_nll():
    lg, y, _ = _data()
    c = E.calibrate(lg, y)
    assert c["val_cal_nll_after"] <= c["val_cal_nll_before"] + 1e-9
    p = M.softmax(lg, c["T"]); conf, ok = M.confidence_and_correct(p, y)
    assert M.selective_metrics(conf, ok, c["tau_r_cov"]).coverage >= 0.9 - 1e-9


def test_bundle_fields_and_selective_counts():
    lg, y, _ = _data(); c = E.calibrate(lg, y)
    b = E.bundle(lg, y, c["T"], list(range(4)), {"r_cov": c["tau_r_cov"], "r_risk": c["tau_r_risk"]})
    s = b["cal"]["selective"]["r_cov"]
    assert s["n_accepted"] + s["n_rejected"] == 600 and 0 <= b["cal"]["macro_f1"] <= 1


def test_bootstrap_interval_contains_point_and_gap_and_pair_run():
    lg, y, cl = _data(); lg2, y2, cl2 = _data(seed=1)
    r = report.bootstrap_set(lg, y, cl, 1.0, 4, 0.5, 100, 0)
    assert r["accuracy"]["lo"] <= r["accuracy"]["point"] <= r["accuracy"]["hi"]
    g = report.gaps(lg, y, cl, lg2, y2, cl2, 1.0, 4, 50, 0); assert "macro_f1" in g
    pr = report.paired_models(lg, 1.0, lg2, 1.0, y, cl, 4, 50, 0); assert abs(pr["accuracy"]["point"]) <= 1
