"""Hand-calculated checks for plantrel.metrics. All expected numbers are derived in comments."""
import numpy as np
import pytest

from plantrel import metrics as m


# ---- classification ------------------------------------------------------------------
# y_true=[0,0,1,1,2,2], y_pred=[0,1,1,1,2,0]
# confusion rows (true) -> [1,1,0],[0,2,0],[1,0,1]
# precision = [1/2, 2/3, 1]; recall = [1/2, 1, 1/2]; F1 = [0.5, 0.8, 2/3]; macro = 0.655556
YT = [0, 0, 1, 1, 2, 2]
YP = [0, 1, 1, 1, 2, 0]


def test_accuracy():
    assert m.accuracy(YT, YP) == pytest.approx(4 / 6)


def test_accuracy_empty_raises():
    with pytest.raises(ValueError):
        m.accuracy([], [])


def test_confusion_matrix_counts_and_normalised():
    cm = m.confusion_matrix(YT, YP, [0, 1, 2])
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    nm = m.confusion_matrix(YT, YP, [0, 1, 2], normalize=True)
    assert nm[0].tolist() == [0.5, 0.5, 0.0]


def test_per_class_and_macro_f1():
    pc = m.per_class_metrics(YT, YP, [0, 1, 2])
    assert pc["precision"] == pytest.approx([0.5, 2 / 3, 1.0])
    assert pc["recall"] == pytest.approx([0.5, 1.0, 0.5])
    assert pc["f1"] == pytest.approx([0.5, 0.8, 2 / 3])
    assert pc["support"].tolist() == [2, 2, 2]
    assert m.macro_f1(YT, YP, [0, 1, 2]) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_absent_class_stays_in_denominator_with_zero_f1():
    # class 3 has no true examples and no predictions: F1 = 0 by the zero-division rule,
    # and the average is over 4 classes, so the class list is never shrunk.
    assert m.macro_f1(YT, YP, [0, 1, 2, 3]) == pytest.approx((0.5 + 0.8 + 2 / 3) / 4)
    pc = m.per_class_metrics(YT, YP, [0, 1, 2, 3])
    assert pc["support"][3] == 0 and pc["f1"][3] == 0.0


def test_predicted_but_absent_class_scores_zero():
    # truth only has 0 and 1; model predicts class 2 once. Class 2: no true -> recall 0, precision 0.
    pc = m.per_class_metrics([0, 1], [0, 2], [0, 1, 2])
    assert pc["f1"].tolist() == [1.0, 0.0, 0.0]
    assert pc["precision"][2] == 0.0


def test_class_without_predictions_has_zero_precision_and_f1():
    pc = m.per_class_metrics([0, 1], [0, 0], [0, 1])
    assert pc["precision"][1] == 0.0 and pc["recall"][1] == 0.0 and pc["f1"][1] == 0.0


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        m.confusion_matrix([0, 5], [0, 1], [0, 1])


def test_empty_support_row_is_nan_when_normalised():
    nm = m.confusion_matrix(YT, YP, [0, 1, 2, 3], normalize=True)
    assert np.isnan(nm[3]).all()


def test_string_labels_and_class_order_follow_the_list():
    cm = m.confusion_matrix(["b", "a"], ["b", "b"], ["a", "b"])
    assert cm.tolist() == [[0, 1], [0, 1]]


# ---- probability scores -------------------------------------------------------------
P2 = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])


def test_brier_sum_convention():
    # row1: 0.3^2+0.2^2+0.1^2 = 0.14 ; row2: 0.1^2+0.2^2+0.1^2 = 0.06 ; mean 0.10
    assert m.brier_score(P2, [0, 1]) == pytest.approx(0.10)


def test_brier_range_is_zero_to_two():
    assert m.brier_score(np.array([[1.0, 0.0]]), [0]) == pytest.approx(0.0)
    assert m.brier_score(np.array([[0.0, 1.0]]), [0]) == pytest.approx(2.0)


def test_nll():
    assert m.nll(P2, [0, 1]) == pytest.approx((-np.log(0.7) - np.log(0.8)) / 2)


def test_nll_floors_zero_probability():
    assert np.isfinite(m.nll(np.array([[0.0, 1.0]]), [0]))


def test_probs_validation():
    with pytest.raises(ValueError):
        m.brier_score(np.array([[0.5, 0.2]]), [0])  # does not sum to 1
    with pytest.raises(ValueError):
        m.nll(P2, [0, 3])  # label out of range


def test_softmax_temperature_keeps_argmax():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(50, 6)) * 3
    for T in (0.3, 1.0, 4.0):
        assert (m.softmax(z, T).argmax(1) == z.argmax(1)).all()
        assert np.allclose(m.softmax(z, T).sum(1), 1.0)


def test_fit_temperature_recovers_known_temperature():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(20000, 5)) * 3.0
    probs = m.softmax(z, 2.0)
    y = np.array([rng.choice(5, p=p) for p in probs])
    assert m.fit_temperature(z, y) == pytest.approx(2.0, abs=0.15)


# ---- ECE ------------------------------------------------------------------------------
CONF = [0.6, 0.7, 0.9, 0.95]
OK = [1, 0, 1, 1]


def test_ece_equal_width_two_bins():
    # all four fall in (0.5, 1]; mean conf 0.7875, acc 0.75 -> 0.0375
    assert m.ece_from_confidence(CONF, OK, 2) == pytest.approx(0.0375)


def test_ece_equal_width_ten_bins():
    # bins (0.5,0.6],(0.6,0.7],(0.8,0.9],(0.9,1.0]: |1-.6|+|0-.7|+|1-.9|+|1-.95| = 1.25 ; /4
    assert m.ece_from_confidence(CONF, OK, 10) == pytest.approx(0.3125)


def test_ece_bin_edge_belongs_to_lower_bin_and_zero_to_first():
    bins = m.calibration_bins([0.5, 0.0, 1.0], [1, 0, 1], 2)
    assert [b["count"] for b in bins] == [2, 1]


def test_ece_equal_mass():
    # sorted: [.6,.7] acc .5 conf .65 -> .15 ; [.9,.95] acc 1 conf .925 -> .075 ; ECE = .5*.15+.5*.075
    assert m.ece_from_confidence(CONF, OK, 2, "equal_mass") == pytest.approx(0.1125)


def test_ece_equal_mass_is_independent_of_input_order_with_ties():
    # four tied confidences 0.8, two wrong two right; ties ordered by correctness -> bins [wrong,wrong],[right,right]
    # ECE = .5*|0-.8| + .5*|1-.8| = 0.5
    conf, ok = [0.8] * 4, [1, 1, 0, 0]
    assert m.ece_from_confidence(conf, ok, 2, "equal_mass") == pytest.approx(0.5)
    assert m.ece_from_confidence(conf[::-1], ok[::-1], 2, "equal_mass") == pytest.approx(0.5)
    assert m.ece_from_confidence([0.8] * 4, [0, 1, 0, 1], 2, "equal_mass") == pytest.approx(0.5)


def test_ece_perfectly_calibrated_is_zero():
    assert m.ece_from_confidence([1.0, 1.0], [1, 1], 15) == 0.0


def test_ece_rejects_bad_input():
    with pytest.raises(ValueError):
        m.ece_from_confidence([1.2], [1])
    with pytest.raises(ValueError):
        m.ece_from_confidence([0.5], [1], scheme="nope")


def test_ece_from_probs_matches_confidence_form():
    p = np.array([[0.6, 0.4], [0.3, 0.7]])
    assert m.ece(p, [0, 0], 2) == pytest.approx(m.ece_from_confidence([0.6, 0.7], [1, 0], 2))


def test_signed_overconfidence():
    p = np.array([[0.9, 0.1], [0.9, 0.1]])
    assert m.signed_overconfidence(p, [0, 1]) == pytest.approx(0.9 - 0.5)


# ---- selective classification ------------------------------------------------------
C = [0.9, 0.8, 0.8, 0.6, 0.5]
K = [1, 1, 0, 1, 0]


def test_selective_metrics_with_ties_at_threshold():
    r = m.selective_metrics(C, K, 0.8)  # accepts .9,.8,.8 ; one error
    assert (r.n_accepted, r.n_rejected, r.n_errors_accepted) == (3, 2, 1)
    assert r.coverage == pytest.approx(0.6) and r.risk == pytest.approx(1 / 3)


def test_no_accepted_predictions_gives_undefined_risk_not_zero():
    r = m.selective_metrics(C, K, 0.95)
    assert r.n_accepted == 0 and r.coverage == 0.0 and np.isnan(r.risk)


def test_accept_everything_gives_full_coverage_risk():
    r = m.selective_metrics(C, K, 0.0)
    assert r.coverage == 1.0 and r.risk == pytest.approx(2 / 5)


def test_risk_coverage_curve_collapses_ties():
    cur = m.risk_coverage_curve(C, K)
    assert cur["threshold"].tolist() == [0.9, 0.8, 0.6, 0.5]
    assert cur["coverage"] == pytest.approx([0.2, 0.6, 0.8, 1.0])
    assert cur["risk"] == pytest.approx([0.0, 1 / 3, 0.25, 0.4])


def test_aurc_with_ties():
    # 0.2*0 + 0.4*(1/3) + 0.2*0.25 + 0.2*0.4
    assert m.aurc(C, K) == pytest.approx(0.4 / 3 + 0.05 + 0.08)


def test_aurc_without_ties_is_mean_of_risk_at_k():
    # risk@k = 0, .5, 1/3, .5
    assert m.aurc([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]) == pytest.approx((0 + 0.5 + 1 / 3 + 0.5) / 4)


def test_aurc_references():
    assert m.aurc_random(K) == pytest.approx(0.4)
    # oracle: C=3 correct, N=5: risk@k = 0,0,0,1/4,2/5 -> mean .13
    assert m.aurc_oracle(K) == pytest.approx(0.13)
    assert m.aurc_oracle(K) <= m.aurc(C, K) <= 1.0


def test_threshold_for_coverage_rcov():
    assert m.threshold_for_coverage(C, 0.5) == 0.8  # first coverage >= .5 is .6 at tau=.8
    assert m.threshold_for_coverage(C, 0.2) == 0.9
    assert m.threshold_for_coverage(C, 1.0) == 0.5


def test_threshold_for_risk_rrisk():
    assert m.threshold_for_risk(C, K, 0.3) == 0.6   # risks .0,.333,.25,.4 -> smallest tau with <= .3
    assert m.threshold_for_risk(C, K, 0.0) == 0.9
    assert m.threshold_for_risk([0.9, 0.8], [0, 0], 0.01) is None


# ---- AUROC for error detection ---------------------------------------------------------
def test_auroc_known_ranking_partial():
    # scores 1-conf = .1,.2,.3,.4 ; incorrect = idx1 (.2), idx3 (.4) ; correct = .1,.3
    # wins: (.2>.1),(.4>.1),(.4>.3) = 3 of 4
    assert m.auroc_error_detection([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]) == pytest.approx(0.75)


def test_auroc_perfect_and_reversed():
    assert m.auroc_error_detection([0.9, 0.8, 0.3, 0.2], [1, 1, 0, 0]) == 1.0
    assert m.auroc_error_detection([0.2, 0.3, 0.8, 0.9], [1, 1, 0, 0]) == 0.0


def test_auroc_all_tied_is_half():
    assert m.auroc_error_detection([0.7] * 4, [1, 0, 1, 0]) == 0.5


def test_auroc_mixed_ties_count_half():
    # scores .1,.5,.5,.9 ; incorrect idx1 (.5), idx3 (.9) ; correct idx0 (.1), idx2 (.5)
    # (.5>.1)=1, (.5 vs .5)=.5, (.9>.1)=1, (.9>.5)=1 -> 3.5/4
    assert m.auroc_error_detection([0.9, 0.5, 0.5, 0.1], [1, 0, 1, 0]) == pytest.approx(0.875)


def test_auroc_undefined_when_one_class_only():
    assert np.isnan(m.auroc_error_detection([0.9, 0.8], [1, 1]))
    assert np.isnan(m.auroc_error_detection([0.9, 0.8], [0, 0]))


def test_auroc_matches_sklearn_on_random_data_with_ties():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    conf = np.round(rng.uniform(0.3, 1.0, 400), 2)  # rounding creates many ties
    correct = rng.random(400) < conf
    expected = sk.roc_auc_score((~correct).astype(int), 1.0 - conf)
    assert m.auroc_error_detection(conf, correct) == pytest.approx(expected)
