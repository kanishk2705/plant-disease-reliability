"""Metric bundles and the source-only calibration step (temperature + abstention thresholds on val_cal)."""
from __future__ import annotations

import numpy as np

from . import metrics as M


def calibrate(val_cal_logits, val_cal_labels, r_cov: float = 0.90, r_risk: float = 0.01) -> dict:
    """Fit T and the two thresholds on val_cal ONLY."""
    T = M.fit_temperature(val_cal_logits, val_cal_labels)
    probs = M.softmax(val_cal_logits, T)
    conf, ok = M.confidence_and_correct(probs, val_cal_labels)
    return {"T": float(T),
            "val_cal_nll_before": M.nll_from_logits(val_cal_logits, val_cal_labels, 1.0),
            "val_cal_nll_after": M.nll_from_logits(val_cal_logits, val_cal_labels, T),
            "tau_r_cov": M.threshold_for_coverage(conf, r_cov),
            "tau_r_risk": M.threshold_for_risk(conf, ok, r_risk),
            "val_cal_accuracy": float(ok.mean()), "n_val_cal": int(len(ok))}


def bundle(logits, labels, T: float, classes: list, taus: dict) -> dict:
    """All point metrics for one evaluation set, before and after temperature scaling."""
    labels = np.asarray(labels)
    out = {"n": int(len(labels)), "T": float(T)}
    for name, temp in (("uncal", 1.0), ("cal", T)):
        p = M.softmax(logits, temp)
        conf, ok = M.confidence_and_correct(p, labels)
        pred = p.argmax(1)
        d = {"accuracy": M.accuracy(labels, pred), "macro_f1": M.macro_f1(labels, pred, list(range(len(classes)))),
             "nll": M.nll(p, labels), "brier_sum": M.brier_score(p, labels),
             "ece15_width": M.ece(p, labels, 15, "equal_width"), "ece15_mass": M.ece(p, labels, 15, "equal_mass"),
             "overconfidence": M.signed_overconfidence(p, labels), "mean_conf": float(conf.mean()),
             "aurc": M.aurc(conf, ok), "aurc_random": M.aurc_random(ok), "aurc_oracle": M.aurc_oracle(ok),
             "auroc_error": M.auroc_error_detection(conf, ok)}
        if name == "cal":
            sel = {}
            for rule, tau in taus.items():
                if tau is None:
                    sel[rule] = None
                    continue
                r = M.selective_metrics(conf, ok, tau)
                sel[rule] = {"tau": r.threshold, "coverage": r.coverage, "risk": r.risk,
                             "n_accepted": r.n_accepted, "n_rejected": r.n_rejected, "n_errors_accepted": r.n_errors_accepted}
            d["selective"] = sel
        out[name] = d
    return out
