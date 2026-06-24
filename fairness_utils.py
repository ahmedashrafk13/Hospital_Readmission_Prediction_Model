"""
Group-fairness metrics and post-processing threshold adjustment.

Equal Opportunity = equal True Positive Rate across sensitive groups. Because
ROC AUC is threshold-independent, per-group threshold tuning equalises TPR at
*zero AUC cost* -- the model's ranking never changes, only the operating point
per group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Minimum positives in a group before we trust a per-group threshold estimate.
MIN_POSITIVES = 25


def _as_arrays(y_true, sensitive):
    y_true = np.asarray(y_true).astype(int)
    sensitive = np.asarray(sensitive).astype(str)
    return y_true, sensitive


def confusion_counts(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def group_metrics(y_true, y_pred, sensitive) -> pd.DataFrame:
    """Per-group fairness metrics as a DataFrame indexed by group."""
    y_true, sensitive = _as_arrays(y_true, sensitive)
    y_pred = np.asarray(y_pred).astype(int)
    rows = []
    for g in sorted(set(sensitive)):
        mask = sensitive == g
        c = confusion_counts(y_true[mask], y_pred[mask])
        tpr = _safe_div(c["tp"], c["tp"] + c["fn"])          # recall / EO
        fpr = _safe_div(c["fp"], c["fp"] + c["tn"])
        ppv = _safe_div(c["tp"], c["tp"] + c["fp"])          # precision
        sel = _safe_div(c["tp"] + c["fp"], mask.sum())       # selection rate
        rows.append({
            "group": g,
            "n": int(mask.sum()),
            "n_positive": int(c["tp"] + c["fn"]),
            "tpr": tpr,
            "fpr": fpr,
            "precision": ppv,
            "selection_rate": sel,
        })
    return pd.DataFrame(rows).set_index("group")


def equal_opportunity_gap(y_true, y_pred, sensitive) -> tuple[dict, float]:
    """Return ``(per_group_tpr, max_gap)`` where gap = max TPR - min TPR.

    Groups with too few positives are excluded from the gap calculation.
    """
    gm = group_metrics(y_true, y_pred, sensitive)
    eligible = gm[gm["n_positive"] >= MIN_POSITIVES]
    per_group = gm["tpr"].to_dict()
    if eligible.empty:
        return per_group, float("nan")
    gap = float(eligible["tpr"].max() - eligible["tpr"].min())
    return per_group, gap


def threshold_for_tpr(scores, y_true, target_tpr: float) -> float:
    """Smallest threshold whose TPR for this group is ~``target_tpr``."""
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    pos = scores[y_true == 1]
    if len(pos) == 0:
        return 0.5
    q = float(np.clip(1.0 - target_tpr, 0.0, 1.0))
    return float(np.quantile(pos, q))


def adjust_thresholds(scores, y_true, sensitive, global_threshold: float,
                      target_tpr: float | None = None) -> tuple[dict, float]:
    """Per-group thresholds that equalise TPR.

    Returns ``(thresholds_by_group, target_tpr)``. Groups with too few
    positives keep the global threshold.
    """
    y_true, sensitive = _as_arrays(y_true, sensitive)
    scores = np.asarray(scores, dtype=float)

    base_pred = (scores >= global_threshold).astype(int)
    base = group_metrics(y_true, base_pred, sensitive)
    eligible = base[base["n_positive"] >= MIN_POSITIVES]

    if target_tpr is None:
        # Aim for the median eligible-group TPR -- a feasible common target.
        target_tpr = float(eligible["tpr"].median()) if not eligible.empty else 0.5

    thresholds = {}
    for g in base.index:
        mask = sensitive == g
        if base.loc[g, "n_positive"] >= MIN_POSITIVES:
            thresholds[g] = threshold_for_tpr(scores[mask], y_true[mask], target_tpr)
        else:
            thresholds[g] = float(global_threshold)
    return thresholds, float(target_tpr)


def apply_group_thresholds(scores, sensitive, thresholds: dict,
                           default: float) -> np.ndarray:
    """Vectorised per-group thresholding -> binary predictions."""
    scores = np.asarray(scores, dtype=float)
    sensitive = np.asarray(sensitive).astype(str)
    thr = np.array([thresholds.get(g, default) for g in sensitive], dtype=float)
    return (scores >= thr).astype(int)


def fairness_summary(y_true, y_pred, sensitive) -> dict:
    """Aggregate fairness gaps for the dashboard."""
    gm = group_metrics(y_true, y_pred, sensitive)
    eligible = gm[gm["n_positive"] >= MIN_POSITIVES]
    if eligible.empty:
        eligible = gm

    def _gap(col: str) -> float:
        vals = eligible[col].dropna()
        return float(vals.max() - vals.min()) if len(vals) else float("nan")

    return {
        "equal_opportunity_gap": _gap("tpr"),
        "demographic_parity_gap": _gap("selection_rate"),
        "predictive_parity_gap": _gap("precision"),
        "equalized_odds_gap": float(max(_gap("tpr"), _gap("fpr"))),
        "table": gm,
    }
