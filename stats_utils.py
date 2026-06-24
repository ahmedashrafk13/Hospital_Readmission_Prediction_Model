"""
Statistical-significance tests for model comparison.

* ``delong_auc_test`` -- DeLong's test for two correlated ROC AUCs (paired,
  same test set). Uses the fast midrank algorithm of Sun & Xu (2014).
* ``kruskal_wallis_test`` -- non-parametric Kruskal-Wallis H-test across an
  arbitrary number of score distributions.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Fast DeLong machinery
# --------------------------------------------------------------------------- #
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midrank = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midrank[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = midrank
    return out


def _fast_delong(predictions_sorted: np.ndarray, n_pos: int):
    """Vectorised DeLong.

    ``predictions_sorted`` is shape ``(k, n)`` with the ``n_pos`` positive
    examples in the leading columns. Returns ``(aucs, covariance)``.
    """
    m = n_pos
    n = predictions_sorted.shape[1] - m
    pos = predictions_sorted[:, :m]
    neg = predictions_sorted[:, m:]
    k = predictions_sorted.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(predictions_sorted[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, cov


def delong_auc_test(y_true, prob_a, prob_b) -> dict:
    """Compare two AUCs on the same labels.

    Returns ``{auc_a, auc_b, z, p}``. ``p`` is two-sided.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_a = np.asarray(prob_a, dtype=float)
    prob_b = np.asarray(prob_b, dtype=float)

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return {"auc_a": float("nan"), "auc_b": float("nan"),
                "z": float("nan"), "p": float("nan")}

    order = np.argsort(-y_true, kind="mergesort")  # positives first, stable
    preds = np.vstack((prob_a, prob_b))[:, order]
    aucs, cov = _fast_delong(preds, n_pos)

    contrast = np.array([[1.0, -1.0]])
    var = float((contrast @ np.asarray(cov).reshape(2, 2) @ contrast.T).item())
    if var <= 0:
        z = 0.0
        p = 1.0
    else:
        z = float((aucs[0] - aucs[1]) / np.sqrt(var))
        p = float(2.0 * stats.norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": z, "p": p}


def delong_pvalue_matrix(y_true, prob_dict: dict) -> tuple[list[str], np.ndarray]:
    """Pairwise DeLong p-value matrix across a dict of ``{name: probs}``.

    Returns ``(names, matrix)`` where ``matrix[i, j]`` is the two-sided p-value
    comparing model ``i`` and model ``j`` (diagonal = 1.0).
    """
    names = list(prob_dict)
    k = len(names)
    mat = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            res = delong_auc_test(y_true, prob_dict[names[i]], prob_dict[names[j]])
            mat[i, j] = mat[j, i] = res["p"]
    return names, mat


# --------------------------------------------------------------------------- #
# Kruskal-Wallis
# --------------------------------------------------------------------------- #
def kruskal_wallis_test(groups_dict: dict) -> dict:
    """Kruskal-Wallis H-test across distributions in ``{label: values}``.

    Returns ``{h, p, dof, n_groups}``.
    """
    samples = [np.asarray(v, dtype=float) for v in groups_dict.values()
               if len(v) > 0]
    if len(samples) < 2:
        return {"h": float("nan"), "p": float("nan"), "dof": 0,
                "n_groups": len(samples)}
    h, p = stats.kruskal(*samples)
    return {"h": float(h), "p": float(p), "dof": len(samples) - 1,
            "n_groups": len(samples)}
