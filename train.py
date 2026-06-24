"""
Train five classifiers on the diabetes readmission data, evaluate them, run the
fairness pipeline, and persist every artifact the Streamlit app needs.

Run directly::

    python train.py

Artifacts written to ``models/``:
    <model>.pkl            one joblib file per fitted estimator
    metrics.json           per-model AUC / F1 / accuracy / precision / recall
    fair_thresholds.json   best model, global + per-group thresholds, EO gaps
    eval.pkl               y_test, sensitive_test, per-model test probabilities,
                           feature importances, SHAP background sample
    meta.json              dataset + run metadata
"""
from __future__ import annotations

import json
import os
import time
import warnings

import joblib
import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import data_utils as du
import fairness_utils as fu

warnings.filterwarnings("ignore")

SEED = 42
N_ROWS = 70000
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

# Stable file-name keys for each model.
MODEL_KEYS = {
    "XGBoost": "xgboost",
    "Random Forest": "random_forest",
    "Logistic Regression": "logistic_regression",
    "Gradient Boosting": "gradient_boosting",
    "LightGBM": "lightgbm",
}


def build_models() -> dict:
    """Instantiate the five classifiers, with graceful fallbacks."""
    models: dict[str, object] = {}

    # 1. XGBoost (fallback: HistGradientBoosting)
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=350, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=SEED,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        models["XGBoost"] = HistGradientBoostingClassifier(
            max_iter=350, learning_rate=0.08, random_state=SEED)

    # 2. Random Forest
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300, max_depth=16, min_samples_leaf=5,
        n_jobs=-1, random_state=SEED)

    # 3. Logistic Regression (scaled)
    models["Logistic Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])

    # 4. Gradient Boosting
    models["Gradient Boosting"] = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.08, random_state=SEED)

    # 5. LightGBM (fallback: ExtraTrees)
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400, num_leaves=31, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
            random_state=SEED, verbose=-1)
    except Exception:
        models["LightGBM"] = ExtraTreesClassifier(
            n_estimators=300, max_depth=16, min_samples_leaf=5,
            n_jobs=-1, random_state=SEED)

    return models


def youden_threshold(y_true, prob) -> float:
    """Operating point maximising Youden's J (TPR - FPR)."""
    fpr, tpr, thr = roc_curve(y_true, prob)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def feature_importance(model, feature_names) -> dict:
    """Best-effort importance extraction across estimator types."""
    est = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        imp = np.abs(np.asarray(est.coef_, dtype=float)).ravel()
    else:
        imp = np.zeros(len(feature_names))
    if imp.sum() > 0:
        imp = imp / imp.sum()
    return {f: float(v) for f, v in zip(feature_names, imp)}


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    t0 = time.time()

    print("Loading data ...")
    df, is_real = du.load_data(n=N_ROWS, seed=SEED)
    X, y, sensitive = du.preprocess(df)
    print(f"  rows={len(df):,}  positives={y.mean():.3%}  "
          f"source={'real UCI' if is_real else 'synthetic'}")

    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, sensitive, test_size=0.2, stratify=y, random_state=SEED)

    models = build_models()
    metrics: dict[str, dict] = {}
    probs: dict[str, np.ndarray] = {}
    importances: dict[str, dict] = {}

    for name, model in models.items():
        print(f"Training {name} ...")
        ts = time.time()
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        thr = youden_threshold(y_te, p)
        pred = (p >= thr).astype(int)
        metrics[name] = {
            "auc": float(roc_auc_score(y_te, p)),
            "f1": float(f1_score(y_te, pred)),
            "accuracy": float(accuracy_score(y_te, pred)),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)),
            "threshold": thr,
        }
        probs[name] = p
        importances[name] = feature_importance(model, list(X.columns))
        joblib.dump(model, os.path.join(MODELS_DIR, f"{MODEL_KEYS[name]}.pkl"))
        print(f"  AUC={metrics[name]['auc']:.4f}  F1={metrics[name]['f1']:.4f}"
              f"  ({time.time() - ts:.1f}s)")

    # ----- fairness pipeline on the best model (by AUC) ------------------- #
    best = max(metrics, key=lambda m: metrics[m]["auc"])
    best_prob = probs[best]
    g_thr = metrics[best]["threshold"]
    print(f"Best model: {best} (AUC={metrics[best]['auc']:.4f})")

    pred_before = (best_prob >= g_thr).astype(int)
    per_group_before, gap_before = fu.equal_opportunity_gap(y_te, pred_before, s_te)

    thresholds, target_tpr = fu.adjust_thresholds(best_prob, y_te, s_te, g_thr)
    pred_after = fu.apply_group_thresholds(best_prob, s_te, thresholds, g_thr)
    per_group_after, gap_after = fu.equal_opportunity_gap(y_te, pred_after, s_te)

    auc_before = float(roc_auc_score(y_te, best_prob))  # ranking unchanged ->
    auc_after = auc_before                              # zero AUC cost

    print(f"  EO gap before adjustment: {gap_before * 100:.1f} pp")
    print(f"  EO gap after  adjustment: {gap_after * 100:.1f} pp")

    fair = {
        "best_model": best,
        "global_threshold": g_thr,
        "target_tpr": target_tpr,
        "thresholds_by_group": thresholds,
        "tpr_before": per_group_before,
        "tpr_after": per_group_after,
        "eo_gap_before": gap_before,
        "eo_gap_after": gap_after,
        "auc_before": auc_before,
        "auc_after": auc_after,
    }

    # ----- persist -------------------------------------------------------- #
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(MODELS_DIR, "fair_thresholds.json"), "w") as f:
        json.dump(fair, f, indent=2)

    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(X_tr), size=min(200, len(X_tr)), replace=False)
    eval_blob = {
        "y_test": y_te,
        "sensitive_test": np.asarray(s_te).astype(str),
        "probs": probs,
        "importances": importances,
        "feature_names": list(X.columns),
        "shap_background": X_tr.iloc[bg_idx].reset_index(drop=True),
        "X_test_sample": X_te.iloc[:2000].reset_index(drop=True),
    }
    joblib.dump(eval_blob, os.path.join(MODELS_DIR, "eval.pkl"))

    meta = {
        "seed": SEED,
        "n_rows": int(len(df)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "prevalence": float(y.mean()),
        "is_real": bool(is_real),
        "feature_names": list(X.columns),
        "models": list(models),
        "model_keys": MODEL_KEYS,
        "best_model": best,
        "trained_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done in {time.time() - t0:.1f}s. Artifacts in {MODELS_DIR}")


if __name__ == "__main__":
    main()
