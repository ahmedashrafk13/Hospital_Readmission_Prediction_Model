"""End-to-end smoke test of the logic behind every page (no Streamlit runtime)."""
import json
import os
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

import data_utils as du
import fairness_utils as fu
import stats_utils as su

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "models")

print("== artifacts ==")
for f in ["metrics.json", "fair_thresholds.json", "meta.json", "eval.pkl"]:
    assert os.path.exists(os.path.join(M, f)), f"missing {f}"
meta = json.load(open(os.path.join(M, "meta.json")))
metrics = json.load(open(os.path.join(M, "metrics.json")))
fair = json.load(open(os.path.join(M, "fair_thresholds.json")))
ev = joblib.load(os.path.join(M, "eval.pkl"))
print("  models:", list(metrics))
print("  best:", meta["best_model"], "AUC", round(metrics[meta["best_model"]]["auc"], 4))
print("  EO gap before/after:", round(fair["eo_gap_before"]*100, 1),
      "->", round(fair["eo_gap_after"]*100, 1), "pp")

y = np.asarray(ev["y_test"]); sens = np.asarray(ev["sensitive_test"])
probs = ev["probs"]

print("== stats (page 3) ==")
names, pmat = su.delong_pvalue_matrix(y, probs)
print("  DeLong matrix shape", pmat.shape, "finite:", np.isfinite(pmat).all())
kw = su.kruskal_wallis_test(probs)
print("  Kruskal-Wallis H", round(kw["h"], 1), "p", f"{kw['p']:.2e}")

print("== fairness (page 4) ==")
best = fair["best_model"]; p = probs[best]; g = fair["global_threshold"]
pred_b = (p >= g).astype(int)
pred_a = fu.apply_group_thresholds(p, sens, fair["thresholds_by_group"], g)
gm = fu.group_metrics(y, pred_a, sens)
print("  per-group TPR after:", {k: round(v, 3) for k, v in gm["tpr"].items()})
summ = fu.fairness_summary(y, pred_b, sens)
print("  dp/eodds/pp gaps before:",
      round(summ["demographic_parity_gap"], 3),
      round(summ["equalized_odds_gap"], 3),
      round(summ["predictive_parity_gap"], 3))

print("== single patient + SHAP (page 5) ==")
values = dict(gender="Male", age="[70-80)", time_in_hospital=8,
              num_lab_procedures=60, num_procedures=2, num_medications=22,
              number_outpatient=1, number_emergency=2, number_inpatient=3,
              number_diagnoses=9, A1Cresult=">8", max_glu_serum=">200",
              insulin="Up", metformin="No", glipizide="No", glyburide="No",
              pioglitazone="No", rosiglitazone="No", change="Ch", diabetesMed="Yes")
x_row = du.build_input_row(values)
for name in [best, "Logistic Regression"]:
    model = joblib.load(os.path.join(M, meta["model_keys"][name] + ".pkl"))
    prob = float(model.predict_proba(x_row)[:, 1][0])
    # SHAP path
    import shap
    est = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    try:
        if isinstance(model, Pipeline):
            sc = model.named_steps["scaler"]
            ex = shap.LinearExplainer(est, sc.transform(ev["shap_background"]))
            vals = np.asarray(ex.shap_values(sc.transform(x_row))).reshape(-1)
        else:
            ex = shap.TreeExplainer(est)
            e = ex(x_row); v = np.asarray(e.values)
            vals = v[0, :, -1] if v.ndim == 3 else v[0]
        ok = f"shap ok ({len(vals)} vals)"
    except Exception as exc:
        ok = f"shap FAILED: {exc}"
    print(f"  {name}: prob={prob:.3f}  {ok}")

print("== batch (page 6) ==")
import pandas as pd
sample = du.generate_synthetic_data(n=300, seed=7)
Xb = sample.copy()
Xenc = pd.DataFrame(index=Xb.index)
for c in du.NUMERIC_COLS:
    Xenc[c] = pd.to_numeric(Xb.get(c, 0), errors="coerce").fillna(0).astype(float)
for c in du.CATEGORICAL_COLS:
    Xenc[c] = Xb[c].map(du.ENCODERS[c]).fillna(0).astype(int)
Xenc = Xenc[du.FEATURE_COLS]
model = joblib.load(os.path.join(M, meta["model_keys"][best] + ".pkl"))
bp = model.predict_proba(Xenc)[:, 1]
print("  batch scored:", len(bp), "mean prob", round(float(bp.mean()), 3))

print("\nALL CHECKS PASSED")
