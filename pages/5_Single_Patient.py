"""Single-patient prediction with a SHAP-style explanation."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.pipeline import Pipeline

import app_common as ac
import data_utils as du

st.set_page_config(page_title="Single Patient", page_icon="🧑‍⚕️", layout="wide")
ac.page_header("🧑‍⚕️ Single-Patient Prediction",
               "Score one encounter, compare the standard vs fairness-adjusted "
               "decision, and explain the drivers.")
ac.require_models()

meta = ac.load_meta()
metrics = ac.load_metrics()
fair = ac.load_fair()
eval_blob = ac.load_eval()


def explain_prediction(model_name, model, x_row, background):
    """Return (method, base_value, contributions[feature->value]). Falls back to
    global feature importance if SHAP is unavailable."""
    feats = list(x_row.columns)
    try:
        import shap
        est = model.named_steps["clf"] if isinstance(model, Pipeline) else model
        if isinstance(model, Pipeline):
            scaler = model.named_steps["scaler"]
            bg = scaler.transform(background)
            xin = scaler.transform(x_row)
            explainer = shap.LinearExplainer(est, bg)
            vals = np.asarray(explainer.shap_values(xin)).reshape(-1)
            base = float(np.ravel(explainer.expected_value)[0])
        else:
            explainer = shap.TreeExplainer(est)
            expl = explainer(x_row)
            v = np.asarray(expl.values)
            if v.ndim == 3:        # (1, n_features, n_classes)
                vals = v[0, :, -1]
                bv = np.ravel(expl.base_values)
                base = float(bv[-1] if bv.size > 1 else bv[0])
            else:                  # (1, n_features)
                vals = v[0]
                base = float(np.ravel(expl.base_values)[0])
        return "shap", base, dict(zip(feats, vals))
    except Exception:
        imp = eval_blob["importances"].get(model_name, {})
        return "importance", 0.0, {f: imp.get(f, 0.0) for f in feats}


# --------------------------------------------------------------------------- #
# Input form
# --------------------------------------------------------------------------- #
with st.form("patient"):
    top = st.columns(4)
    model_name = top[0].selectbox("Model", meta["models"],
                                  index=meta["models"].index(meta["best_model"]))
    race = top[1].selectbox("Race (for fairness adjustment)", du.RACE_LEVELS)
    gender = top[2].selectbox("Gender", du.GENDER_LEVELS)
    age = top[3].selectbox("Age band", du.AGE_LEVELS, index=6)

    st.markdown("**Utilisation & clinical counts**")
    s = st.columns(4)
    time_in_hospital = s[0].slider("Days in hospital", 1, 14, 4)
    number_inpatient = s[1].slider("Prior inpatient visits", 0, 21, 0)
    number_emergency = s[2].slider("Prior emergency visits", 0, 76, 0)
    number_outpatient = s[3].slider("Prior outpatient visits", 0, 42, 0)
    s2 = st.columns(4)
    num_medications = s2[0].slider("Medications", 1, 81, 15)
    num_lab_procedures = s2[1].slider("Lab procedures", 1, 132, 43)
    num_procedures = s2[2].slider("Procedures", 0, 6, 1)
    number_diagnoses = s2[3].slider("Diagnoses", 1, 16, 7)

    with st.expander("Medications & labs (optional)"):
        m = st.columns(4)
        A1Cresult = m[0].selectbox("A1C result", du.A1C_LEVELS)
        max_glu_serum = m[1].selectbox("Max glucose serum", du.GLU_LEVELS)
        insulin = m[2].selectbox("Insulin", du.MED_LEVELS, index=1)
        metformin = m[3].selectbox("Metformin", du.MED_LEVELS)
        m2 = st.columns(4)
        glipizide = m2[0].selectbox("Glipizide", du.MED_LEVELS)
        glyburide = m2[1].selectbox("Glyburide", du.MED_LEVELS)
        change = m2[2].selectbox("Medication change", du.CHANGE_LEVELS)
        diabetesMed = m2[3].selectbox("On diabetes meds", du.DIABETESMED_LEVELS, index=1)
        pioglitazone = "No"
        rosiglitazone = "No"

    submitted = st.form_submit_button("🔮 Predict readmission risk", type="primary",
                                      use_container_width=True)

if not submitted:
    st.info("Set the patient's attributes and click **Predict**.")
    st.stop()

# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
values = dict(
    gender=gender, age=age, time_in_hospital=time_in_hospital,
    num_lab_procedures=num_lab_procedures, num_procedures=num_procedures,
    num_medications=num_medications, number_outpatient=number_outpatient,
    number_emergency=number_emergency, number_inpatient=number_inpatient,
    number_diagnoses=number_diagnoses, A1Cresult=A1Cresult,
    max_glu_serum=max_glu_serum, metformin=metformin, insulin=insulin,
    glipizide=glipizide, glyburide=glyburide, pioglitazone=pioglitazone,
    rosiglitazone=rosiglitazone, change=change, diabetesMed=diabetesMed,
)
x_row = du.build_input_row(values)
model = ac.load_model(model_name)
prob = float(model.predict_proba(x_row)[:, 1][0])

std_thr = metrics[model_name]["threshold"]
is_best = model_name == fair["best_model"]
fair_thr = fair["thresholds_by_group"].get(race, fair["global_threshold"]) \
    if is_best else std_thr

std_pred = prob >= std_thr
fair_pred = prob >= fair_thr

r1, r2, r3 = st.columns(3)
r1.metric("Readmission probability", f"{prob:.1%}")
r2.metric("Standard decision", "READMIT RISK" if std_pred else "Low risk",
          help=f"threshold {std_thr:.3f}")
r3.metric("Fairness-adjusted decision", "READMIT RISK" if fair_pred else "Low risk",
          help=f"group threshold {fair_thr:.3f}"
               + ("" if is_best else " (fair thresholds defined for best model)"))

# gauge
fig = go.Figure(go.Indicator(
    mode="gauge+number", value=prob * 100,
    number={"suffix": "%"},
    gauge={"axis": {"range": [0, 100]},
           "bar": {"color": ac.TEAL},
           "steps": [{"range": [0, std_thr * 100], "color": "#e2e8f0"},
                     {"range": [std_thr * 100, 100], "color": "#fee2e2"}],
           "threshold": {"line": {"color": ac.RED, "width": 3},
                         "value": std_thr * 100}}))
st.plotly_chart(ac.style_fig(fig, height=280, title="30-day readmission risk"),
                use_container_width=True)

if is_best and std_pred != fair_pred:
    st.warning(f"The fairness-adjusted threshold for **{race}** changes this "
               f"patient's decision — illustrating how a single global cut-off "
               f"can systematically under- or over-flag specific groups.")

st.divider()

# --------------------------------------------------------------------------- #
# Explanation
# --------------------------------------------------------------------------- #
st.subheader("Prediction explanation")
method, base, contrib = explain_prediction(
    model_name, model, x_row, eval_blob["shap_background"])

ser = pd.Series(contrib)
top = ser.reindex(ser.abs().sort_values(ascending=False).index).head(10)[::-1]

if method == "shap":
    fig = go.Figure(go.Waterfall(
        orientation="h", y=top.index.tolist(),
        x=top.values.tolist(),
        connector={"line": {"color": ac.SLATE}},
        increasing={"marker": {"color": ac.RED}},
        decreasing={"marker": {"color": ac.TEAL}}))
    fig.update_layout(xaxis_title="contribution to log-odds of readmission")
    st.plotly_chart(ac.style_fig(fig, height=440,
                                 title="SHAP — top feature contributions"),
                    use_container_width=True)
    st.caption("Red pushes risk up, teal pulls it down. Values are SHAP "
               "contributions to the model's log-odds; the probability above "
               "comes directly from the model.")
else:
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                           marker_color=ac.TEAL))
    fig.update_layout(xaxis_title="global feature importance")
    st.plotly_chart(ac.style_fig(fig, height=440,
                                 title="Feature importance (SHAP unavailable)"),
                    use_container_width=True)
    st.caption("SHAP is not installed, so global feature importance is shown "
               "instead. `pip install shap` for per-prediction explanations.")

with st.expander("Encoded feature vector sent to the model"):
    st.dataframe(x_row.T.rename(columns={0: "value"}), use_container_width=True)
