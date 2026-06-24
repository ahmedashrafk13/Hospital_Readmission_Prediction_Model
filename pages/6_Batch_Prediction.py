"""Batch scoring of an uploaded CSV."""
import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import app_common as ac
import data_utils as du
import fairness_utils as fu

st.set_page_config(page_title="Batch Prediction", page_icon="📦", layout="wide")
ac.page_header("📦 Batch Prediction",
               "Upload a CSV of encounters, score them, and inspect the fairness "
               "breakdown of the batch.")
ac.require_models()

meta = ac.load_meta()
metrics = ac.load_metrics()
fair = ac.load_fair()


def encode_frame(frame: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=frame.index)
    for col in du.NUMERIC_COLS:
        X[col] = pd.to_numeric(frame.get(col, 0), errors="coerce").fillna(0).astype(float)
    for col in du.CATEGORICAL_COLS:
        if col in frame.columns:
            X[col] = frame[col].map(du.ENCODERS[col]).fillna(0).astype(int)
        else:
            X[col] = 0
    return X[du.FEATURE_COLS]


# --------------------------------------------------------------------------- #
# Controls + template
# --------------------------------------------------------------------------- #
c1, c2 = st.columns([2, 1])
with c1:
    model_name = st.selectbox("Model", meta["models"],
                              index=meta["models"].index(meta["best_model"]))
    uploaded = st.file_uploader("Upload patient CSV (UCI schema columns)", type="csv")
with c2:
    sample = du.generate_synthetic_data(n=300, seed=7)
    st.download_button(
        "⬇️ Download sample template (300 rows)",
        sample.to_csv(index=False).encode(),
        file_name="readmission_template.csv", mime="text/csv",
        use_container_width=True)
    st.caption("Include a `race` column to get the fairness breakdown, and an "
               "optional `readmitted` (0/1) column to score recall.")

if uploaded is None:
    st.info("Upload a CSV or download the template above to try it out.")
    st.stop()

frame = pd.read_csv(uploaded)
st.success(f"Loaded {len(frame):,} rows · {frame.shape[1]} columns.")

# --------------------------------------------------------------------------- #
# Score
# --------------------------------------------------------------------------- #
X = encode_frame(frame)
model = ac.load_model(model_name)
prob = model.predict_proba(X)[:, 1]
std_thr = metrics[model_name]["threshold"]
is_best = model_name == fair["best_model"]

has_race = "race" in frame.columns
sens = frame["race"].astype(str).to_numpy() if has_race \
    else np.array(["Unknown"] * len(frame))

if is_best and has_race:
    fair_pred = fu.apply_group_thresholds(prob, sens, fair["thresholds_by_group"],
                                          fair["global_threshold"])
else:
    fair_pred = (prob >= std_thr).astype(int)

out = frame.copy()
out["readmit_probability"] = prob.round(4)
out["prediction"] = (prob >= std_thr).astype(int)
out["fair_prediction"] = fair_pred

# --------------------------------------------------------------------------- #
# Aggregate stats
# --------------------------------------------------------------------------- #
m1, m2, m3, m4 = st.columns(4)
m1.metric("Scored", f"{len(out):,}")
m2.metric("Flagged (standard)", f"{int(out['prediction'].sum()):,}",
          f"{out['prediction'].mean():.1%}")
m3.metric("Flagged (fair)", f"{int(out['fair_prediction'].sum()):,}",
          f"{out['fair_prediction'].mean():.1%}")
m4.metric("Mean probability", f"{prob.mean():.1%}")

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Risk-score distribution")
    fig = px.histogram(out, x="readmit_probability", nbins=40,
                       color_discrete_sequence=[ac.TEAL])
    fig.add_vline(x=std_thr, line_dash="dash", line_color=ac.RED,
                  annotation_text="threshold")
    fig.update_layout(xaxis_title="predicted probability", yaxis_title="patients")
    st.plotly_chart(ac.style_fig(fig, height=380), use_container_width=True)

with right:
    st.subheader("Flag rate by race")
    if has_race:
        br = out.groupby("race")[["prediction", "fair_prediction"]].mean().reset_index()
        fig = px.bar(br.melt(id_vars="race", var_name="policy", value_name="rate"),
                     x="race", y="rate", color="policy", barmode="group",
                     color_discrete_sequence=[ac.SLATE, ac.TEAL])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(ac.style_fig(fig, height=380), use_container_width=True)
    else:
        st.info("No `race` column — add one for the fairness breakdown.")

# Optional recall scoring if labels are present
if "readmitted" in frame.columns:
    y = pd.to_numeric(frame["readmitted"], errors="coerce").fillna(0).astype(int).to_numpy()
    st.subheader("Recall (TPR) by race — labels detected")
    gm_std = fu.group_metrics(y, out["prediction"].to_numpy(), sens)
    gm_fair = fu.group_metrics(y, out["fair_prediction"].to_numpy(), sens)
    rec = pd.DataFrame({"standard": gm_std["tpr"], "fair": gm_fair["tpr"]})
    st.dataframe(rec.style.format("{:.1%}"), use_container_width=True)
    _, gap_std = fu.equal_opportunity_gap(y, out["prediction"].to_numpy(), sens)
    _, gap_fair = fu.equal_opportunity_gap(y, out["fair_prediction"].to_numpy(), sens)
    gc1, gc2 = st.columns(2)
    gc1.metric("EO gap (standard)", f"{gap_std * 100:.1f} pp")
    gc2.metric("EO gap (fair)", f"{gap_fair * 100:.1f} pp")

st.divider()
st.subheader("Results")
st.dataframe(out.head(200), use_container_width=True, height=360)
st.download_button(
    "⬇️ Download scored CSV", out.to_csv(index=False).encode(),
    file_name="readmission_predictions.csv", mime="text/csv",
    type="primary")
