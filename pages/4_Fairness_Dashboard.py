"""Fairness dashboard: equal-opportunity audit and post-processing correction."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import app_common as ac
import fairness_utils as fu

st.set_page_config(page_title="Fairness Dashboard", page_icon="⚖️", layout="wide")
ac.page_header("⚖️ Fairness Dashboard",
               "Equal-Opportunity audit across racial groups and a zero-AUC-cost "
               "threshold correction.")
ac.require_models()

eval_blob = ac.load_eval()
fair = ac.load_fair()
meta = ac.load_meta()

best = fair["best_model"]
y = np.asarray(eval_blob["y_test"])
sens = np.asarray(eval_blob["sensitive_test"])
prob = eval_blob["probs"][best]
g_thr = fair["global_threshold"]
thresholds = fair["thresholds_by_group"]

pred_before = (prob >= g_thr).astype(int)
pred_after = fu.apply_group_thresholds(prob, sens, thresholds, g_thr)

# --------------------------------------------------------------------------- #
# Headline
# --------------------------------------------------------------------------- #
st.markdown(f"Sensitive attribute: **race** · model: **{best}** · "
            f"equal opportunity = equal **true-positive rate** (recall) "
            "across groups.")
h1, h2, h3 = st.columns(3)
h1.metric("EO gap — single global threshold", f"{fair['eo_gap_before'] * 100:.1f} pp")
h2.metric("EO gap — per-group thresholds", f"{fair['eo_gap_after'] * 100:.1f} pp",
          delta=f"{(fair['eo_gap_after'] - fair['eo_gap_before']) * 100:.1f} pp",
          delta_color="inverse")
h3.metric("AUC cost", f"{(fair['auc_after'] - fair['auc_before']) * 100:.2f} pp",
          help="Per-group thresholds change only the operating point, not the "
               "model's ranking — so AUC is unchanged.")

st.divider()

# --------------------------------------------------------------------------- #
# EO gap bar chart (TPR by race, before vs after)
# --------------------------------------------------------------------------- #
st.subheader("True-positive rate by race")

gm_before = fu.group_metrics(y, pred_before, sens)
gm_after = fu.group_metrics(y, pred_after, sens)
order = gm_before.sort_values("tpr").index.tolist()

fig = go.Figure()
fig.add_trace(go.Bar(name="Before (global threshold)", x=order,
                     y=[gm_before.loc[g, "tpr"] for g in order],
                     marker_color=ac.SLATE,
                     text=[f"{gm_before.loc[g, 'tpr']:.1%}" for g in order],
                     textposition="outside"))
fig.add_trace(go.Bar(name="After (per-group thresholds)", x=order,
                     y=[gm_after.loc[g, "tpr"] for g in order],
                     marker_color=ac.TEAL,
                     text=[f"{gm_after.loc[g, 'tpr']:.1%}" for g in order],
                     textposition="outside"))
fig.add_hline(y=fair["target_tpr"], line_dash="dash", line_color=ac.AMBER,
              annotation_text=f"target TPR {fair['target_tpr']:.1%}")
fig.update_yaxes(tickformat=".0%", title="true-positive rate (recall)")
fig.update_layout(barmode="group", legend=dict(x=0.02, y=1.12, orientation="h"))
st.plotly_chart(ac.style_fig(fig, height=460), use_container_width=True)
st.caption("Before correction, recall on true readmissions varies sharply by race. "
           "After per-group thresholding, every eligible group's recall converges "
           "on the common target.")

st.divider()

# --------------------------------------------------------------------------- #
# Broader fairness metrics
# --------------------------------------------------------------------------- #
st.subheader("Group-fairness metrics — before vs after")
sum_before = fu.fairness_summary(y, pred_before, sens)
sum_after = fu.fairness_summary(y, pred_after, sens)

comp = pd.DataFrame({
    "Before": [sum_before["equal_opportunity_gap"], sum_before["equalized_odds_gap"],
               sum_before["demographic_parity_gap"], sum_before["predictive_parity_gap"]],
    "After": [sum_after["equal_opportunity_gap"], sum_after["equalized_odds_gap"],
              sum_after["demographic_parity_gap"], sum_after["predictive_parity_gap"]],
}, index=["Equal opportunity (TPR gap)", "Equalized odds (max TPR/FPR gap)",
          "Demographic parity (selection-rate gap)", "Predictive parity (precision gap)"])

mleft, mright = st.columns([3, 2])
with mleft:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before", y=comp.index, x=comp["Before"],
                         orientation="h", marker_color=ac.SLATE))
    fig.add_trace(go.Bar(name="After", y=comp.index, x=comp["After"],
                         orientation="h", marker_color=ac.TEAL))
    fig.update_xaxes(tickformat=".0%", title="disparity (max − min across groups)")
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(ac.style_fig(fig, height=380), use_container_width=True)
with mright:
    st.dataframe(comp.style.format("{:.1%}")
                 .background_gradient(cmap="Reds", axis=None),
                 use_container_width=True, height=360)
    st.caption("Equal-opportunity is the optimisation target; the other metrics "
               "are reported for transparency (improving one fairness criterion "
               "need not improve all — an inherent trade-off).")

st.divider()

# --------------------------------------------------------------------------- #
# Per-group thresholds + confusion matrices
# --------------------------------------------------------------------------- #
st.subheader("Per-group decision thresholds")
thr_df = pd.DataFrame({
    "group": list(thresholds),
    "adjusted_threshold": [thresholds[g] for g in thresholds],
    "global_threshold": g_thr,
    "n_positive": [int(gm_before.loc[g, "n_positive"]) if g in gm_before.index else 0
                   for g in thresholds],
}).set_index("group")
st.dataframe(thr_df.style.format({"adjusted_threshold": "{:.3f}",
                                  "global_threshold": "{:.3f}",
                                  "n_positive": "{:,}"}),
             use_container_width=True)
st.caption("Groups with fewer than the minimum positive count retain the global "
           "threshold (their per-group estimate would be unreliable).")

st.subheader("Per-group confusion matrices")
view = st.radio("Thresholding", ["Before (global)", "After (per-group)"],
                horizontal=True)
pred = pred_before if view.startswith("Before") else pred_after
groups = order
cols = st.columns(len(groups))
for col, g in zip(cols, groups):
    mask = sens == g
    c = fu.confusion_counts(y[mask], pred[mask])
    cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Teal",
                    x=["P:No", "P:Yes"], y=["T:No", "T:Yes"])
    fig.update_coloraxes(showscale=False)
    tpr = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else float("nan")
    col.plotly_chart(ac.style_fig(fig, height=240, title=f"{g} (TPR {tpr:.0%})"),
                     use_container_width=True)
