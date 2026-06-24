"""Model performance: ROC, PR, confusion matrices, metrics."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    auc as sk_auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

import app_common as ac

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
ac.page_header("📈 Model Performance",
               "Discrimination and calibration of the five classifiers on the "
               "held-out test set.")
ac.require_models()

eval_blob = ac.load_eval()
metrics = ac.load_metrics()
y = np.asarray(eval_blob["y_test"])
probs = eval_blob["probs"]

# --------------------------------------------------------------------------- #
# Metrics table
# --------------------------------------------------------------------------- #
table = (pd.DataFrame(metrics).T[["auc", "f1", "accuracy", "precision", "recall"]]
         .sort_values("auc", ascending=False).rename(columns=str.upper))
st.dataframe(
    table.style.format("{:.4f}").background_gradient(cmap="Greens", subset=["AUC"]),
    use_container_width=True,
)

st.divider()
left, right = st.columns(2)

# --------------------------------------------------------------------------- #
# ROC curves
# --------------------------------------------------------------------------- #
with left:
    st.subheader("ROC curves")
    fig = go.Figure()
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} ({metrics[name]['auc']:.3f})",
            line=dict(color=ac.MODEL_COLORS.get(name), width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color=ac.SLATE, dash="dash"),
                             showlegend=False))
    fig.update_layout(xaxis_title="False positive rate",
                      yaxis_title="True positive rate",
                      legend=dict(x=0.4, y=0.06, font=dict(size=11)))
    st.plotly_chart(ac.style_fig(fig, height=460), use_container_width=True)

# --------------------------------------------------------------------------- #
# PR curves
# --------------------------------------------------------------------------- #
with right:
    st.subheader("Precision–Recall curves")
    fig = go.Figure()
    base = float(y.mean())
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ap = sk_auc(rec, prec)
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines", name=f"{name} ({ap:.3f})",
            line=dict(color=ac.MODEL_COLORS.get(name), width=2)))
    fig.add_hline(y=base, line_dash="dash", line_color=ac.SLATE,
                  annotation_text=f"baseline {base:.2f}")
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision",
                      legend=dict(x=0.4, y=0.95, font=dict(size=11)))
    st.plotly_chart(ac.style_fig(fig, height=460), use_container_width=True)

st.divider()

# --------------------------------------------------------------------------- #
# Confusion matrix per model
# --------------------------------------------------------------------------- #
st.subheader("Confusion matrix")
sel = st.selectbox("Model", list(probs), index=list(probs).index(ac.load_meta()["best_model"]))
thr = metrics[sel]["threshold"]
pred = (probs[sel] >= thr).astype(int)
cm = confusion_matrix(y, pred)

cc1, cc2 = st.columns([2, 1])
with cc1:
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Teal",
        x=["Pred: No", "Pred: <30d"], y=["True: No", "True: <30d"])
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(ac.style_fig(fig, height=380,
                                 title=f"{sel} @ threshold {thr:.3f}"),
                    use_container_width=True)
with cc2:
    tn, fp, fn, tp = cm.ravel()
    st.metric("True positives", f"{tp:,}")
    st.metric("False negatives (missed readmits)", f"{fn:,}")
    st.metric("False positives", f"{fp:,}")
    st.metric("True negatives", f"{tn:,}")
    st.caption(f"Recall {tp/(tp+fn):.1%} · Precision {tp/(tp+fp):.1%} "
               f"· Specificity {tn/(tn+fp):.1%}")
