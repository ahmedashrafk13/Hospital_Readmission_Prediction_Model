"""Exploratory data analysis."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import app_common as ac
import data_utils as du

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
ac.page_header("📊 Exploratory Data Analysis",
               "Cohort composition, readmission patterns, and feature structure.")

df, is_real = ac.get_data()
rate_overall = df[du.TARGET_COL].mean()

# --------------------------------------------------------------------------- #
# Summary cards
# --------------------------------------------------------------------------- #
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Patients", f"{len(df):,}")
c2.metric("Readmitted < 30d", f"{df[du.TARGET_COL].sum():,}", f"{rate_overall:.1%}")
c3.metric("Avg. days in hospital", f"{df['time_in_hospital'].mean():.1f}")
c4.metric("Avg. medications", f"{df['num_medications'].mean():.1f}")
c5.metric("On diabetes meds", f"{(df['diabetesMed'] == 'Yes').mean():.0%}")

st.divider()


def rate_by(col: str, order=None) -> pd.DataFrame:
    g = df.groupby(col)[du.TARGET_COL].agg(["mean", "count"]).reset_index()
    g.columns = [col, "readmit_rate", "patients"]
    if order is not None:
        g[col] = pd.Categorical(g[col], categories=order, ordered=True)
        g = g.sort_values(col)
    else:
        g = g.sort_values("readmit_rate", ascending=False)
    return g


# --------------------------------------------------------------------------- #
# Readmission by demographic
# --------------------------------------------------------------------------- #
st.subheader("30-day readmission rate by demographic")
t1, t2, t3 = st.tabs(["By race", "By age", "By gender"])

with t1:
    g = rate_by("race")
    fig = px.bar(g, x="race", y="readmit_rate", text=g["readmit_rate"].map("{:.1%}".format),
                 color="race", color_discrete_sequence=ac.GROUP_COLORS,
                 hover_data={"patients": ":,"})
    fig.add_hline(y=rate_overall, line_dash="dash", line_color=ac.SLATE,
                  annotation_text=f"overall {rate_overall:.1%}")
    fig.update_yaxes(tickformat=".0%", title="readmission rate")
    fig.update_layout(showlegend=False)
    st.plotly_chart(ac.style_fig(fig, height=420), use_container_width=True)

with t2:
    g = rate_by("age", order=du.AGE_LEVELS)
    fig = px.bar(g, x="age", y="readmit_rate", text=g["readmit_rate"].map("{:.1%}".format),
                 color_discrete_sequence=[ac.TEAL], hover_data={"patients": ":,"})
    fig.add_hline(y=rate_overall, line_dash="dash", line_color=ac.SLATE)
    fig.update_yaxes(tickformat=".0%", title="readmission rate")
    st.plotly_chart(ac.style_fig(fig, height=420), use_container_width=True)

with t3:
    g = rate_by("gender")
    fig = px.bar(g, x="gender", y="readmit_rate", text=g["readmit_rate"].map("{:.1%}".format),
                 color="gender", color_discrete_sequence=[ac.NAVY, ac.TEAL],
                 hover_data={"patients": ":,"})
    fig.update_yaxes(tickformat=".0%", title="readmission rate")
    fig.update_layout(showlegend=False)
    st.plotly_chart(ac.style_fig(fig, height=420), use_container_width=True)

st.caption("Note the modest variation in **base readmission rate** across races — "
           "the fairness issue this project addresses is not unequal prevalence "
           "but unequal *detection* of true positives (see Fairness Dashboard).")

st.divider()

# --------------------------------------------------------------------------- #
# Correlation heatmap + cohort composition
# --------------------------------------------------------------------------- #
left, right = st.columns([3, 2])

with left:
    st.subheader("Numeric-feature correlations")
    corr = df[du.NUMERIC_COLS].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="Teal", zmin=-1, zmax=1)
    st.plotly_chart(ac.style_fig(fig, height=520), use_container_width=True)

with right:
    st.subheader("Cohort composition")
    comp = df["race"].value_counts().reset_index()
    comp.columns = ["race", "patients"]
    fig = px.pie(comp, names="race", values="patients", hole=0.5,
                 color_discrete_sequence=ac.GROUP_COLORS)
    st.plotly_chart(ac.style_fig(fig, height=300), use_container_width=True)

    st.markdown("**Numeric feature summary**")
    st.dataframe(
        df[du.NUMERIC_COLS].describe().T[["mean", "std", "min", "max"]]
        .style.format("{:.1f}"),
        use_container_width=True, height=320,
    )

st.divider()

# --------------------------------------------------------------------------- #
# Feature importance (best model)
# --------------------------------------------------------------------------- #
st.subheader("Top predictors (best model)")
if ac.models_exist():
    eval_blob = ac.load_eval()
    best = ac.load_meta()["best_model"]
    imp = pd.Series(eval_blob["importances"][best]).sort_values(ascending=True).tail(12)
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                           marker_color=ac.TEAL))
    fig.update_layout(xaxis_title="relative importance")
    st.plotly_chart(ac.style_fig(fig, height=460,
                                 title=f"{best} — feature importance"),
                    use_container_width=True)
else:
    st.info("Train models on the Home page to see feature importance.")
