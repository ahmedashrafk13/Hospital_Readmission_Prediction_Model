"""Statistical comparison of models: DeLong AUC test + Kruskal-Wallis."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import app_common as ac
import stats_utils as su

st.set_page_config(page_title="Statistical Analysis", page_icon="🔬", layout="wide")
ac.page_header("🔬 Statistical Analysis",
               "Are the models' performances *significantly* different? "
               "DeLong for AUC, Kruskal–Wallis for score distributions.")
ac.require_models()

eval_blob = ac.load_eval()
metrics = ac.load_metrics()
y = np.asarray(eval_blob["y_test"])
probs = eval_blob["probs"]
ALPHA = 0.05

# --------------------------------------------------------------------------- #
# DeLong pairwise AUC test
# --------------------------------------------------------------------------- #
st.subheader("DeLong test — pairwise AUC comparison")
st.caption("Two-sided p-values for the null hypothesis that two models have equal "
           "AUC on the same test set (correlated ROC curves). Cells with "
           f"p < {ALPHA} indicate a statistically significant difference.")

names, pmat = su.delong_pvalue_matrix(y, probs)
pdf = pd.DataFrame(pmat, index=names, columns=names)

fig = px.imshow(pdf, text_auto=".3f", color_continuous_scale="RdYlGn",
                zmin=0, zmax=0.2, aspect="auto")
fig.update_coloraxes(colorbar_title="p-value")
st.plotly_chart(ac.style_fig(fig, height=460,
                             title="Pairwise DeLong p-values"),
                use_container_width=True)

sig_pairs = [(names[i], names[j], pmat[i, j])
             for i in range(len(names)) for j in range(i + 1, len(names))
             if pmat[i, j] < ALPHA]
if sig_pairs:
    st.markdown("**Significant AUC differences (p < 0.05):**")
    st.dataframe(
        pd.DataFrame(sig_pairs, columns=["Model A", "Model B", "p-value"])
        .sort_values("p-value").style.format({"p-value": "{:.4g}"}),
        use_container_width=True, hide_index=True,
    )
else:
    st.info("No pairwise AUC difference reaches significance at α = 0.05 — the "
            "models are statistically comparable on discrimination.")

st.divider()

# --------------------------------------------------------------------------- #
# Kruskal-Wallis
# --------------------------------------------------------------------------- #
st.subheader("Kruskal–Wallis test — predicted-score distributions")
st.caption("Non-parametric test of whether the five models' predicted-probability "
           "distributions share the same location. A significant result means at "
           "least one model scores patients systematically differently.")

kw_all = su.kruskal_wallis_test(probs)
k1, k2, k3 = st.columns(3)
k1.metric("H statistic", f"{kw_all['h']:.1f}")
k2.metric("Degrees of freedom", kw_all["dof"])
k3.metric("p-value", f"{kw_all['p']:.2e}")

st.markdown("**Within the readmitted (positive) cohort only** — do models agree on "
            "how they score true positives?")
pos_scores = {name: p[y == 1] for name, p in probs.items()}
kw_pos = su.kruskal_wallis_test(pos_scores)
st.write(f"H = {kw_pos['h']:.1f}, p = {kw_pos['p']:.2e} "
         f"→ {'significant' if kw_pos['p'] < ALPHA else 'not significant'} "
         f"at α = {ALPHA}.")

# Score distribution visual
long = []
for name, p in probs.items():
    long.append(pd.DataFrame({"model": name, "score": p}))
long = pd.concat(long, ignore_index=True)
fig = px.violin(long, x="model", y="score", color="model", box=True, points=False,
                color_discrete_map=ac.MODEL_COLORS)
fig.update_layout(showlegend=False, yaxis_title="predicted readmission probability")
st.plotly_chart(ac.style_fig(fig, height=440,
                             title="Predicted-score distribution by model"),
                use_container_width=True)

st.divider()
with st.expander("How these tests work"):
    st.markdown(
        "- **DeLong (1988), fast algorithm (Sun & Xu, 2014):** compares two "
        "*correlated* AUCs evaluated on the same samples by estimating the "
        "covariance of the AUC estimators from per-sample placement values, "
        "yielding a z-statistic and p-value.\n"
        "- **Kruskal–Wallis H-test:** the non-parametric analogue of one-way "
        "ANOVA; ranks all observations jointly and tests whether group rank-sums "
        "differ more than expected by chance. No normality assumption.")
