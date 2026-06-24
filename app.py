"""
Hospital Readmission Prediction & Fairness Analysis -- Home.

Run with::

    streamlit run app.py
"""
import pandas as pd
import streamlit as st

import app_common as ac

st.set_page_config(
    page_title="Readmission Risk & Fairness",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

ac.page_header(
    "🏥 Hospital Readmission Prediction & Fairness Analysis",
    "30-day readmission risk for diabetic inpatients — five models, statistical "
    "model comparison, and an equal-opportunity fairness audit.",
)

st.markdown(
    f"""
    <div style="background:linear-gradient(90deg,{ac.NAVY},{ac.TEAL});
                padding:18px 22px;border-radius:12px;color:white;margin:8px 0 18px">
      <b>What this project does.</b> Trains XGBoost, Random Forest, Logistic
      Regression, Gradient Boosting and LightGBM on ~70k diabetic-patient
      encounters; compares them with <i>DeLong</i> and <i>Kruskal–Wallis</i>
      tests; then measures and corrects a racial <i>Equal-Opportunity</i> gap
      using per-group threshold adjustment — at zero AUC cost.
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Training control
# --------------------------------------------------------------------------- #
trained = ac.models_exist()

if not trained:
    st.warning("No trained models found yet. Click **Train models** to build them "
               "(generates data, trains 5 classifiers, runs the fairness pipeline).")

col_btn, col_status = st.columns([1, 3])
with col_btn:
    if st.button("⚙️ Train models", type="primary", use_container_width=True):
        with st.spinner("Training 5 models on ~70k records — this can take a "
                        "couple of minutes…"):
            ok, output = ac.run_training()
        if ok:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Training complete.")
            with st.expander("Training log"):
                st.code(output[-4000:] or "(no output)")
            st.rerun()
        else:
            st.error("Training failed — see log below.")
            st.code(output[-4000:] or "(no output)")
with col_status:
    if trained:
        meta = ac.load_meta()
        st.success(
            f"Models ready · {meta['n_rows']:,} records · "
            f"best model **{meta['best_model']}** · "
            f"data source: {'real UCI CSV' if meta['is_real'] else 'synthetic (UCI schema)'}"
        )

st.divider()

if not trained:
    st.info("Once training finishes, the dataset summary, model leaderboard, and "
            "all six analysis pages (sidebar) become available.")
    st.stop()

# --------------------------------------------------------------------------- #
# Dataset summary cards
# --------------------------------------------------------------------------- #
meta = ac.load_meta()
metrics = ac.load_metrics()
fair = ac.load_fair()

st.subheader("Dataset")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Patient encounters", f"{meta['n_rows']:,}")
c2.metric("Features", len(meta["feature_names"]))
c3.metric("30-day readmission rate", f"{meta['prevalence']:.1%}")
c4.metric("Train / Test", f"{meta['n_train']:,} / {meta['n_test']:,}")
c5.metric("Models trained", len(meta["models"]))

st.divider()

# --------------------------------------------------------------------------- #
# Model leaderboard
# --------------------------------------------------------------------------- #
st.subheader("Model leaderboard")
table = (
    pd.DataFrame(metrics).T[["auc", "f1", "accuracy", "precision", "recall"]]
    .sort_values("auc", ascending=False)
    .rename(columns=str.upper)
)
st.dataframe(
    table.style.format("{:.4f}")
    .background_gradient(cmap="Greens", subset=["AUC"])
    .highlight_max(axis=0, props="font-weight:bold;color:#0d9488"),
    use_container_width=True,
)
st.caption(f"Best model by AUC: **{meta['best_model']}**. Metrics computed at "
           "each model's Youden-optimal threshold on the held-out test set.")

st.divider()

# --------------------------------------------------------------------------- #
# Fairness headline
# --------------------------------------------------------------------------- #
st.subheader("Fairness headline")
f1, f2, f3 = st.columns(3)
f1.metric("Equal-Opportunity gap (before)",
          f"{fair['eo_gap_before'] * 100:.1f} pp")
f2.metric("Equal-Opportunity gap (after)",
          f"{fair['eo_gap_after'] * 100:.1f} pp",
          delta=f"{(fair['eo_gap_after'] - fair['eo_gap_before']) * 100:.1f} pp",
          delta_color="inverse")
f3.metric("AUC cost of correction",
          f"{(fair['auc_after'] - fair['auc_before']) * 100:.2f} pp")
st.caption("Per-group threshold adjustment equalises the true-positive rate "
           "across racial groups while leaving model ranking — and therefore "
           "AUC — unchanged. See the **Fairness Dashboard** page.")

with st.sidebar:
    st.markdown("### Navigate")
    st.markdown(
        "- **EDA** — distributions & correlations\n"
        "- **Model Performance** — ROC / PR / confusion\n"
        "- **Statistical Analysis** — DeLong & Kruskal–Wallis\n"
        "- **Fairness Dashboard** — equal-opportunity audit\n"
        "- **Single Patient** — one prediction + SHAP\n"
        "- **Batch Prediction** — score a CSV"
    )
    st.caption("Re-run `python train.py` any time to retrain.")
