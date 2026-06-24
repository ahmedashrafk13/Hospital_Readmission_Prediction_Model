"""
Shared helpers for the Streamlit app: artifact loading (cached), training
trigger, theming, and small plotting utilities.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import joblib
import streamlit as st

import data_utils as du

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

# --- clinical palette ------------------------------------------------------ #
NAVY = "#0f172a"
TEAL = "#0d9488"
SLATE = "#475569"
AMBER = "#d97706"
RED = "#dc2626"
GREEN = "#059669"

MODEL_COLORS = {
    "XGBoost": "#0d9488",
    "Random Forest": "#2563eb",
    "Logistic Regression": "#d97706",
    "Gradient Boosting": "#7c3aed",
    "LightGBM": "#db2777",
}
GROUP_COLORS = [
    "#0d9488", "#2563eb", "#d97706", "#7c3aed", "#db2777", "#0891b2", "#65a30d",
]


# --------------------------------------------------------------------------- #
# Artifact presence / training
# --------------------------------------------------------------------------- #
def models_exist() -> bool:
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in
               ("metrics.json", "fair_thresholds.json", "eval.pkl", "meta.json"))


def run_training() -> tuple[bool, str]:
    """Run train.py as a subprocess. Returns ``(ok, combined_output)``."""
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "train.py")],
        cwd=HERE, capture_output=True, text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out


def require_models() -> None:
    """Stop the page with a friendly message if artifacts are missing."""
    if not models_exist():
        st.warning(
            "No trained models found. Go to the **Home** page and click "
            "**Train models**, or run `python train.py` from the project folder."
        )
        st.stop()


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    with open(os.path.join(MODELS_DIR, name)) as f:
        return json.load(f)


def load_metrics() -> dict:
    return load_json("metrics.json")


def load_fair() -> dict:
    return load_json("fair_thresholds.json")


def load_meta() -> dict:
    return load_json("meta.json")


@st.cache_resource(show_spinner=False)
def load_eval() -> dict:
    return joblib.load(os.path.join(MODELS_DIR, "eval.pkl"))


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    keys = load_meta()["model_keys"]
    key = keys[model_name]
    return joblib.load(os.path.join(MODELS_DIR, f"{key}.pkl"))


@st.cache_data(show_spinner=False)
def get_data(n: int | None = None, seed: int | None = None):
    meta = load_meta() if models_exist() else {"n_rows": 70000, "seed": 42}
    df, is_real = du.load_data(n=n or meta["n_rows"], seed=seed or meta["seed"])
    return df, is_real


# --------------------------------------------------------------------------- #
# Plotly theming
# --------------------------------------------------------------------------- #
def style_fig(fig, height: int | None = None, title: str | None = None):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="sans-serif", size=13, color=NAVY),
        title=dict(text=title, font=dict(size=17, color=NAVY)) if title else None,
        margin=dict(l=40, r=20, t=50 if title else 20, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        colorway=GROUP_COLORS,
    )
    if height:
        fig.update_layout(height=height)
    return fig


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"<h1 style='color:{NAVY};margin-bottom:0'>{title}</h1>"
        + (f"<p style='color:{SLATE};font-size:1.02rem;margin-top:4px'>{subtitle}</p>"
           if subtitle else ""),
        unsafe_allow_html=True,
    )
