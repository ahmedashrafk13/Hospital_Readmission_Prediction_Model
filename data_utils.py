"""
Data loading & preprocessing for the Hospital Readmission project.

Tries to load the real UCI "Diabetes 130-US hospitals" CSV from
``data/diabetic_data.csv``. If it is not present, a realistic synthetic
dataset that matches the UCI schema is generated instead.

The synthetic generator deliberately seeds an Equal-Opportunity (TPR) gap
across racial groups: the predictive *signal strength* of the recorded
clinical features is weaker for historically under-documented groups, so a
model trained without the race attribute still under-detects true positives
for those groups -- exactly the disparity the fairness module corrects.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Schema constants
# --------------------------------------------------------------------------- #
RACE_LEVELS = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]
GENDER_LEVELS = ["Female", "Male"]
AGE_LEVELS = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]
A1C_LEVELS = ["None", "Norm", ">7", ">8"]
GLU_LEVELS = ["None", "Norm", ">200", ">300"]
MED_LEVELS = ["No", "Steady", "Up", "Down"]
CHANGE_LEVELS = ["No", "Ch"]
DIABETESMED_LEVELS = ["No", "Yes"]

MED_COLS = [
    "metformin", "insulin", "glipizide", "glyburide",
    "pioglitazone", "rosiglitazone",
]

NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

# Categorical columns used as model features (race is excluded -- it is the
# sensitive attribute, kept separate for the fairness analysis).
CATEGORICAL_COLS = (
    ["gender", "age", "A1Cresult", "max_glu_serum"]
    + MED_COLS
    + ["change", "diabetesMed"]
)

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
SENSITIVE_COL = "race"
TARGET_COL = "readmitted"

# Deterministic ordinal encoders (so train-time and inference-time encoding
# always agree -- no LabelEncoder state to persist or accidentally refit).
ENCODERS: dict[str, dict[str, int]] = {
    "gender": {v: i for i, v in enumerate(GENDER_LEVELS)},
    "age": {v: i for i, v in enumerate(AGE_LEVELS)},
    "A1Cresult": {v: i for i, v in enumerate(A1C_LEVELS)},
    "max_glu_serum": {v: i for i, v in enumerate(GLU_LEVELS)},
    "change": {v: i for i, v in enumerate(CHANGE_LEVELS)},
    "diabetesMed": {v: i for i, v in enumerate(DIABETESMED_LEVELS)},
    **{c: {v: i for i, v in enumerate(MED_LEVELS)} for c in MED_COLS},
}

# Race-specific signal strength. Lower -> the recorded features of true
# positives overlap more with negatives -> the model under-detects positives
# -> lower TPR -> Equal-Opportunity gap.
RACE_SIGNAL = {
    "Caucasian": 1.00,
    "Asian": 1.00,
    "Other": 0.85,
    "Hispanic": 0.52,
    "AfricanAmerican": 0.42,
}

# Generative knobs (tuned so the trained model lands at a realistic AUC ~0.70
# with a ~17pp Equal-Opportunity gap before correction).
LABEL_INTERCEPT = -2.18   # controls 30-day readmission prevalence (~11%)
SIGNAL_SCALE = 0.18       # global multiplier on how strongly the label shifts features
LABEL_FLIP = 0.10         # fraction of labels decoupled from features (adds noise)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #
def generate_synthetic_data(n: int = 70000, seed: int = 42,
                            signal_scale: float = SIGNAL_SCALE,
                            intercept: float = LABEL_INTERCEPT,
                            flip: float = LABEL_FLIP,
                            race_signal: dict | None = None) -> pd.DataFrame:
    """Generate ~`n` rows matching the UCI diabetes schema.

    Produces a ~11% 30-day readmission rate and an engineered Equal-Opportunity
    gap across racial groups (see module docstring). The generative knobs
    (``signal_scale``, ``intercept``, ``flip``) are exposed for calibration.
    """
    rng = np.random.default_rng(seed)
    race_signal = race_signal or RACE_SIGNAL

    # --- demographics ----------------------------------------------------- #
    race = rng.choice(RACE_LEVELS, size=n, p=[0.749, 0.189, 0.020, 0.013, 0.029])
    gender = rng.choice(GENDER_LEVELS, size=n, p=[0.537, 0.463])
    age = rng.choice(
        AGE_LEVELS, size=n,
        p=[0.002, 0.007, 0.018, 0.038, 0.095, 0.170, 0.224, 0.256, 0.166, 0.024],
    )
    age_mid = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
    age_index = {a: i for i, a in enumerate(AGE_LEVELS)}
    age_num = np.array([age_mid[age_index[a]] for a in age], dtype=float)

    s = np.array([race_signal[r] for r in race])  # per-patient signal strength

    # --- latent readmission label (race-independent prevalence) ----------- #
    base_logit = intercept + 0.014 * (age_num - 65.0) + rng.normal(0, 0.45, n)
    y = (rng.random(n) < _sigmoid(base_logit)).astype(int)

    # `yf` drives the feature shifts. A fraction `flip` of labels are decoupled
    # from the features (label noise) so the trained model lands at a realistic
    # AUC rather than separating the classes perfectly.
    flip_mask = rng.random(n) < flip
    yf = y.astype(float).copy()
    yf[flip_mask] = rng.integers(0, 2, size=int(flip_mask.sum())).astype(float)
    sig = signal_scale * s * yf  # effective per-patient signal

    # --- clinical features conditioned on (label, signal) ----------------- #
    number_inpatient = rng.poisson(0.35 + 2.30 * sig).clip(0, 21)
    number_emergency = rng.poisson(0.18 + 0.95 * sig).clip(0, 76)
    number_outpatient = rng.poisson(0.40 + 0.45 * sig).clip(0, 42)
    time_in_hospital = np.round(
        rng.normal(3.8 + 3.0 * sig, 2.4)).clip(1, 14).astype(int)
    num_medications = np.round(
        rng.normal(15 + 6.5 * sig, 7.0)).clip(1, 81).astype(int)
    number_diagnoses = np.round(
        rng.normal(6.4 + 2.6 * sig, 2.6)).clip(1, 16).astype(int)
    num_lab_procedures = np.round(
        rng.normal(43 + 6.0 * sig, 20.0)).clip(1, 132).astype(int)
    num_procedures = rng.poisson(1.2 + 0.6 * sig).clip(0, 6)

    # --- categorical clinical features (also signal-scaled) --------------- #
    a1c_risk = _sigmoid(-0.5 + 2.2 * sig + rng.normal(0, 0.6, n))
    A1Cresult = np.where(
        rng.random(n) < 0.45, "None",
        np.where(a1c_risk > 0.72, ">8",
                 np.where(a1c_risk > 0.52, ">7", "Norm")),
    )
    glu_risk = _sigmoid(-1.0 + 1.5 * sig + rng.normal(0, 0.6, n))
    max_glu_serum = np.where(
        rng.random(n) < 0.90, "None",
        np.where(glu_risk > 0.7, ">300",
                 np.where(glu_risk > 0.5, ">200", "Norm")),
    )

    insulin_p = _sigmoid(-0.3 + 1.4 * sig + rng.normal(0, 0.5, n))
    insulin = np.where(insulin_p > 0.70, "Up",
                       np.where(insulin_p > 0.45, "Steady",
                                np.where(insulin_p > 0.30, "Down", "No")))

    def _rand_med(steady_p: float) -> np.ndarray:
        return rng.choice(
            MED_LEVELS, size=n,
            p=[1 - steady_p - 0.04, steady_p, 0.02, 0.02],
        )

    metformin = _rand_med(0.20)
    glipizide = _rand_med(0.12)
    glyburide = _rand_med(0.11)
    pioglitazone = _rand_med(0.07)
    rosiglitazone = _rand_med(0.06)

    diabetesMed = np.where(rng.random(n) < 0.77, "Yes", "No")
    change = np.where(_sigmoid(-0.6 + 1.2 * sig) > rng.random(n), "Ch", "No")

    df = pd.DataFrame({
        "race": race,
        "gender": gender,
        "age": age,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "A1Cresult": A1Cresult,
        "max_glu_serum": max_glu_serum,
        "metformin": metformin,
        "insulin": insulin,
        "glipizide": glipizide,
        "glyburide": glyburide,
        "pioglitazone": pioglitazone,
        "rosiglitazone": rosiglitazone,
        "change": change,
        "diabetesMed": diabetesMed,
        "readmitted": y,
    })
    return df


# --------------------------------------------------------------------------- #
# Real-UCI loader (maps the raw CSV onto our schema)
# --------------------------------------------------------------------------- #
def _load_real_uci(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, na_values=["?"])
    raw["readmitted"] = (raw["readmitted"] == "<30").astype(int)
    raw["race"] = raw.get("race", "Other").fillna("Other")
    # Keep only schema columns that exist; fill the rest with sensible defaults.
    df = pd.DataFrame()
    df["race"] = raw["race"].where(raw["race"].isin(RACE_LEVELS), "Other")
    df["gender"] = raw.get("gender", "Female").where(
        raw.get("gender", pd.Series(["Female"] * len(raw))).isin(GENDER_LEVELS),
        "Female",
    )
    df["age"] = raw.get("age", "[60-70)")
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(raw.get(col, 0), errors="coerce").fillna(0)
    df["A1Cresult"] = raw.get("A1Cresult", "None").fillna("None")
    df["max_glu_serum"] = raw.get("max_glu_serum", "None").fillna("None")
    for col in MED_COLS:
        df[col] = raw.get(col, "No").fillna("No")
    df["change"] = raw.get("change", "No").fillna("No")
    df["diabetesMed"] = raw.get("diabetesMed", "Yes").fillna("Yes")
    df["readmitted"] = raw["readmitted"]
    return df


def load_data(path: str | None = None, n: int = 70000, seed: int = 42) -> tuple[pd.DataFrame, bool]:
    """Return ``(dataframe, is_real)``.

    Loads the real UCI CSV if found; otherwise generates synthetic data.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [path] if path else []
    candidates.append(os.path.join(here, "data", "diabetic_data.csv"))
    for cand in candidates:
        if cand and os.path.exists(cand):
            try:
                return _load_real_uci(cand), True
            except Exception:
                break
    return generate_synthetic_data(n=n, seed=seed), False


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def encode_value(col: str, value) -> int:
    """Encode a single categorical value with the deterministic mapping."""
    mapping = ENCODERS.get(col, {})
    return int(mapping.get(value, 0))


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Return ``(X, y, sensitive)`` with categoricals ordinally encoded."""
    X = pd.DataFrame(index=df.index)
    for col in NUMERIC_COLS:
        X[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
    for col in CATEGORICAL_COLS:
        X[col] = df[col].map(ENCODERS[col]).fillna(0).astype(int)
    X = X[FEATURE_COLS]
    y = df[TARGET_COL].astype(int).to_numpy()
    sensitive = df[SENSITIVE_COL].astype(str).reset_index(drop=True)
    return X, y, sensitive


def build_input_row(values: dict) -> pd.DataFrame:
    """Build a single encoded feature row from a dict of raw values.

    Missing keys fall back to neutral defaults. Used by the single-patient page.
    """
    row = {}
    for col in NUMERIC_COLS:
        row[col] = float(values.get(col, 0))
    for col in CATEGORICAL_COLS:
        row[col] = encode_value(col, values.get(col, list(ENCODERS[col])[0]))
    return pd.DataFrame([row])[FEATURE_COLS]
