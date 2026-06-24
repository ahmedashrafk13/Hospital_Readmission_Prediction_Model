# Hospital Readmission Prediction & Fairness Analysis

Predicts **30-day hospital readmission** for diabetic inpatients, compares five
classifiers with formal statistical tests, and audits + corrects a racial
**Equal-Opportunity** disparity — all wrapped in a six-page Streamlit app.

> **Stack:** XGBoost · scikit-learn · LightGBM · SciPy · SHAP · Streamlit · Plotly

---

## Highlights

- **Five classifiers** — XGBoost, Random Forest, Logistic Regression, Gradient
  Boosting, LightGBM — trained on **~70k** diabetic-patient encounters
  (UCI *Diabetes 130-US hospitals* schema).
- **Statistical model comparison** — pairwise **DeLong** test for correlated
  AUCs and a **Kruskal–Wallis** test across predicted-score distributions.
- **Fairness audit** — measures the **Equal-Opportunity (TPR) gap** across
  racial groups, then applies **per-group threshold adjustment** to shrink it.
  Because AUC is threshold-independent, the correction costs **zero AUC**.
- **Six-page app** — Home, EDA, Model Performance, Statistical Analysis,
  Fairness Dashboard, Single-Patient prediction (with SHAP), Batch prediction.

## Data

The app first looks for the real UCI CSV at `data/diabetic_data.csv`
([download here](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)).
If it is absent, a **realistic synthetic dataset** matching the UCI schema is
generated — ~11% readmission prevalence and a deliberately seeded TPR gap across
racial groups (the recorded features of true positives carry weaker signal for
historically under-documented groups, so a race-blind model still under-detects
them — exactly the disparity the fairness page corrects).

## Quick start

```bash
cd hospital_readmission
pip install -r requirements.txt

# Train the models (generates data, trains 5 models, runs the fairness pipeline)
python train.py

# Launch the app
streamlit run app.py
```

Or skip the CLI step: launch the app and click **Train models** on the Home page.

## Project layout

```
hospital_readmission/
├── app.py                      # Home: leaderboard, dataset summary, train button
├── train.py                    # Trains 5 models + fairness pipeline, writes models/
├── data_utils.py               # Load real UCI CSV or generate synthetic data
├── stats_utils.py              # DeLong AUC test, Kruskal–Wallis
├── fairness_utils.py           # EO gap, per-group threshold adjustment, metrics
├── app_common.py               # Cached loaders, theming, helpers
├── pages/
│   ├── 1_EDA.py
│   ├── 2_Model_Performance.py
│   ├── 3_Statistical_Analysis.py
│   ├── 4_Fairness_Dashboard.py
│   ├── 5_Single_Patient.py
│   └── 6_Batch_Prediction.py
├── models/                     # Written by train.py (pickles + JSON artifacts)
├── requirements.txt
└── README.md
```

## Notes

- `train.py` is seeded (`SEED = 42`) so results are reproducible.
- If `xgboost` / `lightgbm` / `shap` are not installed, the code degrades
  gracefully (HistGradientBoosting / ExtraTrees fallbacks; global feature
  importance instead of SHAP).
