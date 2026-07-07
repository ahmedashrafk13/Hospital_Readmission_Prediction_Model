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

## Results

Trained on the **real UCI "Diabetes 130-US hospitals"** dataset (`seed=42`).

| Metric | Value |
| --- | --- |
| Patient encounters | 101,766 |
| Features | 20 |
| 30-day readmission rate | 11.2% |
| Train / Test split | 81,412 / 20,354 |
| Models trained | 5 |

**Model leaderboard** — metrics computed at each model's Youden-optimal threshold on the held-out test set, sorted by AUC:

| Model | AUC | F1 | Accuracy | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| **Gradient Boosting** | **0.6465** | 0.2598 | 0.6408 | 0.1687 | 0.5649 |
| XGBoost | 0.6453 | 0.2530 | 0.5717 | 0.1570 | 0.6499 |
| LightGBM | 0.6445 | 0.2539 | 0.6224 | 0.1629 | 0.5760 |
| Random Forest | 0.6408 | 0.2411 | 0.4628 | 0.1431 | 0.7649 |
| Logistic Regression | 0.6402 | 0.2538 | 0.6252 | 0.1631 | 0.5711 |

Best model by AUC: **Gradient Boosting**.

**Fairness (Equal-Opportunity / TPR gap across racial groups)** — per-group threshold
adjustment shrinks the gap from **18.1 pp → 1.5 pp** at **zero AUC cost** (0.6465
before and after, since ranking is unchanged).

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
