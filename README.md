# Customer-Churn-Prediction-Analytics-Python-Project
This project analyzes **customer churn** to understand **who leaves** and **why**, then builds an explainable model to **predict churn risk** and support retention decisions. The workflow mirrors a real analytics engagement: business framing → **exploratory data analysis (EDA)** → **Decision Tree** modeling (gini/entropy with depth tuning) → model evaluation (classification report & confusion matrices) → **stakeholder-ready recommendations** and KPIs.

---

# Methods & Techniques Used

## Data Loading
- Loaded the dataset with `pandas.read_csv()`:
  ```python
  import pandas as pd
  df = pd.read_csv('/content/drive/MyDrive/IS470 Colab Datasets/CutomerChurnData.csv')
  ```
  _Repo note:_ place the CSV under `data/` and update the path (e.g., `data/CutomerChurnData.csv`).

## Data Cleaning
- **Missing values scan:** `df.isnull().sum()` to identify fields requiring attention.
- **Null handling strategy:**
  - If a feature was **non-critical** (low correlation to churn, high sparsity), rows were dropped or the field was excluded.
  - If a feature was **important**, simple imputations were applied (mode for categoricals, median for numerics) to avoid biasing distributions.
- **Type normalization:** ensured correct dtypes (e.g., category vs. numeric) for consistent encoding and stats.
- **Outlier & range checks:** reviewed histograms/quantiles; where helpful, created **tenure buckets** or clipped extreme values to stabilize splits.
- **Category hygiene:** standardized text categories (e.g., casing, trailing spaces) so one-hot encoding produced a clean, minimal set of columns.
- **Target label check:** validated `churn` label presence and class balance to set expectations for evaluation (recall emphasis when catching churners is the goal).

## Analysis (EDA)
- **Descriptive statistics:** `df.describe()` for central tendency and spread on tenure/usage/spend fields.
- **Distributions & counts:** `value_counts()` and plots for plan/region/internet/add-ons to understand mix.
- **Churn vs. non-churn comparisons:** group-wise means/medians and rate deltas to surface early risk patterns.
- **Quick associations:** simple correlation-style checks to prioritize likely drivers for modeling and interpretation.

## Feature Preparation
- **One-hot encoding:** `pd.get_dummies()` applied to both train and test sets for categorical variables.
- **Column alignment:** `train, test = train.align(test, join="left", axis=1, fill_value=0)` to guarantee identical feature matrices post-encoding.
- **Feature sanity checks:** verified there were no all-zero or duplicate columns after alignment.

## Modeling
- **Split:** `train_test_split(test_size=0.3, random_state=1, stratify=target)` for reproducible, balanced evaluation.
- **Models:** `sklearn.tree.DecisionTreeClassifier` trained with **gini** and **entropy** criteria.
- **Depth tuning:** experimented with `max_depth` (e.g., shallow vs. deeper trees) to balance interpretability, generalization, and **recall on churn**.

## Evaluation & Visualization
- **Metrics:** `classification_report` with overall **Accuracy** and class-level **Precision/Recall/F1** (focus on the churn class).
- **Confusion matrices:** `confusion_matrix` + `ConfusionMatrixDisplay` to understand **false negatives** (missed churners) vs. **false positives**.
- **Interpretation:** documented precision–recall trade-offs and signs of overfitting for deeper trees.

---

# Data Loading (simple summary)
- Original file name in notebook: **`CutomerChurnData.csv`**
- Original Colab/Drive path: `/content/drive/MyDrive/IS470 Colab Datasets/CutomerChurnData.csv`
- Repo usage: place the file in `data/` and reference it as `data/CutomerChurnData.csv`.

---

# Data Inspection
- `df.head()` and `df.info()` to confirm shape, dtypes, and non-null counts.
- Checked `churn` class balance to guide metric emphasis (opt for higher **recall** if catching churners is the priority).

---

# Exploratory Data Analysis (EDA)
- Descriptive stats (`describe()`), distribution plots, and category counts.
- Churn vs. non-churn comparisons to surface patterns (e.g., shorter **tenure**, certain **plan** categories, absence of **add-ons**).
- EDA findings informed feature prep (bucketing/flags) and model depth choices.

---

# Data Analysis (Modeling)
- Train/test split (70/30) with stratification.
- One-hot encoding on train/test; **column alignment** to keep feature sets identical.
- Trained Decision Trees with **entropy** (interpretable splits) and **gini** (baseline & full-depth) for comparison.
- Tuned `max_depth` to manage bias/variance and improve **recall** for the churn class.

---

# Model Evaluation
- Reported **Accuracy, Precision, Recall, F1** via `classification_report` on holdout test data.
- Visualized **confusion matrices** to quantify precision–recall trade-offs (FP vs. FN).
- Observed pattern: **shallower trees** → better generalization but lower churn recall; **deeper trees** → higher recall with some overfitting risk.

> Add your exported confusion-matrix image here:  
> `![Confusion Matrix](reports/model-metrics.png)`

---

# Findings & Recommendations
- **Patterns/Drivers:** *(replace with your specifics)* e.g., **short tenure**, specific **plan** types, absence of **internet/add-ons** associate with higher churn.
- **High-risk segment:** *(name a cohort)* shows **~K×** higher churn than baseline.
- **Recommendations:**
  1) **Target Segment X** with [offer/feature/messaging] during [time window].
  2) **Trigger outreach** when **[driver]** crosses **[threshold]** to catch at-risk users earlier (boost recall).
  3) **Measure lift** via monthly retention KPIs and churn rankers by segment; recalibrate thresholds/depths as behavior shifts.

---

# CRISP-DM Framework
**What it is:** Cross-Industry Standard Process for Data Mining — a practical lifecycle for analytics/ML projects.

**How it’s used here:**
1. **Business Understanding** — Reduce churn by identifying at-risk customers and key drivers; enable who/when/how to intervene and how to measure improvement.
2. **Data Understanding** — Profiled schema, label balance, distributions; compared churn vs. non-churn cohorts to surface early signals.
3. **Data Preparation** — Cleaned missing values, normalized categories, one-hot encoded features, and aligned columns across train/test for consistent inputs.
4. **Modeling** — Trained Decision Trees (gini & entropy) with `max_depth` tuning; compared interpretability vs. performance with emphasis on churn recall.
5. **Evaluation** — Reported accuracy/precision/recall/F1; used confusion matrices to understand FN/FP trade-offs and select a configuration aligned to the business goal (capture more true churners).
