# Customer-Churn-Prediction-Analytics-Python-Project

## Project Overview
This project analyzes customer churn to understand why customers leave, then builds an explainable **decision** model to predict churn risk and support retention decisions. The workflow mirrors a real analytics engagement: business framing → exploratory data analysis (EDA) → Decision Tree modeling (gini/entropy with depth tuning) → model evaluation (classification report & confusion matrices) → stakeholder-ready recommendations and KPIs.

## Key Results
- **Goal:** Identify churn drivers and predict at-risk customers.
- **Built:** End-to-end Python workflow with EDA, Decision Tree (ML), and evaluation (accuracy/precision/recall/F1 + confusion matrices).
- **How:** Cleaned `CustomerChurnData.csv`, one-hot encoded categoricals **after** the split, aligned train/test features with `DataFrame.align(..., fill_value=0)`, and trained four Decision Tree models by varying criterion (gini vs. entropy) and depth (shallow vs. deeper) to compare performance.
- **Best Model:** **Decision Tree (Gini, full depth)** — **Accuracy:** 88%, **Precision (Churn):** 0.84, **Recall (Churn):** 0.68, **F1 (Churn):** 0.75  
  *(Approx confusion counts on test n=300: TP ≈ 52, FN ≈ 25, FP ≈ 10, TN ≈ 213)*

## Concepts & Techniques Used

### Data Loading
- Loaded CSV with `pandas.read_csv()` from Colab/Drive.
- Verified structure with `df.head()` and `df.info()`; confirmed presence of target `churn`.

### Data Cleaning
- Scanned missingness with `df.isnull().sum()`; applied simple imputations (median for numerics, mode for categoricals) or dropped low-value sparse fields.
- Normalized dtypes/labels to avoid duplicate dummies; reviewed ranges/outliers; optionally bucketed features (e.g., tenure bands) where useful.

### Exploratory Data Analysis (EDA)
- Descriptive stats (`df.describe()`), distribution checks (tenure/usage/spend).
- Churn vs. non-churn comparisons to surface likely drivers.

### Visualization
- Matplotlib bar/hist plots for key distributions and churn breakdowns.
- Confusion matrices (`ConfusionMatrixDisplay`) to visualize FP/FN trade-offs.

### Modeling (Machine Learning)
- **Split:** `train_test_split` (70/30) with `stratify=churn` and `random_state=1`.
- **Encoding:** `pd.get_dummies()` performed **after** the split on train/test; columns aligned with `DataFrame.align(..., fill_value=0)`.
- **Classifier:** `sklearn.tree.DecisionTreeClassifier` with **gini** and **entropy** criteria.
- **Variants Trained (4):**
  1) Entropy, max_depth = 3  
  2) Gini, max_depth = 3  
  3) Entropy, max_depth = 5  
  4) **Gini, full depth (max_depth=None)**
- **Rationale:** Depth and criterion tuned for interpretability and improved recall on the churn class.

### Evaluation (Metrics)
- `classification_report` with **Accuracy, Precision, Recall, F1** (emphasis on the churn = “Y” class).
- Confusion matrices to quantify costs of missed churners (FN) and over-flagging (FP).
- **Model Comparison (Churn class):**
  - Entropy, depth=3 → Acc 81%, Prec 0.68, Rec 0.51, F1 0.58  
  - Gini, depth=3 → Acc 82%, Prec 0.69, Rec 0.57, F1 0.62  
  - Entropy, depth=5 → Acc 81%, Prec 0.72, Rec 0.40, F1 0.52  
  - **Gini, full depth → Acc 88%, Prec 0.84, Rec 0.68, F1 0.75 (best)**

## CRISP-DM (What I Did at Each Stage)

### 1) Business Understanding
- **Objective:** Identify customers most likely to churn and the drivers behind attrition so teams can target interventions.
- **Decisions enabled:** Who to target, timing of outreach, and how to measure improvement.
- **Success focus:** Prioritize recall on the churn class while monitoring precision.
- **Constraints/assumptions:** Single-period snapshot; model must be explainable (tree-based) for non-technical stakeholders.

### 2) Data Understanding
- **Source:** `CustomerChurnData.csv` (loaded from Colab/Drive).
- **Schema review:** `df.head()`, `df.info()`, `df.describe()` to confirm dtypes, ranges, non-null counts.
- **Label balance:** Checked churn vs. non-churn distribution to set metric expectations.
- **Signals (EDA):** Churn vs. non-churn cohort comparisons; distribution and category mix inspection.

### 3) Data Preparation
- **Missing values:** Scanned and applied simple imputations (median/mode) or dropped sparse fields.
- **Categoricals:** Normalized labels; one-hot encoded with `pd.get_dummies()`.
- **Train/test consistency:** `DataFrame.align(..., fill_value=0)` to ensure identical columns/order across splits.
- **Feature sanity:** Verified no all-zero or duplicate columns post-alignment.

### 4) Modeling (Machine Learning)
- **Split:** `train_test_split` (70/30) with `stratify=churn`, fixed seed.
- **Model family:** DecisionTreeClassifier for interpretability; tried gini and entropy criteria.
- **Hyperparameters:** Tuned `max_depth` (and full depth) to balance generalization and churn recall.
- **Training:** Fit on train set; predicted on held-out test set.

### 5) Evaluation
- **Metrics:** Accuracy, Precision, Recall, F1 via `classification_report` with attention to churn recall.
- **Confusion matrices:** Visualized FP/FN trade-offs and operational costs.
- **Model choice:** Selected **Gini, full depth** for best balance of accuracy and churn-class F1/recall; documented trade-offs.

## Next Steps
- **Pruning & tuning:** Cross-validate `ccp_alpha`, `min_samples_leaf`, `min_samples_split` to reduce overfitting while maintaining recall.
- **Threshold calibration:** Choose operating point by cost–benefit (missed churn vs. outreach cost); consider `class_weight='balanced'` if higher recall is needed.
- **Model families:** Benchmark against Random Forest / Gradient Boosting / XGBoost for robustness.
- **Monitoring:** Track precision/recall by month and by segment; retrain on a schedule to handle drift.
