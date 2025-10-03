# Customer-Churn-Prediction-Analytics-Python-Project
This project analyzes **customer churn** to understand **what customers leave** and **why**, then builds an explainable decuision model to **predict churn risk** and support retention decisions. The workflow mirrors a real analytics engagement: business framing → **exploratory data analysis (EDA)** → **Decision Tree** modeling (gini/entropy with depth tuning), classification models → model evaluation (classification report & confusion matrices) → **stakeholder-ready recommendations** and KPIs.

---

# Methods & Techniques Used

### Data Loading
- Loaded CSV with `pandas.read_csv()` from the original Colab/Drive path:  
  `/content/drive/MyDrive/IS470 Colab Datasets/CutomerChurnData.csv`  
  *(In this repo, place the file under `data/` and update to `data/CutomerChurnData.csv`.)*
- Confirmed structure with `df.head()` and `df.info()`; validated presence of target `churn`.

### Data Cleaning
- Scanned missingness with `df.isnull().sum()`; applied simple imputations (median for numerics, mode for categoricals) or dropped low-value sparse fields.
- Normalized dtypes (categorical vs numeric) and standardized category labels to avoid duplicate dummies.
- Reviewed ranges/outliers; created practical buckets (e.g., tenure bands) where useful.

### Exploratory Data Analysis (EDA)
- Descriptive stats with `df.describe()`; distribution checks for tenure/usage/spend.
- Segment cuts (churn vs non-churn) to inspect median/mean shifts and rate deltas.
- Quick association checks to prioritize likely drivers for modeling and communication.

### Visualization
- Matplotlib bar/hist plots for key distributions and churn breakdowns.
- Confusion matrix visualizations (`ConfusionMatrixDisplay`) to interpret FP/FN trade-offs.

### Modeling (Machine Learning)
- Train/test split (`train_test_split`, 70/30) with `stratify=churn` and fixed `random_state`.
- One-hot encoding with `pd.get_dummies()` on both train/test; **column alignment** via `DataFrame.align(..., fill_value=0)` to keep features identical.
- **Classifier:** `sklearn.tree.DecisionTreeClassifier` with **gini** and **entropy** criteria.
- **Hyperparameters:** tuned `max_depth` for a balance of interpretability, generalization, and churn **recall**.

### Evaluation (Metrics)
- `classification_report` with **Accuracy, Precision, Recall, F1** (emphasis on the churn class).
- Confusion matrices to quantify costs of missed churners (FN) vs over-flagging (FP).
