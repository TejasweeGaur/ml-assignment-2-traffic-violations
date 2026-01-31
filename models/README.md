
# Indian Traffic Violations — Multi‑Class Classification

## Problem Context

Predict the **type of traffic violation** from structured inputs such as time of day, road category, vehicle type, speed, weather, location (city/sector/police station), and driver attributes (age band, license class).

**Violation Types (classes):**

- `Drunk Driving`, `Driving without License`,  `No Helmet`, `No Seatbelt`, `Signal Jumping`, `Wrong Parking`, `Overspeeding`, `Overloading`, `Other`

**Why multi‑class?** A single incident belongs to one **primary violation** among many categories; the model must **discriminate** between several plausible outcomes, not just “violation vs no‑violation”.

## Evaluation for Multi‑Class Problems

- **Accuracy** — fraction of correct predictions across all classes.
- **AUC** — for multi‑class, use **One‑vs‑Rest (OVR)** or **OVO** with **macro/weighted** averaging.
- **Precision, Recall, F1** — **weighted** (handles imbalance) and, where needed, **macro** (treats all classes equally).
- **MCC** (Matthews Correlation Coefficient) — robust single‑number summary even under imbalance.
- **Confusion Matrix** — vital for understanding which violations are confused (e.g., `Wrong Lane` vs `Dangerous Driving`).

> **Note:** For enforcement, per‑class **Recall** might matter more for serious offenses (e.g., `Dangerous Driving`), while **Precision** matters to avoid false challans for minor violations.

---

## Model Briefs

### 1) Logistic Regression (Multinomial / OvR)

A linear classifier modeling class probabilities via softmax (multinomial) or a set of binary “one‑vs‑rest” models. It finds a weight per feature per class and assumes a linear decision boundary in the transformed feature space.

**Why it helps for Indian violations:**

- With proper **one‑hot encoding** of location, vehicle type, and enforcement source, it reveals **clear, interpretable correlations** (e.g., higher odds of `No Helmet` for certain 2W segments at late evening hours).
- Acts as a **strong baseline** that’s fast to train and easy to deploy on limited infrastructure (e.g., roadside devices or low‑CPU servers).

**Strengths:**

- **Interpretability**: Inspect coefficients to explain which signals tilt predictions toward a class (great for policy memos).
- **Calibration**: Often provides **well‑calibrated** probabilities after Platt/temperature scaling; useful for threshold fines severity.

**Limitations:**

- Struggles with **non‑linear** interactions (e.g., “late evening + festival + city center”).
- Sensitive to **multicollinearity**; standardize numeric features.

---

### 2) Decision Tree Classifier

A hierarchical set of rules that splits data on features to maximize class purity. Produces if‑else decision paths.

**Why it helps:**

- Mirrors how traffic officers reason:  
  *If road type = city arterial & hour in [18–21] & vehicle = 2W ⇒ high chance of `No Helmet`*.
- Very **interpretable**: export a small tree diagram for briefings to enforcement agencies planning meetings.

**Strengths:**

- Captures **non‑linear** interactions and feature hierarchies naturally.
- Handles mixed feature types (numeric/categorical) with minimal preprocessing.

**Limitations:**

- Can overfit (very deep trees), leading to unstable rules across districts/dates.
- Single trees may be less accurate than ensembles.

---

### 3) k‑Nearest Neighbors (kNN)

A non‑parametric method that predicts the class based on the majority vote of the k closest samples in feature space.

**Why it helps:**

- Automatically adapts to local patterns (e.g., a specific camera junction in Gurugram with unique mixing of offenses at 8–9 AM).
- Useful as a sanity check baseline to see if simple locality in feature space carries predictive power.

**Strengths:**

- No training time; can capture complex class boundaries if distance is meaningful.
- Intuitive “similar cases” explanations for investigators and auditors.

**Limitations:**

- **Slow** at inference for large datasets (needs efficient indexing).
- Sensitive to **feature scaling** and to **curse of dimensionality**; requires careful selection/weighting of features.
- Struggles with high **cardinality** categories unless encoded thoughtfully.

---

### 4) Naive Bayes (Gaussian or Multinomial)

A probabilistic classifier assuming conditional independence of features given the class.  

- **Multinomial NB** is great for **counts/frequencies** (e.g., bucketized speed bins, aggregated violation counts by location/time).  
- **Gaussian NB** assumes features are **normally distributed** (works with continuous, standardized features).

**Why it helps:**

- Extremely **fast** and **memory‑efficient**, suitable for **edge** or **resource‑constrained** deployments (e.g., on‑device triage).
- Provides **probabilistic outputs** that are simple to threshold for different **fine severity** tiers.

**Strengths:**

- Robust with **high‑dimensional sparse** features (from one‑hot encodings).
- Surprisingly strong baseline for **text‑like** or **count‑based** features (e.g., prior violations encoded as counts per category/location).

**Limitations:**

- The **independence** assumption is often violated (e.g., hour and road type are correlated), which can limit accuracy.
- **Multinomial NB** requires non‑negative inputs; scaling pipelines must preserve that.

---

### 5) Random Forest (Ensemble)

An ensemble of decision trees, each trained on bootstrapped samples with feature randomness, final prediction via majority vote.

**Why it helps:**

- Captures **rich non‑linearities** among features common in traffic data (time × location × vehicle × weather).
- Offers **feature importance** to spotlight actionable levers (e.g., posted speed limit quality, specific hotspots, rain conditions).

**Strengths:**

- Generally **strong out‑of‑the‑box** accuracy and robustness.
- Handles **missing values** reasonably (depending on implementation), mixed data types, and noise.

**Limitations:**

- Less interpretable than a single tree (but still more explainable than boosted models via permutation importance, partial dependence).
- Can be **computationally heavier** than linear models.

---

### 6) XGBoost (Gradient Boosted Trees)

A powerful gradient boosting algorithm that builds trees sequentially, each one correcting the previous ones’ residual errors. Often state‑of‑the‑art on structured/tabular data.

**Why it helps:**

- Excels at complex, non‑linear pattern discovery typical of traffic systems (rare but high‑impact interactions like “dust storm + highway construction + late night” driving `Dangerous Driving` or `Wrong Lane`).
- Strong performance with imbalanced data via parameter tuning and scale‑pos weighting (per‑class strategies in multi‑class).

**Strengths:**

- High predictive accuracy, handles missing values, robust to outliers, supports regularization to control overfitting.
- Extensive tooling for feature importance and SHAP explanations to support policy decisions.

**Limitations:**

- More **complex** to tune; risk of overfitting without validation discipline.
- Slightly less interpretable; rely on **global** (importance/PDP) and **local** (SHAP) explainers.

---

## Processing Workflow

The processing workflow typically involves the following steps:
1. **Data Input**: Accepts and validates the input data required for processing.
2. **Preprocessing**: Performs any necessary transformations or preparations on the input data.
3. **Core Processing**: Executes the main logic or algorithm to achieve the desired outcome.
4. **Postprocessing**: Applies any final adjustments or formatting to the processed data.
5. **Output**: Returns the final result or stores it in the appropriate location.

Each step in the workflow is designed to ensure efficiency, accuracy, and maintainability.
This documentation provides an overview of the workflow and its purpose.

1. **Load raw dataset**  
  Import the dataset containing traffic violation records from **Kaggle**, ensuring it includes all relevant features such as time, location, vehicle type, and violation type.

2. **Extract Date/Time features**  
  Derive meaningful features like hour of the day, day of the week from the raw date/time column to capture temporal patterns.

3. **Drop unused columns**  
  Remove irrelevant or redundant columns (e.g., IDs, free-text notes) that do not contribute to the prediction task.

4. **Split into train/test (80/20, stratified)**  
  Divide the dataset into training and testing subsets, ensuring the class distribution remains consistent across both splits.

5. **Build preprocessing pipeline**  
  Design a pipeline to handle missing values, encode categorical variables, scale numeric features, and apply any necessary transformations.

6. **Fit preprocessing only on training data**  
  Apply the preprocessing pipeline exclusively to the training data to avoid data leakage into the test set.

7. **Train model**  
  Use the preprocessed training data to fit the selected machine learning model(s), optimizing **hyperparameters** as needed.

8. **Evaluate on test data**  
  Assess the model's performance on the test set using metrics like accuracy, precision, recall, F1-score, and confusion matrix to ensure generalization.

```text
Raw Dataset (Kaggle)
        ↓
data_analysis.py      → feature decisions
        ↓
data_preparation.py   → train.csv / test.csv
        ↓
data_preprocessing.py → ColumnTransformer
        ↓
MODEL TRAINING        → Trains all 6 models
        ↓
Saved .pkl files      → Saves the trained object so we don't have to run the training every time application runs
        ↓
Streamlit UI          → User Interface to upload and test all 6 models based on metrics and confusion matrix