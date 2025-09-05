

  ## 🟢 GlucoTrack – Beginner Track

## ✅ Week 1: Exploratory Data Analysis (EDA)

---

### 📦 1. Data Integrity & Structure

**Q:** Are there any missing, duplicate, or incorrectly formatted entries in the dataset?  
**A:** No missing values were found across the 253,680 instances. A total of **24,206 rows (~9.5%)** were flagged as duplicates and removed to avoid bias from repeated observations. No formatting issues were present. However, all features were stored as `float64`; several should instead be integers or categoricals.

**Q:** Are all data types appropriate (e.g., numeric, categorical)?  
**A:** Not fully. While numeric measures (e.g., BMI, MentHlth, PhysHlth) were correctly stored as floats, many **binary** and **ordinal** variables were misclassified as continuous. We corrected these using domain-informed mappings.

**Feature types overview**

| Type                  | Columns                                                                                                                                                                                                                                     |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Binary categorical    | Outcome, Sex, HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk                                                                     |
| Ordinal / Categorical | GenHlth, Age, Education, Income                                                                                                                                                                                                             |
| Numeric               | BMI, MentHlth, PhysHlth                                                                                                                                                                                                                     |

**Q:** Did you detect any constant, near-constant, or irrelevant features?  
**A:** No constant features. At this stage nothing was dropped as irrelevant; all were retained for modeling and later importance checks.

---

### 🎯 2. Target Variable Assessment

**Q:** What is the distribution of `Diabetes_binary`?  
**A:** After removing duplicates, the target remains **imbalanced**: **84.7% = 0 (no diabetes)** vs **15.3% = 1 (diabetes)**.

**Q:** How might this imbalance influence metrics/strategy?  
**A:** Accuracy is misleading. We emphasize **Recall**, **Precision**, **F1**, **ROC-AUC**, and **PR-AUC**, and use strategies like **class weights** and **resampling (SMOTE)**.

---

### 📊 3. Feature Distribution & Quality

**Q:** Which numerical features are skewed or contain outliers?  
**A:** Using a Z-score > 3, **15,328** potential outlier rows were flagged across **BMI, MentHlth, PhysHlth**. MentHlth and PhysHlth are strongly right-skewed; BMI is moderately right-skewed.

**Q:** Any unrealistic/problematic values?  
**A:** None outside expected ranges, but many statistically unusual values exist (especially MentHlth, PhysHlth). These may affect certain models.

**Q:** Helpful transformations?  
**A:**  
- MentHlth, PhysHlth: **log(x+1)** or **sqrt**.  
- BMI: **log/sqrt/Box-Cox/Yeo-Johnson** or **quantile** transforms.  
Choice will be validated empirically.

---

### 📈 4. Feature Relationships & Patterns

**Q:** Categorical patterns vs `Diabetes_binary`?  
**A:**  
- **GenHlth:** Worse self-reported health -> higher diabetes prevalence.  
- **PhysActivity:** No activity 21.14% diabetic vs 11.61% if active.  
- **Smoker:** Slightly higher prevalence among smokers (16.29% vs 12.06%).

**Q:** Pairwise relationships/multicollinearity?  
**A:** Moderate, intuitive correlations (e.g., GenHlth with PhysHlth ~0.42; Income with Education ~0.45). None exceeded 0.7–0.8, so no severe multicollinearity concerns.

---

### 🧰 5. EDA Summary & Preprocessing Plan

**Top takeaways**
- **Severe class imbalance** (15.3% positive).  
- **Skewness/outliers** in MentHlth/PhysHlth (and some in BMI).  
- **GenHlth** strongly associated with diabetes.  
- **Lifestyle** factors (PhysActivity, Smoking) show signal.  
- **Correlations** moderate and manageable.

**Planned preprocessing**
- **Scaling** numeric features (BMI, MentHlth, PhysHlth; also Age/Income if used continuously).  
- **Encoding:** Binary already 0/1; Ordinal (GenHlth, Education, Age bands, Income bands) preserved as ordered integers.  
- **Exclusion:** None at this stage. Consider PCA if needed.

**Cleaned data shape:** **(229,474 rows, 22 columns)** after duplicate removal.

---

## ✅ Week 2: Feature Engineering & Preprocessing

### 🏷️ 1. Encoding

- **Binary columns** (already 0/1):  
  `['HighBP','HighChol','CholCheck','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','DiffWalk','Sex','Diabetes_binary']` -> **kept as is**.
- **Ordinal columns:**  
  - `GenHlth`: 1=Excellent … 5=Poor (**preserve order**).  
  - `Education`: 1=None … 6=College Grad (**preserve order**).  
- **Nominal columns:** None identified -> **no one-hot** needed (if introduced later, we will one-hot to avoid false ordering).
- **Scaling:** Applied to continuous features (BMI, MentHlth, PhysHlth, etc.) with **StandardScaler** (fit on train only).

### ✨ 2. Feature Creation

- **`BMI_category`** (CDC cutoffs):  
  - Underweight < 18.5; Normal 18.5–24.9; Overweight 25–29.9; Obese >= 30.  

  | Category    | Count  | Percent |
  |-------------|--------|---------|
  | Overweight  | 93,749 | ~37%    |
  | Obese       | 87,851 | ~35%    |
  | Normal      | 68,953 | ~27%    |
  | Underweight | 3,127  | ~1%     |

  *Insight:* ~72% Overweight/Obese -> strong nonlinearity; the category feature can help.

- **`TotalHealthDays` = `PhysHlth` + `MentHlth`**  
  Rationale: captures total “unhealthy days” (physical + mental) in past 30 days; reflects overall health burden.

### ✂️ 3. Data Splitting

- **80/20 split**, **stratified** on `Diabetes_binary`.  
- Shapes (from run):  
  - `X_train`: **(202,944, 23)**  
  - `X_test`: **(50,736, 23)**  
  - `y_train`: **(202,944)**  
  - `y_test`: **(50,736)**

**Why split before SMOTE/scaling?** To avoid **data leakage**. Fit resampling/scalers on **train only**.

### ⚖️ 4. Imbalance Handling & Final Preprocessing

- **SMOTE** on training only:  
  - Before: `Counter({0: 174,667, 1: 28,277})`  
  - After:  `Counter({0: 174,667, 1: 174,667})`  
  - `X_train_resampled`: **(349,334, 25)**; `y_train_resampled`: **(349,334)**

- **Scaling:** StandardScaler **fit on train**, then transform train/test.  
  - `X_train_scaled`: **(349,334, 25)**  
  - `X_test_scaled`: **(50,736, 25)**

---

## ✅ Week 3: Model Development & Experimentation

### 🤖 1. Baselines (test set)

| Model               | Accuracy | Precision | Recall | F1    | AUC   |
|---------------------|---------:|----------:|------:|------:|------:|
| Naive Bayes         | 0.666    | 0.259     | 0.748 | 0.384 | 0.700 |
| Decision Tree       | 0.727    | 0.238     | 0.437 | 0.308 | 0.605 |
| Logistic Regression | 0.666    | 0.259     | 0.748 | 0.384 | 0.700 |

**Notes:** NB shows strong **recall** (catching positives), LR offers a better balance, DT underperforms on AUC.

### 🧪 2. Extended models & comparison (MLflow)

| Model               | Accuracy | Precision | Recall | F1    | AUC   |
|---------------------|---------:|----------:|------:|------:|------:|
| Naive Bayes         | 0.644    | 0.256     | 0.813 | 0.389 | 0.715 |
| Decision Tree       | 0.750    | 0.254     | 0.412 | 0.314 | 0.608 |
| Logistic Regression | 0.727    | 0.297     | 0.701 | 0.417 | 0.716 |
| Random Forest       | **0.780**| **0.308** | 0.466 | 0.371 | 0.648 |
| Gradient Boosting   | 0.720    | 0.295     | 0.730 | **0.420** | **0.724** |
| k-Nearest Neighbors | 0.720    | 0.268     | 0.583 | 0.367 | 0.663 |

**Takeaways:**  
- **Recall priority** -> NB & GB strongest at catching positives.  
- **Overall balance** -> GB has best F1/AUC; RF best Accuracy/Precision.

### 📈 3. Experiment tracking (MLflow)
Tracked: algorithm, key hyperparameters, train/test metrics (Accuracy, Precision, Recall, F1, AUC), timestamp, data version.  
Benefit: **side-by-side comparability**, reproducibility, and quick identification of promising settings.

### 🕵️ 4. Error Analysis
Confusion-matrix focus: minimize **FN** (missed diabetics).  
Example (Logistic Regression): more **FP (11,728)** than **FN (2,116)**. Threshold tuning and class weights can rebalance.

### 📝 5. Model Selection & Insights
- **Current best direction:** **Gradient Boosting** (best F1/AUC) with tuning; LR and RF are competitive.  
- **Top insights:**  
  1) **Class imbalance** drives metric choice; SMOTE helped learning.  
  2) **Precision–Recall trade-off** dictates thresholding for healthcare.  
  3) **Ensembles** (GB, RF) generally dominate single trees.  
  4) **Feature engineering** (BMI_category, TotalHealthDays) adds interpretability and signal.  
  5) **Hyperparameter tuning** likely to improve GB/LR/RF further.

**Non-technical summary:** The model screens for diabetes risk well, emphasizing **catching true cases** (recall). Some healthy people may be flagged (precision trade-off), which is manageable via inexpensive follow-up tests and threshold adjustments.

---

## ✅ Week 4: Model Tuning & Finalization

### 🛠️ 1) Hyperparameter Tuning
**Method:** Stratified 5-fold CV with `RandomizedSearchCV`.  
**Models tuned:** Gradient Boosting (GB), Random Forest (RF), Logistic Regression (LR).  
**Scoring:** F1 for the classification track; for the probability track we compared calibrated models on Brier/LogLoss (no SMOTE).

**Search spaces (high level):**
- **GB:** `n_estimators` [100–300], `learning_rate` [0.01–0.2], `max_depth` [2–4], `subsample` [0.7–1.0]  
- **RF:** `n_estimators` [200–600], `max_depth` [None, 6–14], `max_features` [sqrt, log2, 0.5, 0.8]  
- **LR:** `C` [1e-3–1e2], `penalty` [l2], `class_weight` [None, balanced]

**Outcome:** RF tended to give the best F1 at a cost-aware threshold (e.g., F1 ~ **0.3466** at R=5), while **calibrated GB** produced the best probability quality for decisioning (see below). Full CV params/tables are in the notebook.

---

### 🔄 2) Cross-Validation
Performed **Stratified 5-fold CV** during tuning; selections were based on mean F1 (classification track) and on calibration losses (probability track). Variance across folds was modest; selected settings generalized to the held-out test set. (See notebook for fold-by-fold metrics and chosen hyperparameters.)

---

### 🏆 3) Final Model Selection
**Chosen for risk scoring & decisioning:** **Gradient Boosting with sigmoid calibration** (best probability calibration + ranking).  
**Test (threshold-free) quality:** Brier **0.0481**, LogLoss **0.1770**, ROC-AUC **0.8651**, PR-AUC **0.4778**.

**Decision setting (cost-aware, R = 5):** optimal threshold `t* = 0.2667`  
- Confusion matrix (n = 50,736): **TN 22,750 · FP 20,917 · FN 781 · TP 6,288**  
- Metrics @ `t*`: **Precision 0.2311 · Recall 0.8895 · F1 0.3669**  
- Expected cost: `5*FN + 1*FP = 24,822` -> **$489.10 per 1,000** cases

---

### 📏 4) Probability Calibration (risk scoring)
**Calibration method:** `CalibratedClassifierCV(method="sigmoid")` on GB; no SMOTE (used class weights when applicable).  
**Calibration quality (GB, test):** Brier **0.0481**, LogLoss **0.1770**, ROC-AUC **0.8651**, PR-AUC **0.4778**.

**Cost-aware thresholds (GB, calibrated):**

| Cost ratio (C_FN:C_FP) | Optimal t* | Recall @ t* | Expected cost per 1,000 |
|---|---:|---:|---:|
| 2:1  | 0.4343 | 0.8008 | $290.96 |
| 5:1  | 0.2667 | 0.8895 | $489.10 |
| 10:1 | 0.1737 | 0.9427 | $805.07 |

**Risk bands (GB, calibrated):**
- Low [0, 0.05): **87.80%**  
- Medium [0.05, 0.15): **7.28%**  
- High [0.15, 1.0]: **4.92%**

**Top-k triage (GB, calibrated):**
- Top 1% (~535): Precision@1 = **0.2299**, Recall@1 = **0.1070**  
- Top 5% (~2,676): Precision@5 = **0.1510**, Recall@5 = **0.3523**  
- Top 10% (~5,353): Precision@10 = **0.1145**, Recall@10 = **0.5352**

**Interpretation:** As FN cost increases, the optimal threshold decreases — recall rises, precision falls, and total cost eventually increases due to many FPs. Bands and top-k views let you operate under staffing constraints without changing the model.

---

### 📊 5) Feature Importance & Interpretation
Across RF/GB the top drivers were consistent: **TotalHealthDays**, **BMI_category_Obese**, **GenHlth**, **HighBP**, **Age**. These align with domain intuition (overall health burden, obesity, self-reported health, hypertension, and age are materially associated with diabetes risk).

---

**Flagging policy (operational):**  
- **High risk (p >= 0.15):** “Likely diabetic” -> immediate follow-up (A1C/clinician).  
- **Medium risk (0.05 <= p < 0.15):** “Elevated risk” -> outreach + guidance; retest in 3–6 months.  
- **Low risk (p < 0.05):** “Unlikely now” -> routine monitoring. 
 
- If one cutoff is required for R = 5, use **`t* = 0.2667`** (recall ~ **0.8895**). With limited capacity, triage by **Top 5–10%** instead of a fixed threshold.
