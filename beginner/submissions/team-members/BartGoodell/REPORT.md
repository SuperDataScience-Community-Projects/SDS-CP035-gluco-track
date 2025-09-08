# 🟢 GlucoTrack – Beginner Track

## ✅ Week 1: Exploratory Data Analysis (EDA)

---

### 📦 1. Data Integrity & Structure

**Q:** Are there any missing, duplicate, or incorrectly formatted entries in the dataset?  
**A:** No missing values were found across the 253,680 instances. A total of **24,206 rows (~9.5%)** were flagged as duplicates and removed to avoid bias from repeated observations. No formatting issues were present. However, all features were stored as `float64`; several should instead be integers or categoricals.

**Q:** Are all data types appropriate (e.g., numeric, categorical)?  
**A:** Not fully. While numeric measures (e.g., BMI, MentHlth, PhysHlth) were correctly stored as floats, many **binary** and **ordinal** variables were misclassified as continuous. We corrected these using domain-informed mappings.

**Feature types overview**

| Type                 | Columns                                                                                                                                                                                                                                   |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Binary categorical   | Outcome, Sex, HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk                                                                   |
| Ordinal / Categorical| GenHlth, Age, Education, Income                                                                                                                                                                                                           |
| Numeric              | BMI, MentHlth, PhysHlth                                                                                                                                                                                                                   |

**Q:** Did you detect any constant, near-constant, or irrelevant features?  
**A:** No constant features. At this stage nothing was dropped as irrelevant; all were retained for modeling and later importance checks.

---

### 🎯 2. Target Variable Assessment

**Q:** What is the distribution of `Diabetes_binary`?  
**A:** After removing duplicates, the target remains **imbalanced**: **84.7% = 0 (no diabetes)** vs **15.3% = 1 (diabetes)**.

**Q:** How might this imbalance influence metrics/strategy?  
**A:** Accuracy is misleading. We’ll emphasize **Recall**, **Precision**, **F1**, **ROC-AUC**, and **PR-AUC**, and use strategies like **class weights** and **resampling (SMOTE)**.

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
- **GenHlth**: Worse self-reported health → higher diabetes prevalence.  
- **PhysActivity**: No activity 21.14% diabetic vs 11.61% if active.  
- **Smoker**: Slightly higher prevalence among smokers (16.29% vs 12.06%).

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
- **Encoding**: Binary already 0/1; Ordinal (GenHlth, Education, Age bands, Income bands) preserved as ordered integers.  
- **Exclusion**: None at this stage. Consider PCA if needed.

**Cleaned data shape**: **(229,474 rows, 22 columns)** after duplicate removal.

---

## ✅ Week 2: Feature Engineering & Preprocessing

### 🏷️ 1. Encoding

- **Binary columns** (already 0/1):  
  `['HighBP','HighChol','CholCheck','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','DiffWalk','Sex','Diabetes_binary']` → **kept as is**.
- **Ordinal columns**:  
  - `GenHlth`: 1=Excellent … 5=Poor (**preserve order**).  
  - `Education`: 1=None … 6=College Grad (**preserve order**).  
- **Nominal columns**: None identified → **no one-hot** needed (if introduced later, we’ll one-hot to avoid false ordering).
- **Scaling**: Applied to continuous features (BMI, MentHlth, PhysHlth, etc.) with **StandardScaler** (fit on train only).

### ✨ 2. Feature Creation

- **`BMI_category`** (CDC cutoffs):  
  - Underweight < 18.5; Normal 18.5–24.9; Overweight 25–29.9; Obese ≥ 30.  

  | Category    | Count  | Percent |
  |-------------|--------|---------|
  | Overweight  | 93,749 | ~37%    |
  | Obese       | 87,851 | ~35%    |
  | Normal      | 68,953 | ~27%    |
  | Underweight | 3,127  | ~1%     |

  *Insight:* ~72% Overweight/Obese → strong nonlinearity; the category feature can help.

- **`TotalHealthDays` = `PhysH**

### ✂️ 3. Data Splitting

- **80/20 split**, **stratified** on `Diabetes_binary`.  
- Shapes (example from run):  
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

- **Scaling**: StandardScaler **fit on train**, then transform train/test.  
  - `X_train_scaled`: **(349,334, 25)**  
  - `X_test_scaled`: **(91,258, 25)**

---

## ✅ Week 3: Model Development & Experimentation

### 🤖 1. Baselines

Initial baselines (test set):

| Model               | Accuracy | Precision | Recall  | F1     | AUC    |
|---------------------|---------:|----------:|--------:|-------:|-------:|
| Naive Bayes         | 0.666    | 0.259     | 0.748   | 0.384  | 0.700  |
| Decision Tree       | 0.727    | 0.238     | 0.437   | 0.308  | 0.605  |
| Logistic Regression | 0.666    | 0.259     | 0.748   | 0.384  | 0.700  |

**Notes:** NB shows strong **recall** (catching positives), LR offers a better balance, DT underperforms on AUC.

### 🧪 2. Extended models & comparison

Logged runs (via MLflow):

| Model                | Accuracy | Precision | Recall  | F1     | AUC    |
|----------------------|---------:|----------:|--------:|-------:|-------:|
| Naive Bayes          | 0.644    | 0.256     | 0.813   | 0.389  | 0.715  |
| Decision Tree        | 0.750    | 0.254     | 0.412   | 0.314  | 0.608  |
| Logistic Regression  | 0.727    | 0.297     | 0.701   | 0.417  | 0.716  |
| Random Forest        | **0.780**| **0.308** | 0.466   | 0.371  | 0.648  |
| Gradient Boosting    | 0.720    | 0.295     | 0.730   | **0.420** | **0.724** |
| k-Nearest Neighbors  | 0.720    | 0.268     | 0.583   | 0.367  | 0.663  |

**Takeaways:**  
- **Recall priority** → NB & GB strongest at catching positives;  
- **Overall balance** → **Gradient Boosting** has best F1/AUC; **Random Forest** best Accuracy/Precision.

### 📈 3. Experiment tracking (MLflow)

- Tracked: algorithm, key hyperparameters, train/test metrics (Accuracy, Precision, Recall, F1, AUC), timestamp, data version.  
- Benefit: **side-by-side comparability**, reproducibility, and quick identification of promising settings.

### 🕵️ 4. Error Analysis

- **Confusion matrix** focus: minimize **FN** (missed diabetics).  
- Example (Logistic Regression): more **FP (11,728)** than **FN (2,116)**. Threshold tuning and class weights can rebalance.


### 📝 5. Model Selection & Insights

- **Current best direction:** **Gradient Boosting** (best F1/AUC) with tuning; LR and RF are competitive.  
- **Top insights:**  
  1) **Class imbalance drives** metric choice; SMOTE helped learning.  
  2) **Precision–Recall trade-off** dictates thresholding for healthcare.  
  3) **Ensembles** (GB, RF) generally dominate single trees.  
  4) **Feature engineering** (BMI_category, TotalHealthDays) adds interpretability and signal.  
  5) **Hyperparameter tuning** likely to improve GB/LR/RF further.

**Non-technical summary:** The model screens for diabetes risk well, emphasizing **catching true cases** (recall). Some healthy people may be flagged (precision trade-off), which is manageable via inexpensive follow-up tests and threshold adjustments.

---

## ✅ Week 4: Tuning & Finalization (Preview)

- **Tuning plan:** Grid/Random search for GB (n_estimators, learning_rate, max_depth, subsample), LR (C, penalty, class_weight), RF (n_estimators, max_depth, max_features).  
- **Cross-validation:** Stratified K-fold to verify stability across folds; watch variance.  
- **Feature importance:** GB feature importances/SHAP for interpretability; confirm alignment with domain knowledge (GenHlth, BMI, HighBP/Chol, activity).

