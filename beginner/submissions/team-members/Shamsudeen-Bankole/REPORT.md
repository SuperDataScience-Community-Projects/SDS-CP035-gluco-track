## ✅ Week 1: Exploratory Data Analysis (EDA)

---
### 📦 1. Data Integrity & Structure

Q: Are there any missing, duplicate, or incorrectly formatted entries in the dataset?  
A: No missing values were found in the dataset, as 253680 instances are complete.
   24206 rows (corresponding to 9.5% of the dataset) were identified as duplicates and would be further assessed for removal to ensure each observation is 
   unique and does not bias the model.
   No formatting issues were observed; however, all features were stored as float64, while many of them should be integers or categories.
    
Q: Are all data types appropriate (e.g., numeric, categorical)?  
A: No, not all data types are appropriate in the raw dataset. Therefore, the data types needed explicit correction based on careful observation.

Q: Did you detect any constant, near-constant, or irrelevant features?  
A: No constant features (i.e., columns with only one unique value) were detected. Every feature had at least two unique values, indicating variability in responses.
There are a few features that showed near-constant behaviour (i.e. one category accounted for the vast majority of entries). For example, CholCheck               96.27%, Stroke 95.94%, AnyHealthcare 95.11%. These features are not removed immediately but flagged for possible exclusion or dimensionality reduction if 
they do not significantly contribute during model evaluation.
Initially, there are no clear indications of irrelevant features. All retained features may have potential predictive power for diabetes classification      and should be evaluated during model training for importance.

---
### 🎯 2. Target Variable Assessment 

Q: What is the distribution of `Diabetes_binary`?  
A: The distribution of target_class is highly imbalanced, with class 0 (no diabetes) accounting for approximately 86% of the data (218334 instances), class 1 (prediabetes or diabetes) accounting for 14.0% (35,346 instances). This suggests a strong class skew that may require addressing in modelling.

Q: Is there a class imbalance? If so, how significant is it?  
A: Yes, there is a significant class imbalance. Class 0 dominates the dataset with approximately 86.07% (218334 out of 253,680 instances), while class 1 accounts for 13.93% (35,346 instances). The disparity between the majority and minority classes is substantial and may adversely affect model performance (due to both data and algorithmic bias) if not addressed.

Q: How might this imbalance influence your choice of evaluation metrics or model strategy?  
A: The imbalance makes it important to use balanced evaluation metrics (such as precision, recall, F1-score, and area under the precision-recall curve) and model strategies that ensure fair learning across all classes. Ignoring this can lead to a model that performs poorly on critical minority outcomes, which is unacceptable in sensitive domains like healthcare.
We may have to consider resampling methods such as SMOTE, ADASYN, oversampling, undersampling or class weighting.
We may have to consider threshold tuning to optimise recall or the F1 score, depending on the clinical priority

---
### 📊 3. Feature Distribution & Quality

Q: Which numerical features are skewed or contain outliers?  
A: MentHlth has strong positive skew (~2.72) with many zeros: about 14% flagged as outliers.
- PhysHlth is positively skewed (~2.21) with about 16% flagged as outliers
- BMI: Right skewed (2.12) with a long tail: 3.9% flagged as outlier, and the maximum observed at 98.
   
Q: Did any features contain unrealistic or problematic values?
A: unrealistic or problematic values were found.
    
Q: What transformation methods (if any) might improve these feature distributions?  
A: BMI: Winsorizing or capping a high percentile or Use RobustScaler or Yeo-Johnson 
- MentHlth and PhysHlth: Yeo-Johnson or Square-root transformation for variance stabilisation
 
---
### 📈 4. Feature Relationships & Patterns

Q: Which categorical features (e.g., `GenHealth`, `PhysicalActivity`, `Smoking`) show visible patterns in relation to `Diabetes_binary`?  
A: Based on a careful analysis of the dataset, several categorical features exhibit visible and statistically significant patterns in relation to Diabetes_binary:
- HighBP and HighChol: Elevated blood pressure and cholesterol levels are strongly associated with diabetes prevalence.
- Smoker and HvyAlcoholConsump: Lifestyle factors such as smoking and heavy alcohol consumption show significant correlations with diabetes status.
- Stroke and HeartDiseaseorAttack: A history of cardiovascular events is a strong predictor of diabetes.
- PhysActivity, Fruits, and Veggies: Health-promoting behaviors—including regular physical activity and consumption of fruits and vegetables—are significantly linked to lower diabetes risk.
- AnyHealthcare and NoDocbcCost: Access to healthcare services and affordability of medical consultations influence the likelihood of diabetes diagnosis.
- DiffWalk: Mobility limitations are predictive of diabetes outcomes, possibly reflecting broader health impairments.
- Sex: Gender shows a statistically significant association with diabetes status.
- CholCheck: Individuals with recent cholesterol checks show a significantly higher diabetes diagnosis rate than those without.
Additionally, several ordinal features demonstrate monotonic relationships with Diabetes_binary based on Spearman correlation:
- GenHlth: Moderate positive correlation (r = 0.288, p < 0.0001); individuals reporting poorer general health are more likely to have diabetes.
- Age: Weak positive correlation (r = 0.178, p < 0.0001); older individuals tend to have higher diabetes prevalence.
- Education: Weak negative correlation (r = –0.120, p < 0.0001); higher educational attainment is associated with lower diabetes risk.
- Income: Weak-to-moderate negative correlation (r = –0.163, p < 0.0001); individuals with higher income levels are less likely to have diabetes.


Q: Are there any strong pairwise relationships or multicollinearity between features?  
A: No strong pairwise correlations were observed among the numeric features. The relationships between predictors are generally weak, and there is no evidence of multicollinearity. This suggests that the features contribute independently to the outcome and are suitable for inclusion in multivariate models without redundancy concerns.

Q: What trends or correlations stood out during your analysis?
A: Several key patterns emerged
- Health conditions such as high blood pressure, high cholesterol, stroke, and heart disease were strongly associated with diabetes.
- Lifestyle factors—including smoking, heavy alcohol use, low physical activity, and poor diet—showed significant links to diabetes prevalence.
- Socioeconomic indicators like lower income and education levels were negatively correlated with diabetes risk.
- General health ratings had the strongest monotonic relationship (ρ = 0.288), indicating that self-reported poor health is a key predictor.
- Among numeric features, PhysHlth and MentHlth showed the highest pairwise correlation (ρ ≈ 0.35), though still weak.
- BMI had the highest individual correlation with the outcome among numeric variables (ρ = 0.22), suggesting a modest but notable association.
- No strong pairwise correlations or multicollinearity were detected, indicating that predictors contribute independently to the model.

---
### 🧰 5. EDA Summary & Preprocessing Plan

Q: What are your 3–5 biggest takeaways from EDA?  
A: **Strong categorical associations with diabetes**
- Features like HighBP, HighChol, Smoker, Stroke, and HeartDiseaseorAttack show clear, statistically significant links to diabetes status.
**Lifestyle and access matter**
- Low physical activity, poor diet, and limited healthcare access (NoDocbcCost, AnyHealthcare) are associated with higher diabetes prevalence.
**Self-reported health is a key signal**
- GenHlth shows the strongest monotonic correlation with diabetes (ρ = 0.288), followed by BMI (ρ = 0.22), indicating that subjective health ratings and body mass index are meaningful predictors.
**Data quality considerations**
- The target class (Diabetes_binary) is imbalanced, requiring careful handling to avoid biased model performance.
- Features with near-constant values may be dropped to reduce noise and improve model efficiency.
- Outliers should be addressed through appropriate scaling or transformation to preserve model stability.
- No multicollinearity detected
- Weak pairwise correlations across features suggest low redundancy, supporting the use of all predictors in multivariate models.

Q: Which features will you scale, encode, or exclude in preprocessing?  
A: - BMI, MentHlth, PhysHlth will be scaled to handle outliers and normalize their distributions for model stability
- Ordinal features such as Income, Age, Education, GenHlth, MentHlth, PhysHlth will be encoded using ordinal encoding to preserve their inherent order and semantic meaning.
- Features with near-constant values or low variance features such as CholCheck, Stroke and AnyHealthcare, may be excluded to reduce noise and improve model efficiency

Q: What does your cleaned dataset look like (rows, columns, shape)?  
A:  After removing duplicate entries, the cleaned dataset will contain 229,474 rows and 22 columns, resulting in a shape of (229474, 22).


## ✅ Week 2: Feature Engineering & Preprocessing

---
### 🏷️ 1. Feature Encoding

Q: Identify the binary (`0` or `1`) categorical features and apply a simple mapping or encoder. Which features did you encode?  
A:  The binary features were retained in their original 0/1 format, as no additional encoding was necessary.

Q: The `GenHealth` and `Education` features are ordinal. Apply a custom mapping that preserves their inherent order and justify the order you chose.  
A:  Ordinal features were encoded using ordinal encoding, with values starting from 0 and increasing to the maximum possible rank.

Q: For any remaining nominal categorical features, apply one-hot encoding. Why is this method more suitable for nominal data than a simple integer label?  
A: One-hot encoding is better for nominal data because it avoids implying any order or hierarchy between categories. Integer labels suggest false relationships  while one-hot treats each category as equally distinct, preserving semantic neutrality and improving model interpretability.


---
### ✨ 2. Feature Creation

Q: Create a new feature for BMI categories (e.g., Underweight, Normal, Overweight, Obese) from the `BMI` column. Display the value counts for your new categories.  
A:  Done

Q: Create a new feature named `TotalHealthDays` by combining `PhysHlth` and `MentHlth`. What is the rationale behind creating this feature?  
A:  

---

### ✂️ 3. Data Splitting

Q: Split your dataset into training and testing sets (an 80/20 split is recommended). Use stratification on the `Diabetes_binary` target variable.  
A:  

Q: Why is it critical to split the data *before* applying techniques like SMOTE or scaling?  
A:  It is critical to split data before scaling to avoid information leakage

Q: Show the shape of your `X_train`, `X_test`, `y_train`, and `y_test` arrays to confirm the split.  
A:  Done

---

### ⚖️ 4. Imbalance Handling & Final Preprocessing

Q: Apply the SMOTE technique to address class imbalance. Importantly, apply it *only* to the training data. Show the class distribution of the training target variable before and after.  
A:  

Q: Normalize the numerical features using `StandardScaler`. Fit the scaler *only* on the training data, then transform both the training and testing data. Why must you not fit the scaler on the test data?  
A: Data leakage: Fitting on the test set exposes your model to information it shouldn't have during training. This can lead to overly optimistic performance estimates.


Q: Display the shape of your final, preprocessed training features (`X_train_processed`) and testing features (`X_test_processed`).  
A:

✅ Week 3: Model Development & Experimentation
🤖 1. Baseline Model Training
Q: Which baseline models did you choose for this classification task, and why?
A:I chose Logistic Regression, Decision Tree, and Naive Bayes as baseline models for their simplicity, interpretability, and diverse learning assumptions. This enables meaningful comparison across linear, non-linear, and probabilistic approaches, particularly under class imbalance strategies such as SMOTE and class weighting.

Q: How did you implement and evaluate Logistic Regression, Decision Tree, and Naive Bayes models?
A: I implemented Logistic Regression, Decision Tree, and Naive Bayes using scikit-learn, training each model on a stratified training set with class imbalance handled via either SMOTE or class weighting. Each model was evaluated on a separate validation set using four key metrics: accuracy, precision, recall, and F1-score.
To support error analysis, I generated and logged confusion matrices for each run, highlighting false positives and false negatives. All experiments were tracked using MLflow, allowing for a side-by-side comparison of model performance and imbalance strategies. This setup provided a reproducible framework for selecting the most effective baseline under different conditions

Q: What challenges did you encounter during model training, and how did you address them?
A:During model training, the main challenge was handling class imbalance. I addressed this by comparing two strategies: SMOTE-based oversampling and class-weighted loss adjustment. Each was evaluated using validation metrics and confusion matrices to assess its impact on precision, recall, and error patterns. Stratified data splitting ensured fair evaluation without leakage.

📈 2. Experiment Tracking
Q: How did you use MLflow (or another tool) to track your experiments?
A: I used MLflow to systematically track all baseline model experiments. For each run, I logged key parameters (e.g., model name, imbalance strategy), performance metrics (accuracy, precision, recall, F1-score), and confusion matrices as visual artifacts. I tagged each run with the imbalance-handling method (SMOTE or class weighting) to enable clear comparison in the MLflow UI. This setup allowed me to monitor results across models and strategies, identify trade-offs, and maintain a reproducible record of the entire evaluation process.

Q: What key parameters and metrics did you log for each model run?
A: For each model run, I logged the model type, imbalance strategy, and key metrics: accuracy, precision, recall, and F1-score. I also saved confusion matrix plots and the trained model as artifacts using MLflow for easy comparison and reproducibility


Q: How did experiment tracking help you compare and select the best model?
A: Experiment tracking with MLflow allowed me to log and compare each model’s performance under different imbalance strategies. By recording metrics like accuracy, precision, recall, and F1-score, along with confusion matrices and strategy tags, I could easily identify which models performed best for specific goals—such as maximizing recall or balancing precision. This made model selection transparent, reproducible, and aligned with the task’s priorities.

🧮 3. Model Evaluation
Q: Which evaluation metrics did you use to assess model performance, and why are they appropriate for this problem?
A: I used accuracy, precision, recall, and F1-score to assess model performance. These metrics are well-suited for imbalanced classification tasks, helping capture both overall correctness and the trade-off between false positives and false negatives


Q: How did you interpret the accuracy, precision, recall, and F1-score for your models?
A: I interpreted the metrics to understand each model’s strengths and trade-offs under class imbalance. Accuracy showed overall correctness but was less informative alone. Precision revealed how reliably the model predicted positives, while recall highlighted its ability to detect actual positives—crucial for minimizing false negatives. F1-score balanced both, helping identify models that performed consistently across precision and recall. This guided selection based on whether the task prioritized sensitivity or specificity


Q: Did you observe any trade-offs between different metrics? How did you decide which metric(s) to prioritize?
A: Yes, I observed trade-offs between precision and recall. Models using SMOTE achieved higher recall, which is valuable for identifying individuals at risk of diabetes, but at the cost of lower precision. Class-weighted models offered better precision but missed more true cases. Given the health context, I prioritized recall to reduce false negatives and ensure potential cases weren’t overlooked.


🕵️ 4. Error Analysis
Q: How did you use confusion matrices to analyze model errors?
A: I used confusion matrices to identify and compare error patterns across models and imbalance strategies. They revealed how often each model produced false positives (predicting diabetes when not present) and false negatives (missing actual cases). This helped assess the trade-off between precision and recall, especially under SMOTE and class weighting. By visualising these matrices, I could prioritise models that minimised false negatives—critical for early detection in diabetes risk prediction.


Q: What types of misclassifications were most common, and what might explain them?
A: False positives were most common with SMOTE, likely due to oversampling making the model more sensitive. Class-weighted models showed more false negatives, missing actual cases of diabetes. These patterns reflect the trade-off between precision and recall in imbalanced health data.

Q: How did your error analysis inform your next steps in model improvement?
A: Error analysis showed that SMOTE increased recall but led to more false positives, while class weighting reduced false alarms but missed more true cases. These insights guided my next steps: prioritising recall for early diabetes risk detection and considering threshold tuning or ensemble methods to better balance sensitivity and specificity.

📝 5. Model Selection & Insights
Q: Based on your experiments, which model performed best and why?
A: Based on my experiments, Logistic Regression performed best overall. It consistently achieved a strong balance between recall and precision, especially under both SMOTE and class-weighted strategies. While Naive Bayes had a higher recall with SMOTE, its precision was too low for reliable predictions. Decision Tree showed inconsistent performance, with low F1-scores across both strategies. Logistic Regression offered the most stable and interpretable results, making it the most suitable baseline for early diabetes risk detection


Q: What are your top 3–5 insights from model development and experimentation?
A: - Class imbalance significantly affects model behavior — SMOTE improves recall but increases false positives, while class weighting offers better precision with lower sensitivity.
- Logistic Regression was the most balanced performer, consistently achieving strong F1-scores and stable precision-recall trade-offs across both imbalance strategies.
- Confusion matrix analysis was essential — it revealed that minimizing false negatives is critical in health contexts, guiding the prioritization of recall over precision.
- Experiment tracking with MLflow streamlined model comparison, making it easy to evaluate performance across strategies and maintain reproducibility.
- Metric trade-offs must align with domain goals — in diabetes risk prediction, missing true cases (false negatives) is more costly than flagging false ones, shaping model selection accordingly.


Q: How would you communicate your model’s strengths and limitations to a non-technical stakeholder?
A: The model is good at spotting people who may be at risk of diabetes — it catches most cases early, which is important for prevention. That’s its biggest strength. However, it sometimes flags people who aren’t actually at risk, which could lead to unnecessary follow-ups. We’ve tuned it to prioritize early detection, because missing a real case would be more harmful than a false alarm.”


✅ Week 4: Model Tuning & Finalization
🛠️ 1. Hyperparameter Tuning
Q: Which hyperparameters did you tune for your models, and what methods (e.g., grid search, random search) did you use?
A:

Q: How did you select the range or values for each hyperparameter?
A:

Q: What impact did hyperparameter tuning have on your model’s performance?
A:

🔄 2. Cross-Validation
Q: How did you use cross-validation to assess model stability and generalization?
A:

Q: What were the results of your cross-validation, and did you observe any variance across folds?
A:

Q: Why is cross-validation important in this context?
A:

🏆 3. Final Model Selection
Q: How did you choose your final model after tuning and validation?
A:

Q: Did you retrain your final model on the full training set before evaluating on the test set? Why or why not?
A:

Q: What were the final test set results, and how do they compare to your validation results?
A:

📊 4. Feature Importance & Interpretation
Q: How did you assess feature importance for your final model?
A:

Q: Which features were most influential in predicting diabetes risk, and do these results align with domain knowledge?
A:

Q: How would you explain your model’s decision process to a non-technical audience?
A:


