# 🟢 GlucoTrack – Beginner Track

## ✅ Week 1: Exploratory Data Analysis (EDA)

---

### 📦 1. Data Integrity & Structure

Q: Are there any missing, duplicate, or incorrectly formatted entries in the dataset?  
A:  

Q: Are all data types appropriate (e.g., numeric, categorical)?  
A:  

Q: Did you detect any constant, near-constant, or irrelevant features?  
A:  

---

### 🎯 2. Target Variable Assessment

Q: What is the distribution of `Diabetes_binary`?  
A:  

Q: Is there a class imbalance? If so, how significant is it?  
A:  

Q: How might this imbalance influence your choice of evaluation metrics or model strategy?  
A:  

---

### 📊 3. Feature Distribution & Quality

Q: Which numerical features are skewed or contain outliers?  
A:  

Q: Did any features contain unrealistic or problematic values?  
A:  

Q: What transformation methods (if any) might improve these feature distributions?  
A:  

---

### 📈 4. Feature Relationships & Patterns

Q: Which categorical features (e.g., `GenHealth`, `PhysicalActivity`, `Smoking`) show visible patterns in relation to `Diabetes_binary`?  
A:  

Q: Are there any strong pairwise relationships or multicollinearity between features?  
A:  

Q: What trends or correlations stood out during your analysis?  
A:  

---

### 🧰 5. EDA Summary & Preprocessing Plan

Q: What are your 3–5 biggest takeaways from EDA?  
A:  

Q: Which features will you scale, encode, or exclude in preprocessing?  
A:  

Q: What does your cleaned dataset look like (rows, columns, shape)?  
A:  

---

## ✅ Week 2: Feature Engineering & Preprocessing

---

### 🏷️ 1. Feature Encoding

Q: Identify the binary (`0` or `1`) categorical features and apply a simple mapping or encoder. Which features did you encode?  
A:  

Q: The `GenHealth` and `Education` features are ordinal. Apply a custom mapping that preserves their inherent order and justify the order you chose.  
A:  

Q: For any remaining nominal categorical features, apply one-hot encoding. Why is this method more suitable for nominal data than a simple integer label?  
A:  

---

### ✨ 2. Feature Creation

Q: Create a new feature for BMI categories (e.g., Underweight, Normal, Overweight, Obese) from the `BMI` column. Display the value counts for your new categories.  
A:  

Q: Create a new feature named `TotalHealthDays` by combining `PhysHlth` and `MentHlth`. What is the rationale behind creating this feature?  
A:  

---

### ✂️ 3. Data Splitting

Q: Split your dataset into training and testing sets (an 80/20 split is recommended). Use stratification on the `Diabetes_binary` target variable.  
A:  

Q: Why is it critical to split the data *before* applying techniques like SMOTE or scaling?  
A:  

Q: Show the shape of your `X_train`, `X_test`, `y_train`, and `y_test` arrays to confirm the split.  
A:  

---

### ⚖️ 4. Imbalance Handling & Final Preprocessing

Q: Apply the SMOTE technique to address class imbalance. Importantly, apply it *only* to the training data. Show the class distribution of the training target variable before and after.  
A:  

Q: Normalize the numerical features using `StandardScaler`. Fit the scaler *only* on the training data, then transform both the training and testing data. Why must you not fit the scaler on the test data?  
A:  

Q: Display the shape of your final, preprocessed training features (`X_train_processed`) and testing features (`X_test_processed`).  
A:

---

## ✅ Week 3: Model Development & Experimentation

---

### 🤖 1. Baseline Model Training

Q: Which baseline models did you choose for this classification task, and why?  
A:  

Q: How did you implement and evaluate Logistic Regression, Decision Tree, and Naive Bayes models?  
A:  

Q: What challenges did you encounter during model training, and how did you address them?  
A:  

---

### 📈 2. Experiment Tracking

Q: How did you use MLflow (or another tool) to track your experiments?  
A:  

Q: What key parameters and metrics did you log for each model run?  
A:  

Q: How did experiment tracking help you compare and select the best model?  
A:  

---

### 🧮 3. Model Evaluation

Q: Which evaluation metrics did you use to assess model performance, and why are they appropriate for this problem?  
A:  

Q: How did you interpret the accuracy, precision, recall, and F1-score for your models?  
A:  

Q: Did you observe any trade-offs between different metrics? How did you decide which metric(s) to prioritize?  
A:  

---

### 🕵️ 4. Error Analysis

Q: How did you use confusion matrices to analyze model errors?  
A:  

Q: What types of misclassifications were most common, and what might explain them?  
A:  

Q: How did your error analysis inform your next steps in model improvement?  
A:  

---

### 📝 5. Model Selection & Insights

Q: Based on your experiments, which model performed best and why?  
A:  

Q: What are your top 3–5 insights from model development and experimentation?  
A:  

Q: How would you communicate your model’s strengths and limitations to a non-technical stakeholder?  
A:  

---

## ✅ Week 4: Model Tuning & Finalization

---

### 🛠️ 1. Hyperparameter Tuning

Q: Which hyperparameters did you tune for your models, and what methods (e.g., grid search, random search) did you use?  
A:  

Q: How did you select the range or values for each hyperparameter?  
A:  

Q: What impact did hyperparameter tuning have on your model’s performance?  
A:  

---

### 🔄 2. Cross-Validation

Q: How did you use cross-validation to assess model stability and generalization?  
A:  

Q: What were the results of your cross-validation, and did you observe any variance across folds?  
A:  

Q: Why is cross-validation important in this context?  
A:  

---

### 🏆 3. Final Model Selection

Q: How did you choose your final model after tuning and validation?  
A:  

Q: Did you retrain your final model on the full training set before evaluating on the test set? Why or why not?  
A:  

Q: What were the final test set results, and how do they compare to your validation results?  
A:  

---

### 📊 4. Feature Importance & Interpretation

Q: How did you assess feature importance for your final model?  
A:  

Q: Which features were most influential in predicting diabetes risk, and do these results align with domain knowledge?  
A:  

Q: How would you explain your model’s decision process to a non-technical audience?  
A:

---

# 🚀 Week 5: Deployment

### Interview-Style Questions

Q: How did you design the Streamlit app interface for user-friendly diabetes risk prediction?

Q: What steps did you take to integrate your trained model into the Streamlit app?

Q: How did you validate that the app produces correct predictions for different user inputs?

Q: What challenges did you face when deploying the app to Streamlit Community Cloud, and how did you resolve them?

Q: How did you ensure the app provides clear and actionable feedback to users about their diabetes risk?

Q: What would you do differently if deploying this app for real-world healthcare use (security, privacy, reliability)?

Q: How would you update or retrain the model after deployment if new data becomes available?

