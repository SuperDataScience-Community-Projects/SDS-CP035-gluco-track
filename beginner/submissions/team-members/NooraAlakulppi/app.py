import streamlit as st
import joblib
import pandas as pd

# Load the trained RandomForest model
rf_classifier = joblib.load("rf_classifier.joblib")

# Define the feature names in the same order as during training
features = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
    'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk',
    'Sex', 'Age', 'Education', 'Income'
]

st.title("Diabetes Prediction App")

st.write("Enter the following health indicators to predict diabetes risk:")

# Age groups (coded as 1-13)
age_options = [
    (1, "18-24"),
    (2, "25-29"),
    (3, "30-34"),
    (4, "35-39"),
    (5, "40-44"),
    (6, "45-49"),
    (7, "50-54"),
    (8, "55-59"),
    (9, "60-64"),
    (10, "65-69"),
    (11, "70-74"),
    (12, "75-79"),
    (13, "80 or older")
]

# Education levels (1-6)
education_options = [
    (1, "Never attended school or only kindergarten"),
    (2, "Grades 1 through 8 (Elementary)"),
    (3, "Grades 9 through 11 (Some high school)"),
    (4, "Grade 12 or GED (High school graduate)"),
    (5, "College 1 year to 3 years (Some college or technical school)"),
    (6, "College 4 years or more (College graduate)")
]

# Income levels (1-8)
income_options = [
    (1, "Less than $10,000"),
    (2, "$10,000 to less than $15,000"),
    (3, "$15,000 to less than $20,000"),
    (4, "$20,000 to less than $25,000"),
    (5, "$25,000 to less than $35,000"),
    (6, "$35,000 to less than $50,000"),
    (7, "$50,000 to less than $75,000"),
    (8, "$75,000 or more")
]

# Collect user input for each feature
user_input = {}
user_input['HighBP'] = st.selectbox("High Blood Pressure (0=No, 1=Yes)", [0, 1])
user_input['HighChol'] = st.selectbox("High Cholesterol (0=No, 1=Yes)", [0, 1])
user_input['CholCheck'] = st.selectbox("Cholesterol Check in past 5 years (0=No, 1=Yes)", [0, 1])
user_input['BMI'] = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0)
user_input['Smoker'] = st.selectbox("Smoker (0=No, 1=Yes)", [0, 1])
user_input['Stroke'] = st.selectbox("Ever had a stroke? (0=No, 1=Yes)", [0, 1])
user_input['HeartDiseaseorAttack'] = st.selectbox("Heart Disease or Attack (0=No, 1=Yes)", [0, 1])
user_input['PhysActivity'] = st.selectbox("Physical Activity in past 30 days (0=No, 1=Yes)", [0, 1])
user_input['Fruits'] = st.selectbox("Consumes Fruit 1+ times/day (0=No, 1=Yes)", [0, 1])
user_input['Veggies'] = st.selectbox("Consumes Vegetables 1+ times/day (0=No, 1=Yes)", [0, 1])
user_input['HvyAlcoholConsump'] = st.selectbox("Heavy Alcohol Consumption (0=No, 1=Yes)", [0, 1])
user_input['AnyHealthcare'] = st.selectbox("Any Health Care Coverage (0=No, 1=Yes)", [0, 1])
user_input['NoDocbcCost'] = st.selectbox("Could Not See Doctor Due to Cost (0=No, 1=Yes)", [0, 1])
user_input['GenHlth'] = st.slider("General Health (1=Excellent, 5=Poor)", min_value=1, max_value=5, value=3)
user_input['MentHlth'] = st.slider("Mental Health (days not good, past 30 days)", min_value=0, max_value=30, value=0)
user_input['PhysHlth'] = st.slider("Physical Health (days not good, past 30 days)", min_value=0, max_value=30, value=0)
user_input['DiffWalk'] = st.selectbox("Difficulty Walking or Climbing Stairs (0=No, 1=Yes)", [0, 1])
user_input['Sex'] = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
user_input['Age'] = st.selectbox(
    "Age Group",
    options=[age[0] for age in age_options],
    format_func=lambda x: dict(age_options)[x]
)
user_input['Education'] = st.selectbox(
    "Education Level",
    options=[edu[0] for edu in education_options],
    format_func=lambda x: dict(education_options)[x]
)
user_input['Income'] = st.selectbox(
    "Income Level",
    options=[inc[0] for inc in income_options],
    format_func=lambda x: dict(income_options)[x]
)

# ...existing code...

if st.button("Predict Diabetes Risk"):
    # Prepare the input as a DataFrame in the correct order
    input_df = pd.DataFrame([[user_input[feature] for feature in features]], columns=features)
    prediction = rf_classifier.predict(input_df)[0]
    probability = rf_classifier.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"High risk of diabetes. (Probability: {probability:.2f})")
    else:
        st.success(f"Low risk of diabetes. (Probability: {probability:.2f})")

    # Add user-friendly summary
    # Average probability is 0.40
    average_probability = 0.40
    if probability > average_probability:
        comparison = "higher"
    elif probability < average_probability:
        comparison = "lower"
    else:
        comparison = "equal to"

    st.markdown(
        f"---\n"
        f"**Your estimated lifetime probability of getting diabetes, if nothing changes in your selections above, is "
        f"{probability:.2f} (or {probability*100:.1f}%).**\n\n"
        f"This is **{comparison}** than the average probability of getting diabetes in the USA population used for this model "
        f"(average: {average_probability:.2f} or {average_probability*100:.1f}%).")