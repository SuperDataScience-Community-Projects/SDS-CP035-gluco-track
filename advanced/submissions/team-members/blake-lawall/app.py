# Week 5: Complete Hugging Face Spaces Deployment with Gradio
# File: app.py

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import io
import warnings
from functools import lru_cache
warnings.filterwarnings('ignore')

# Model Architecture Definition
class OptimizedFFNN(nn.Module):
    """Optimized neural network from Week 4"""
    def __init__(self, hidden_sizes=[128, 64, 32], dropout_rates=[0.3, 0.3, 0.2]):
        super().__init__()
        layers = []
        input_size = 21
        
        for hidden_size, dropout in zip(hidden_sizes, dropout_rates):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Sigmoid())
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

# Global variables for model, config, and scaler
MODEL = None
CONFIG = None
SCALER = None
FEATURE_PLOT_CACHE = None

def load_model_and_config():
    """Load the trained model and configuration"""
    global MODEL, CONFIG, SCALER
    
    try:
        # Load deployment configuration
        with open('deployment_config.json', 'r') as f:
            CONFIG = json.load(f)
        
        # Load and initialize model
        device = torch.device('cpu')  # Use CPU for deployment
        MODEL = OptimizedFFNN(
            hidden_sizes=CONFIG['model_architecture']['hidden_sizes'],
            dropout_rates=CONFIG['model_architecture']['dropout_rates']
        )
        
        # Load trained weights
        MODEL.load_state_dict(torch.load('week4_best_model.pth', map_location=device))
        MODEL.eval()
        
        # Load preprocessing scaler
        with open('robust_scaler.pkl', 'rb') as f:
            SCALER = pickle.load(f)
        
        return True
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def preprocess_inputs(inputs):
    """Preprocess user inputs for model prediction"""
    try:
        # Extract values in correct feature order
        feature_order = CONFIG['feature_names']
        
        # Map inputs to feature vector
        feature_values = []
        for feature in feature_order:
            if feature in inputs:
                feature_values.append(inputs[feature])
            else:
                # Default values for missing features
                feature_values.append(0)
        
        # Convert to numpy array
        X = np.array(feature_values).reshape(1, -1)
        
        # Scale continuous features (BMI, MentHlth, PhysHlth, Age)
        # These are typically at indices 3, 14, 15, 18 but adjust based on your feature order
        continuous_indices = [3, 14, 15, 18]  # Adjust based on your actual feature order
        X_scaled = X.copy()
        
        if SCALER is not None and len(continuous_indices) > 0:
            try:
                X_scaled[:, continuous_indices] = SCALER.transform(X[:, continuous_indices])
            except Exception as e:
                print(f"Scaling warning: {e}, using original values")
                # If scaling fails, use original values
                pass
        
        return X_scaled
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        # Return a default array if preprocessing fails
        return np.zeros((1, 21))

def make_prediction(X_scaled):
    """Make prediction using the trained model"""
    device = torch.device('cpu')
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    with torch.no_grad():
        probability = MODEL(X_tensor).squeeze().item()
    
    # Use optimal threshold from Week 4
    threshold = CONFIG['threshold']
    prediction = 1 if probability > threshold else 0
    
    return prediction, probability, threshold

def predict_diabetes_risk(
    # Health Conditions
    high_bp, high_chol, chol_check, smoker, stroke, heart_disease,
    # Lifestyle
    phys_activity, fruits, veggies, heavy_alcohol,
    # Healthcare Access
    any_healthcare, no_doc_cost, diff_walk,
    # Demographics
    sex, age,
    # Health Metrics
    bmi, mental_health_days, physical_health_days,
    # Ordinal Features
    general_health, education, income
):
    """Main prediction function for Gradio interface"""
    
    if MODEL is None or CONFIG is None or SCALER is None:
        return "❌ Model not loaded. Please check deployment files.", "", ""
    
    # Convert string values to integers for the model
    def convert_yes_no(value):
        """Convert Yes/No to 1/0"""
        return 1 if value == "Yes" else 0
    
    def convert_sex(sex_str):
        """Convert Male/Female to 1/0"""
        return 1 if sex_str == "Male" else 0
    
    def convert_general_health(health_str):
        mapping = {
            "1 - Excellent": 1, "2 - Very Good": 2, "3 - Good": 3, 
            "4 - Fair": 4, "5 - Poor": 5
        }
        return mapping.get(health_str, 3)
    
    def convert_education(edu_str):
        mapping = {
            "Never attended": 1, "Elementary": 2, "Some high school": 3,
            "High school graduate": 4, "Some college": 5, "College graduate": 6
        }
        return mapping.get(edu_str, 4)
    
    def convert_income(income_str):
        mapping = {
            "<$10k": 1, "$10k-$15k": 2, "$15k-$20k": 3, "$20k-$25k": 4,
            "$25k-$35k": 5, "$35k-$50k": 6, "$50k-$75k": 7, "$75k+": 8
        }
        return mapping.get(income_str, 5)
    
    # Create input dictionary
    inputs = {
        'HighBP': convert_yes_no(high_bp),
        'HighChol': convert_yes_no(high_chol),
        'CholCheck': convert_yes_no(chol_check),
        'Smoker': convert_yes_no(smoker),
        'Stroke': convert_yes_no(stroke),
        'HeartDiseaseorAttack': convert_yes_no(heart_disease),
        'PhysActivity': convert_yes_no(phys_activity),
        'Fruits': convert_yes_no(fruits),
        'Veggies': convert_yes_no(veggies),
        'HvyAlcoholConsump': convert_yes_no(heavy_alcohol),
        'AnyHealthcare': convert_yes_no(any_healthcare),
        'NoDocbcCost': convert_yes_no(no_doc_cost),
        'DiffWalk': convert_yes_no(diff_walk),
        'Sex': convert_sex(sex),
        'Age': age,
        'BMI': bmi,
        'MentHlth': mental_health_days,
        'PhysHlth': physical_health_days,
        'GenHlth': convert_general_health(general_health),
        'Education': convert_education(education),
        'Income': convert_income(income)
    }
    
    try:
        # Preprocess inputs
        X_scaled = preprocess_inputs(inputs)
        
        # Make prediction
        prediction, probability, threshold = make_prediction(X_scaled)
        
        # Create prediction result
        if prediction == 1:
            risk_level = "⚠️ HIGH RISK"
            risk_color = "🔴"
            main_message = f"{risk_color} **HIGH DIABETES RISK DETECTED**\n\n"
            main_message += f"**Risk Probability: {probability:.1%}**\n\n"
            main_message += "This assessment suggests elevated diabetes risk. "
            main_message += "**Medical consultation is recommended** for proper evaluation and preventive care."
        else:
            risk_level = "✅ LOW RISK"
            risk_color = "🟢"
            main_message = f"{risk_color} **LOW DIABETES RISK**\n\n"
            main_message += f"**Risk Probability: {probability:.1%}**\n\n"
            main_message += "This assessment suggests lower diabetes risk. "
            main_message += "Continue healthy lifestyle practices and regular check-ups."
        
        # Risk factor analysis
        risk_factors = analyze_risk_factors(inputs)
        
        # Recommendations
        recommendations = generate_recommendations(prediction, inputs)
        
        return main_message, risk_factors, recommendations
        
    except Exception as e:
        return f"❌ Error making prediction: {str(e)}", "", ""

def analyze_risk_factors(inputs):
    """Analyze individual risk factors"""
    risk_factors_lines = ["## 📊 Risk Factor Analysis\n"]
    
    high_risk_factors = []
    protective_factors = []
    
    # Define risk thresholds
    if inputs.get('HighBP', 0) == 1:
        high_risk_factors.append("**High Blood Pressure** - Significantly increases diabetes risk")
    
    if inputs.get('HighChol', 0) == 1:
        high_risk_factors.append("**High Cholesterol** - Associated with increased diabetes risk")
    
    if inputs.get('BMI', 25) >= 30:
        high_risk_factors.append(f"**Obesity** - BMI {inputs.get('BMI', 25):.1f} (≥30 increases risk)")
    elif inputs.get('BMI', 25) >= 25:
        high_risk_factors.append(f"**Overweight** - BMI {inputs.get('BMI', 25):.1f} (≥25 increases risk)")
    
    if inputs.get('Age', 50) >= 45:
        high_risk_factors.append(f"**Age** - {inputs.get('Age', 50)} years (≥45 increases risk)")
    
    if inputs.get('PhysActivity', 1) == 0:
        high_risk_factors.append("**Lack of Physical Activity** - Increases diabetes risk")
    else:
        protective_factors.append("**Regular Physical Activity** - Helps reduce diabetes risk")
    
    if inputs.get('Smoker', 0) == 1:
        high_risk_factors.append("**Smoking History** - Increases diabetes risk")
    
    if inputs.get('GenHlth', 3) >= 4:
        high_risk_factors.append("**Poor General Health** - Associated with higher diabetes risk")
    
    if inputs.get('Fruits', 1) == 1 and inputs.get('Veggies', 1) == 1:
        protective_factors.append("**Healthy Diet** - Regular fruit and vegetable consumption")
    
    # Build risk factors text
    if high_risk_factors:
        risk_factors_lines.append("### ⚠️ Risk Factors Identified:")
        risk_factors_lines.extend([f"• {factor}" for factor in high_risk_factors])
        risk_factors_lines.append("")
    
    if protective_factors:
        risk_factors_lines.append("### ✅ Protective Factors:")
        risk_factors_lines.extend([f"• {factor}" for factor in protective_factors])
        risk_factors_lines.append("")
    
    if not high_risk_factors:
        risk_factors_lines.append("### ✅ No Major Risk Factors Identified")
        risk_factors_lines.append("Current health indicators suggest lower diabetes risk.\n")
    
    return "\n".join(risk_factors_lines)

def generate_recommendations(prediction, inputs):
    """Generate personalized recommendations"""
    recommendations_lines = ["## 💡 Personalized Recommendations\n"]
    
    recommendations = []
    
    # BMI-based recommendations
    bmi = inputs.get('BMI', 25)
    if bmi >= 30:
        recommendations.append("🏃‍♀️ **Weight Management**: Consult healthcare provider about safe weight loss strategies (target BMI <30)")
    elif bmi >= 25:
        recommendations.append("⚖️ **Weight Maintenance**: Maintain healthy weight through balanced diet and regular exercise")
    
    # Physical activity
    if inputs.get('PhysActivity', 1) == 0:
        recommendations.append("💪 **Exercise Program**: Aim for 150+ minutes of moderate exercise weekly")
        recommendations.append("🚶‍♀️ **Start Simple**: Begin with daily walks and gradually increase activity")
    
    # Diet recommendations
    if inputs.get('Fruits', 1) == 0:
        recommendations.append("🍎 **Fruit Intake**: Include at least 1 serving of fruit daily")
    
    if inputs.get('Veggies', 1) == 0:
        recommendations.append("🥗 **Vegetable Intake**: Include at least 1 serving of vegetables daily")
    
    # Health monitoring for high-risk conditions
    if inputs.get('HighBP', 0) == 1:
        recommendations.append("🩺 **Blood Pressure**: Continue regular monitoring and medication compliance")
    
    if inputs.get('HighChol', 0) == 1:
        recommendations.append("💊 **Cholesterol Management**: Follow prescribed treatment and dietary guidelines")
    
    # Smoking cessation
    if inputs.get('Smoker', 0) == 1:
        recommendations.append("🚭 **Smoking Cessation**: Consider quitting programs - reduces diabetes risk significantly")
    
    # General recommendations
    recommendations.extend([
        "📅 **Regular Check-ups**: Schedule annual health screenings with healthcare provider",
        "🩸 **Blood Sugar Monitoring**: Consider periodic glucose testing, especially if at higher risk",
        "📚 **Health Education**: Stay informed about diabetes prevention strategies",
        "🧘‍♀️ **Stress Management**: Practice stress reduction techniques (meditation, yoga, adequate sleep)"
    ])
    
    # Add recommendations to text
    for i, rec in enumerate(recommendations, 1):
        recommendations_lines.append(f"{i}. {rec}\n")
    
    # Add important disclaimers
    if prediction == 1:
        recommendations_lines.append("---\n")
        recommendations_lines.append("⚠️ **Important**: This assessment is for screening purposes only. ")
        recommendations_lines.append("**Please consult with a healthcare professional** for proper medical evaluation, ")
        recommendations_lines.append("diagnosis, and personalized treatment planning.\n")
    
    recommendations_lines.append("ℹ️ **Note**: These recommendations are general guidelines. ")
    recommendations_lines.append("Always consult healthcare professionals for personalized medical advice.")
    
    return "\n".join(recommendations_lines)

def create_feature_importance_plot():
    """Create feature importance visualization"""
    global FEATURE_PLOT_CACHE
    if FEATURE_PLOT_CACHE is not None:
        return FEATURE_PLOT_CACHE
        
    if CONFIG is None:
        return None
    
    try:
        # Get top 10 features
        importance_data = CONFIG['feature_importance'][:10]
        
        features = [item['Feature'] for item in importance_data]
        importance = [item['SHAP_Importance'] for item in importance_data]
        
        # Create matplotlib plot
        plt.figure(figsize=(10, 6))
        bars = plt.barh(features, importance, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.xlabel('Feature Importance (SHAP Value)')
        plt.title('🔍 Top 10 Most Important Features for Diabetes Prediction')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for bar, value in zip(bars, importance):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        FEATURE_PLOT_CACHE = buf.getvalue()
        return FEATURE_PLOT_CACHE
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        return None

@lru_cache(maxsize=1)
def get_model_info():
    """Get model information for display"""
    if CONFIG is None:
        return "Model information not available."
    
    info_text = f"""
## 📊 Model Performance Metrics

**Accuracy**: {CONFIG['performance']['accuracy']:.1%}
**F1-Score**: {CONFIG['performance']['f1_score']:.3f}
**AUC-ROC**: {CONFIG['performance']['auc_roc']:.3f}

## 🧠 Model Architecture

**Type**: Optimized Feedforward Neural Network
**Input Features**: 21 health indicators
**Hidden Layers**: {' → '.join(map(str, CONFIG['model_architecture']['hidden_sizes']))}
**Regularization**: Dropout {CONFIG['model_architecture']['dropout_rates']}
**Optimization**: Threshold tuning + hyperparameter search

## 🔬 Training Details

**Dataset**: BRFSS (Behavioral Risk Factor Surveillance System)
**Class Imbalance**: 6.18:1 ratio (healthy:diabetic)
**Optimization**: Weighted loss function with optimal threshold
**Validation**: Stratified cross-validation
**Preprocessing**: RobustScaler for continuous features

## ⚠️ Important Disclaimers

- This tool is for **educational and screening purposes only**
- Results should **not replace professional medical advice**
- Always **consult healthcare professionals** for medical decisions
- Model has limitations and may not capture all risk factors
"""
    
    return info_text

# Initialize model on startup
print("Loading model and configuration...")
if load_model_and_config():
    print("✅ Model loaded successfully!")
else:
    print("❌ Failed to load model!")

# Create Gradio interface
def create_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(
        title="🩺 Diabetes Risk Predictor",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .risk-high {
            background-color: #ffebee !important;
            border: 2px solid #f44336 !important;
            border-radius: 10px;
            padding: 20px;
        }
        .risk-low {
            background-color: #e8f5e8 !important;
            border: 2px solid #4caf50 !important;
            border-radius: 10px;
            padding: 20px;
        }
        """
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🩺 AI-Powered Diabetes Risk Predictor
        
        **Advanced machine learning model for diabetes risk assessment based on health indicators and lifestyle factors.**
        
        *This tool is for educational and screening purposes only. Always consult healthcare professionals for medical advice.*
        """)
        
        with gr.Tabs():
            # Tab 1: Risk Assessment
            with gr.TabItem("🔮 Risk Assessment"):
                gr.Markdown("### Complete the health assessment form below:")
                
                with gr.Row():
                    # Column 1: Health Conditions
                    with gr.Column():
                        gr.Markdown("#### 🩺 Health Conditions")
                        high_bp = gr.Radio(["No", "Yes"], label="High Blood Pressure", value="No", info="Do you have high blood pressure?")
                        high_chol = gr.Radio(["No", "Yes"], label="High Cholesterol", value="No", info="Do you have high cholesterol?")
                        chol_check = gr.Radio(["No", "Yes"], label="Cholesterol Check (Past 5 Years)", value="Yes", info="Cholesterol check in past 5 years?")
                        smoker = gr.Radio(["No", "Yes"], label="Smoking History", value="No", info="Have you smoked 100+ cigarettes in your lifetime?")
                        stroke = gr.Radio(["No", "Yes"], label="Stroke History", value="No", info="Have you ever had a stroke?")
                        heart_disease = gr.Radio(["No", "Yes"], label="Heart Disease/Attack", value="No", info="Coronary heart disease or heart attack?")
                        
                    # Column 2: Lifestyle
                    with gr.Column():
                        gr.Markdown("#### 🏃‍♀️ Lifestyle Factors")
                        phys_activity = gr.Radio(["No", "Yes"], label="Physical Activity (Past 30 Days)", value="Yes", info="Any physical activity in past 30 days?")
                        fruits = gr.Radio(["No", "Yes"], label="Daily Fruit Consumption", value="Yes", info="Do you eat fruit 1+ times per day?")
                        veggies = gr.Radio(["No", "Yes"], label="Daily Vegetable Consumption", value="Yes", info="Do you eat vegetables 1+ times per day?")
                        heavy_alcohol = gr.Radio(["No", "Yes"], label="Heavy Alcohol Use", value="No", info="Heavy drinking (men: 14+/week, women: 7+/week)?")
                        any_healthcare = gr.Radio(["No", "Yes"], label="Healthcare Coverage", value="Yes", info="Do you have health insurance?")
                        no_doc_cost = gr.Radio(["No", "Yes"], label="Cost Barrier to Doctor", value="No", info="Couldn't see doctor due to cost in past year?")
                        diff_walk = gr.Radio(["No", "Yes"], label="Difficulty Walking", value="No", info="Serious difficulty walking/climbing stairs?")
                        
                    # Column 3: Demographics & Health Metrics
                    with gr.Column():
                        gr.Markdown("#### 📊 Demographics & Health Metrics")
                        sex = gr.Radio(["Female", "Male"], label="Sex", value="Female", info="Select your sex")
                        age = gr.Slider(18, 80, value=50, step=1, label="Age (years)")
                        bmi = gr.Slider(12.0, 50.0, value=25.0, step=0.1, label="BMI (Body Mass Index)")
                        mental_health_days = gr.Slider(0, 30, value=0, step=1, label="Poor Mental Health Days (Past 30)")
                        physical_health_days = gr.Slider(0, 30, value=0, step=1, label="Poor Physical Health Days (Past 30)")
                        
                        gr.Markdown("#### 📋 Health & Socioeconomic Status")
                        general_health = gr.Dropdown(
                            choices=["1 - Excellent", "2 - Very Good", "3 - Good", "4 - Fair", "5 - Poor"],
                            value="3 - Good", label="General Health"
                        )
                        education = gr.Dropdown(
                            choices=["Never attended", "Elementary", "Some high school", 
                                   "High school graduate", "Some college", "College graduate"],
                            value="High school graduate", label="Education Level"
                        )
                        income = gr.Dropdown(
                            choices=["<$10k", "$10k-$15k", "$15k-$20k", "$20k-$25k",
                                   "$25k-$35k", "$35k-$50k", "$50k-$75k", "$75k+"],
                            value="$25k-$35k", label="Annual Household Income"
                        )
                
                # Prediction button
                predict_btn = gr.Button("🔍 Assess Diabetes Risk", variant="primary", size="lg")
                
                # Output sections
                with gr.Row():
                    with gr.Column():
                        prediction_output = gr.Markdown(label="Risk Assessment")
                    
                with gr.Row():
                    with gr.Column():
                        risk_factors_output = gr.Markdown(label="Risk Factor Analysis")
                    with gr.Column():
                        recommendations_output = gr.Markdown(label="Recommendations")
                
                # Connect prediction function
                predict_btn.click(
                    fn=predict_diabetes_risk,
                    inputs=[
                        high_bp, high_chol, chol_check, smoker, stroke, heart_disease,
                        phys_activity, fruits, veggies, heavy_alcohol,
                        any_healthcare, no_doc_cost, diff_walk,
                        sex, age, bmi, mental_health_days, physical_health_days,
                        general_health, education, income
                    ],
                    outputs=[prediction_output, risk_factors_output, recommendations_output]
                )
            
            # Tab 2: Model Insights
            with gr.TabItem("📊 Model Insights"):
                gr.Markdown("## 🔍 Model Performance & Feature Importance")
                
                model_info = gr.Markdown(get_model_info())
                
                if CONFIG is not None:
                    gr.Markdown("### 📈 Feature Importance (Top 10)")
                    feature_plot = create_feature_importance_plot()
                    if feature_plot:
                        # Convert bytes to PIL Image for Gradio 5.x compatibility
                        from PIL import Image
                        import io
                        pil_image = Image.open(io.BytesIO(feature_plot))
                        gr.Image(value=pil_image, label="Feature Importance")
                    
                    gr.Markdown("""
                    ### 💡 Understanding Feature Importance
                    
                    The chart above shows the most influential factors in diabetes risk prediction based on SHAP 
                    (SHapley Additive exPlanations) analysis. Higher values indicate greater impact on predictions.
                    
                    **Key Insights:**
                    - **Medical conditions** (High BP, High Cholesterol) typically rank highest
                    - **Lifestyle factors** (Physical Activity, BMI) have significant impact  
                    - **Demographics** (Age, General Health) provide important context
                    - **Socioeconomic factors** (Income, Education) influence health outcomes
                    """)
            
            # Tab 3: About
            with gr.TabItem("❓ About"):
                gr.Markdown("""
                ## 🎯 About This Application
                
                This diabetes risk assessment tool demonstrates the application of machine learning 
                in healthcare screening. It was developed through a comprehensive 5-week project:
                
                - **Week 1**: Exploratory Data Analysis of BRFSS dataset
                - **Week 2**: Feature engineering and preprocessing pipeline  
                - **Week 3**: Neural network design and baseline training
                - **Week 4**: Model optimization and explainability integration
                - **Week 5**: Production deployment on Hugging Face Spaces
                
                ### 📊 Dataset & Training
                
                **Data Source**: Behavioral Risk Factor Surveillance System (BRFSS) 2015
                - **Size**: 250,000+ health survey responses
                - **Features**: 21 health indicators, demographics, and lifestyle factors
                - **Target**: Binary diabetes diagnosis
                - **Challenge**: 6.18:1 class imbalance (healthy:diabetic)
                
                ### 🧠 Model Architecture
                
                **Neural Network Design**:
                - **Input**: 21 features (health conditions, lifestyle, demographics)
                - **Architecture**: Progressive reduction (21→128→64→32→1)
                - **Optimization**: Adam optimizer with learning rate scheduling
                - **Regularization**: Dropout layers, early stopping
                - **Loss Function**: Weighted Binary Cross-Entropy (handles imbalance)
                
                ### 🔬 Model Performance
                
                **Optimization Process**:
                - **Threshold Tuning**: Optimized decision boundary for medical screening
                - **Hyperparameter Search**: Learning rate, dropout, architecture variants
                - **Class Imbalance**: Weighted loss + optimal threshold
                - **Validation**: Stratified cross-validation maintaining class distribution
                
                ### 🔍 Explainability
                
                **SHAP Analysis**: Provides transparent, interpretable predictions
                - **Global Importance**: Which features matter most overall
                - **Local Explanations**: How features affect individual predictions
                - **Medical Relevance**: Clinically interpretable feature contributions
                
                ### ⚠️ Important Disclaimers
                
                - **Educational Purpose**: This tool is for learning and screening demonstration
                - **Not Medical Advice**: Results cannot replace professional medical consultation  
                - **Limitations**: Model trained on survey data, may not capture all risk factors
                - **Individual Variation**: Personal and family medical history not included
                - **Consult Professionals**: Always seek healthcare provider guidance for medical decisions
                
                ### 🔗 Technical Resources
                
                - **Framework**: PyTorch for deep learning, Gradio for interface
                - **Deployment**: Hugging Face Spaces with automatic scaling
                - **Preprocessing**: RobustScaler for outlier-robust feature scaling
                - **Explainability**: SHAP (SHapley Additive exPlanations)
                - **Validation**: Medical AI best practices and responsible deployment
                
                ### 👨‍💻 Development Notes
                
                This application represents a complete machine learning pipeline from 
                raw data to production deployment, emphasizing:
                - **Medical AI Ethics**: Responsible development and deployment
                - **Explainable AI**: Transparent and interpretable predictions
                - **Production Quality**: Robust, scalable, user-friendly interface
                - **Educational Value**: Demonstrating ML best practices in healthcare
                
                ---
                
                **Remember**: This tool is for educational and demonstration purposes. 
                Always consult qualified healthcare professionals for medical advice and diagnosis.
                """)
        
        return demo

# Create and launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
