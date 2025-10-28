"""
LiveMore MVP - Biological Age Predictor with XAI
=================================================
Streamlit app for personalized biological age prediction using Random Forest + SHAP explanations.

Model Performance:
- Training R²: 0.9499
- Training MAE: 2.02 years
- RF vs Linear Gain: +3.09%

Features: 9 simplified inputs (demographics, lifestyle, risk factors, genetics)
"""

import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================
# Configuration
# ==========================

st.set_page_config(
    page_title="LiveMore - Biological Age Predictor",
    page_icon="🧬",
    layout="centered"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "app_model")

# ==========================
# Load Model Artifacts
# ==========================

@st.cache_resource
def load_model_artifacts():
    """Load Random Forest model, scaler, and SHAP explainer."""
    try:
        model_path = os.path.join(MODEL_DIR, "livemore_rf_v2.joblib")
        scaler_path = os.path.join(MODEL_DIR, "livemore_scaler_v2.joblib")
        explainer_path = os.path.join(MODEL_DIR, "livemore_explainer_v2.pkl")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(explainer_path, "rb") as f:
            explainer = pickle.load(f)
        
        return model, scaler, explainer
    except FileNotFoundError as e:
        st.error(f"❌ Model artifacts not found: {e}")
        st.info("Please ensure model files are in `antiaging-mvp/streamlit_app/app_model/`")
        st.stop()

# ==========================
# Feature Definitions
# ==========================

FEATURE_INFO = {
    "age": {
        "label": "Age",
        "min": 25, "max": 80, "default": 45,
        "help": "Your chronological age in years"
    },
    "gender": {
        "label": "Gender",
        "options": ["Male", "Female"],
        "values": [1, 0],
        "help": "Biological sex (minimal impact on prediction: 0.1%)"
    },
    "exercise_hours_week": {
        "label": "Exercise (hours/week)",
        "min": 0.0, "max": 20.0, "default": 5.0, "step": 0.5,
        "help": "Weekly exercise hours (diminishing returns pattern)"
    },
    "diet_quality_score": {
        "label": "Diet Quality Score",
        "min": 1, "max": 10, "default": 7,
        "help": "1=Poor, 10=Excellent. Quadratic benefit at 8+"
    },
    "sleep_hours": {
        "label": "Sleep (hours/night)",
        "min": 4.0, "max": 10.0, "default": 7.5, "step": 0.5,
        "help": "Average nightly sleep. Optimal: 7.5h (U-curve pattern)"
    },
    "stress_level": {
        "label": "Stress Level",
        "min": 1, "max": 10, "default": 5,
        "help": "1=Low, 10=High. Exponential damage pattern"
    },
    "smoking_pack_years": {
        "label": "Smoking (pack-years)",
        "min": 0, "max": 40, "default": 0,
        "help": "Packs per day × years smoked. Critical risk factor (11.3% importance)"
    },
    "alcohol_drinks_week": {
        "label": "Alcohol (drinks/week)",
        "min": 0, "max": 30, "default": 5,
        "help": "Weekly alcohol consumption. Protective ≤7, harmful >7"
    },
    "genetic_risk_score": {
        "label": "Genetic Risk Score",
        "min": 0.0, "max": 1.0, "default": 0.5, "step": 0.05,
        "help": "Simplified genetic risk (0=low, 1=high). Interacts with lifestyle."
    }
}

FEATURE_ORDER = [
    "age", "gender", "exercise_hours_week", "diet_quality_score",
    "sleep_hours", "stress_level", "smoking_pack_years",
    "alcohol_drinks_week", "genetic_risk_score"
]

# Business-friendly feature names for display
FEATURE_DISPLAY_NAMES = {
    "age": "Age",
    "gender": "Gender",
    "exercise_hours_week": "Exercise",
    "diet_quality_score": "Diet Quality",
    "sleep_hours": "Sleep",
    "stress_level": "Stress",
    "smoking_pack_years": "Smoking",
    "alcohol_drinks_week": "Alcohol",
    "genetic_risk_score": "Genetic Risk"
}

# ==========================
# Helper Functions
# ==========================

def create_input_dataframe(inputs):
    """Convert user inputs to DataFrame matching model training format."""
    df = pd.DataFrame([inputs], columns=FEATURE_ORDER)
    return df

def predict_biological_age(model, scaler, input_df):
    """Make prediction using loaded model."""
    X_scaled = scaler.transform(input_df)
    prediction = model.predict(X_scaled)[0]
    return prediction

def generate_shap_explanation(explainer, scaler, input_df):
    """Generate SHAP values for input."""
    X_scaled = scaler.transform(input_df)
    shap_values = explainer.shap_values(X_scaled)
    return shap_values[0]  # Return values for single prediction

def interpret_age_difference(predicted, chronological):
    """Provide business-friendly interpretation of age difference."""
    diff = predicted - chronological
    
    if abs(diff) < 2:
        return "🎯 **Excellent!** Your biological age matches your chronological age.", "success"
    elif diff < 0:
        years = abs(diff)
        return f"✨ **Great news!** You're biologically {years:.1f} years younger than your age.", "success"
    elif diff < 5:
        return f"⚠️ Your biological age is {diff:.1f} years older. Consider lifestyle improvements.", "warning"
    else:
        return f"🔴 **Action needed:** {diff:.1f} years older biologically. Prioritize health interventions.", "error"

# ==========================
# Main App
# ==========================

def main():
    # Load model artifacts
    model, scaler, explainer = load_model_artifacts()
    
    # Header
    st.title("🧬 LiveMore - Biological Age Predictor")
    st.markdown("""
    Discover your biological age using AI and get personalized insights about factors affecting your aging.
    
    **Model Performance:** R²=0.95, MAE=2.02 years | **Powered by:** Random Forest + SHAP XAI
    """)
    
    st.divider()
    
    # Sidebar - Input Form
    with st.sidebar:
        st.header("📋 Your Health Profile")
        st.markdown("*Fill in your information below*")
        
        inputs = {}
        
        # Demographics
        st.subheader("Demographics")
        inputs["age"] = st.slider(
            FEATURE_INFO["age"]["label"],
            FEATURE_INFO["age"]["min"],
            FEATURE_INFO["age"]["max"],
            FEATURE_INFO["age"]["default"],
            help=FEATURE_INFO["age"]["help"]
        )
        
        gender_display = st.radio(
            FEATURE_INFO["gender"]["label"],
            FEATURE_INFO["gender"]["options"],
            help=FEATURE_INFO["gender"]["help"]
        )
        inputs["gender"] = FEATURE_INFO["gender"]["values"][
            FEATURE_INFO["gender"]["options"].index(gender_display)
        ]
        
        # Lifestyle
        st.subheader("Lifestyle")
        inputs["exercise_hours_week"] = st.slider(
            FEATURE_INFO["exercise_hours_week"]["label"],
            FEATURE_INFO["exercise_hours_week"]["min"],
            FEATURE_INFO["exercise_hours_week"]["max"],
            FEATURE_INFO["exercise_hours_week"]["default"],
            FEATURE_INFO["exercise_hours_week"]["step"],
            help=FEATURE_INFO["exercise_hours_week"]["help"]
        )
        
        inputs["diet_quality_score"] = st.slider(
            FEATURE_INFO["diet_quality_score"]["label"],
            FEATURE_INFO["diet_quality_score"]["min"],
            FEATURE_INFO["diet_quality_score"]["max"],
            FEATURE_INFO["diet_quality_score"]["default"],
            help=FEATURE_INFO["diet_quality_score"]["help"]
        )
        
        inputs["sleep_hours"] = st.slider(
            FEATURE_INFO["sleep_hours"]["label"],
            FEATURE_INFO["sleep_hours"]["min"],
            FEATURE_INFO["sleep_hours"]["max"],
            FEATURE_INFO["sleep_hours"]["default"],
            FEATURE_INFO["sleep_hours"]["step"],
            help=FEATURE_INFO["sleep_hours"]["help"]
        )
        
        inputs["stress_level"] = st.slider(
            FEATURE_INFO["stress_level"]["label"],
            FEATURE_INFO["stress_level"]["min"],
            FEATURE_INFO["stress_level"]["max"],
            FEATURE_INFO["stress_level"]["default"],
            help=FEATURE_INFO["stress_level"]["help"]
        )
        
        # Risk Factors
        st.subheader("Risk Factors")
        inputs["smoking_pack_years"] = st.slider(
            FEATURE_INFO["smoking_pack_years"]["label"],
            FEATURE_INFO["smoking_pack_years"]["min"],
            FEATURE_INFO["smoking_pack_years"]["max"],
            FEATURE_INFO["smoking_pack_years"]["default"],
            help=FEATURE_INFO["smoking_pack_years"]["help"]
        )
        
        inputs["alcohol_drinks_week"] = st.slider(
            FEATURE_INFO["alcohol_drinks_week"]["label"],
            FEATURE_INFO["alcohol_drinks_week"]["min"],
            FEATURE_INFO["alcohol_drinks_week"]["max"],
            FEATURE_INFO["alcohol_drinks_week"]["default"],
            help=FEATURE_INFO["alcohol_drinks_week"]["help"]
        )
        
        # Genetics
        st.subheader("Genetics")
        inputs["genetic_risk_score"] = st.slider(
            FEATURE_INFO["genetic_risk_score"]["label"],
            FEATURE_INFO["genetic_risk_score"]["min"],
            FEATURE_INFO["genetic_risk_score"]["max"],
            FEATURE_INFO["genetic_risk_score"]["default"],
            FEATURE_INFO["genetic_risk_score"]["step"],
            help=FEATURE_INFO["genetic_risk_score"]["help"]
        )
        
        st.divider()
        predict_button = st.button("🔮 Predict My Biological Age", type="primary", use_container_width=True)
    
    # Main Content - Results
    if predict_button:
        with st.spinner("Analyzing your health profile..."):
            # Create input DataFrame
            input_df = create_input_dataframe(inputs)
            
            # Make prediction
            predicted_age = predict_biological_age(model, scaler, input_df)
            chronological_age = inputs["age"]
            
            # Generate SHAP explanation
            shap_values = generate_shap_explanation(explainer, scaler, input_df)
            
            # Display Results
            st.header("📊 Your Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Chronological Age",
                    f"{chronological_age} years",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Biological Age",
                    f"{predicted_age:.1f} years",
                    delta=f"{predicted_age - chronological_age:+.1f} years",
                    delta_color="inverse"
                )
            
            with col3:
                age_diff = abs(predicted_age - chronological_age)
                st.metric(
                    "Age Difference",
                    f"{age_diff:.1f} years",
                    delta=None
                )
            
            # Interpretation
            message, message_type = interpret_age_difference(predicted_age, chronological_age)
            if message_type == "success":
                st.success(message)
            elif message_type == "warning":
                st.warning(message)
            else:
                st.error(message)
            
            st.divider()
            
            # SHAP Explanation
            st.header("🔍 What's Affecting Your Biological Age?")
            st.markdown("*Each factor's contribution to your biological age (in years)*")
            
            # Create SHAP waterfall plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort features by absolute SHAP value
            feature_names = [FEATURE_DISPLAY_NAMES[f] for f in FEATURE_ORDER]
            shap_df = pd.DataFrame({
                'feature': feature_names,
                'shap_value': shap_values
            })
            shap_df['abs_shap'] = abs(shap_df['shap_value'])
            shap_df = shap_df.sort_values('abs_shap', ascending=True)
            
            # Plot horizontal bar chart
            colors = ['#d62728' if x > 0 else '#2ca02c' for x in shap_df['shap_value']]
            ax.barh(shap_df['feature'], shap_df['shap_value'], color=colors, alpha=0.7)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_xlabel('Impact on Biological Age (years)', fontsize=11)
            ax.set_title('Feature Contributions to Your Biological Age', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (feature, value) in enumerate(zip(shap_df['feature'], shap_df['shap_value'])):
                label = f'{value:+.1f}'
                x_pos = value + (0.3 if value > 0 else -0.3)
                ax.text(x_pos, i, label, va='center', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Explanation text
            st.markdown("""
            **How to read this chart:**
            - 🔴 **Red bars** (positive): Factors increasing your biological age
            - 🟢 **Green bars** (negative): Factors decreasing your biological age
            - **Longer bars** = stronger impact on your biological age
            """)
            
            st.divider()
            
            # Recommendations
            st.header("💡 Personalized Recommendations")
            
            # Find top 3 negative contributors (areas to improve)
            shap_df_sorted = shap_df.sort_values('shap_value', ascending=False)
            top_negative = shap_df_sorted[shap_df_sorted['shap_value'] > 0.5].head(3)
            
            if len(top_negative) > 0:
                st.markdown("**Focus on improving these areas:**")
                for idx, row in top_negative.iterrows():
                    st.markdown(f"- **{row['feature']}**: Adding {row['shap_value']:.1f} years to your biological age")
            else:
                st.success("🎉 Excellent! No major negative factors detected. Keep up the great work!")
            
            # Find top protective factors
            top_positive = shap_df_sorted[shap_df_sorted['shap_value'] < -0.5].tail(3)
            
            if len(top_positive) > 0:
                st.markdown("**Your protective factors (keep it up!):**")
                for idx, row in top_positive.iterrows():
                    st.markdown(f"- **{row['feature']}**: Reducing your biological age by {abs(row['shap_value']):.1f} years")
            
            st.divider()
            
            # Raw data (collapsible)
            with st.expander("🔬 View Technical Details"):
                st.subheader("Your Input Data")
                st.dataframe(input_df)
                
                st.subheader("SHAP Values (Raw)")
                shap_raw_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Your Value': input_df.iloc[0].values,
                    'SHAP Value (years)': shap_values
                })
                st.dataframe(shap_raw_df)
                
                st.subheader("Model Information")
                st.markdown("""
                - **Algorithm:** Random Forest Regressor (200 estimators, max_depth=15)
                - **Training R²:** 0.9499
                - **Training MAE:** 2.02 years
                - **RF vs Linear Gain:** +3.09%
                - **Explanation Method:** SHAP (TreeExplainer with 500 background samples)
                """)
    
    else:
        # Welcome message when no prediction yet
        st.info("👈 **Get started:** Fill in your health profile in the sidebar and click 'Predict' to see your results!")
        
        st.markdown("""
        ### About This Tool
        
        LiveMore uses a **Random Forest machine learning model** trained on 5,000 synthetic health profiles to predict your biological age based on:
        
        - **Demographics:** Age, gender
        - **Lifestyle:** Exercise, diet, sleep, stress
        - **Risk Factors:** Smoking, alcohol consumption
        - **Genetics:** Simplified genetic risk score
        
        ### Why Biological Age Matters
        
        Your **chronological age** is just a number—it's your **biological age** that truly reflects how your body is aging. By understanding which factors are accelerating or slowing your aging process, you can make informed decisions to improve your health span.
        
        ### About the Technology
        
        This MVP uses **Explainable AI (XAI)** through SHAP values to show you exactly which factors are affecting your biological age and by how much. Unlike "black box" AI, you can see and understand every prediction.
        
        ---
        
        **⚠️ Disclaimer:** This is a research prototype using synthetic data. Not intended for medical diagnosis or treatment decisions. Always consult healthcare professionals for medical advice.
        """)

if __name__ == "__main__":
    main()
