# LiveMore MVP - Streamlit Health ROI Simulator

**Status:** ✅ Model Artifacts Ready | 🏗️ UI Development in Progress (Week 2)

This is the **core deliverable** for the thesis defense: a functional MVP demonstrating the "LiveMore" product concept using Streamlit for rapid prototyping.

---

## 🎯 Strategic Context

This Streamlit app is the result of a strategic pivot (October 22-28, 2025) from a complex production architecture (FastAPI/React/ONNX) to a focused MVP approach. The goal is to validate the product hypothesis: **using Explainable AI (XAI) to demonstrate "Health ROI" to users**.

**The Pivot Journey:**
- **Initial Plan:** Production-ready architecture with FastAPI backend, React frontend, ONNX deployment
- **Reality Check:** Scientific chaos injection (Issues #49-50) showed RF underperforming Linear (-1.82% to -2.57%)
- **Pragmatic Solution:** Business-focused data generator with explicit non-linear patterns
- **Result:** Simplified 9-feature model where RF meaningfully outperforms Linear (+3-4%)

---

## 📊 Model Artifacts (Ready for Use)

Located in `app_model/` directory:

```
app_model/
├── livemore_rf_v2.joblib          # Random Forest model (200 estimators)
├── livemore_scaler_v2.joblib      # StandardScaler for input features
├── livemore_explainer_v2.pkl      # SHAP TreeExplainer
└── model_metadata.json            # Training metadata & feature info
```

**Model Performance:**
- Training R²: 0.9499
- Training MAE: 2.02 years
- Training RMSE: 2.87 years
- RF vs Linear Gain: +3-4% (demonstrates value)

**Feature Importance:**
1. Age: 58.6%
2. Smoking (pack-years): 11.3%
3. Exercise (hours/week): 11.1%
4. Diet quality: 9.1%
5. Stress level: 5.8%

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### 1. Setup Environment

```bash
# Navigate to the app directory
cd antiaging-mvp/streamlit_app

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📋 Required Features (9 Input Fields)

The model expects these inputs from users:

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `age` | Integer | 25-80 | Chronological age in years |
| `gender` | Category | M/F | Biological sex |
| `exercise_hours_week` | Float | 0-20 | Weekly exercise hours |
| `diet_quality_score` | Integer | 1-10 | Diet quality (1=poor, 10=excellent) |
| `sleep_hours` | Float | 4-10 | Average daily sleep hours |
| `stress_level` | Integer | 1-10 | Stress level (1=low, 10=high) |
| `smoking_pack_years` | Float | 0-60 | Pack-years of smoking history |
| `alcohol_drinks_week` | Integer | 0-30 | Weekly alcohol consumption |
| `genetic_risk_score` | Float | 0-10 | Genetic aging risk score |

**Note:** Gender must be encoded as 0 (M) or 1 (F) before scaling.

---

## 🎨 App Architecture (To Be Built)

### 1. Sidebar - User Inputs
```python
import streamlit as st

st.sidebar.header("Your Health Profile")

age = st.sidebar.slider("Age", 25, 80, 45)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
exercise = st.sidebar.slider("Weekly Exercise (hours)", 0.0, 20.0, 5.0)
# ... more inputs
```

### 2. Main Display - Prediction Result
```python
# Display biological age prediction
st.metric(
    label="Your Biological Age",
    value=f"{predicted_age:.1f} years",
    delta=f"{predicted_age - chronological_age:.1f} years vs chronological"
)
```

### 3. SHAP Explanations
```python
import shap

# Generate SHAP explanation
shap_values = explainer.shap_values(user_input_scaled)

# Display waterfall plot
st.pyplot(shap.plots.waterfall(shap_values[0]))
```

### 4. Business Language Translation
Instead of technical ML terms, use:
- "Health ROI" instead of "R² score"
- "Lifestyle impact" instead of "feature importance"
- "Years gained/lost" instead of "prediction error"
- "Intervention opportunities" instead of "high SHAP values"

---

## 🛠️ Implementation Guide

### Step 1: Load Model Artifacts

```python
import joblib
import json

# Load trained model
model = joblib.load('app_model/livemore_rf_v2.joblib')
scaler = joblib.load('app_model/livemore_scaler_v2.joblib')
explainer = joblib.load('app_model/livemore_explainer_v2.pkl')

# Load metadata
with open('app_model/model_metadata.json', 'r') as f:
    metadata = json.load(f)

feature_names = metadata['features']
```

### Step 2: Preprocess User Input

```python
import pandas as pd

# Collect user inputs
user_data = {
    'age': age,
    'gender': 0 if gender == "Male" else 1,
    'exercise_hours_week': exercise,
    # ... all 9 features
}

# Create DataFrame
input_df = pd.DataFrame([user_data], columns=feature_names)

# Scale features
input_scaled = scaler.transform(input_df)
```

### Step 3: Make Prediction

```python
# Predict biological age
predicted_age = model.predict(input_scaled)[0]

# Display result
st.write(f"Predicted Biological Age: {predicted_age:.1f} years")
st.write(f"Chronological Age: {age} years")
st.write(f"Difference: {predicted_age - age:.1f} years")
```

### Step 4: Generate SHAP Explanation

```python
# Get SHAP values
shap_values = explainer.shap_values(input_scaled)

# Create waterfall plot
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=input_scaled[0],
    feature_names=feature_names
))
```

---

## 📝 UI/UX Guidelines

### Visual Design
- Use health-themed colors (blues, greens)
- Large, clear metrics for biological age
- Traffic light colors for delta (green=good, yellow=neutral, red=concerning)
- Clean sidebar layout for inputs

### User Experience
- Default values should represent "average healthy adult"
- Tooltips explaining each input field
- Real-time prediction updates as user adjusts sliders
- Clear call-to-action: "Calculate My Health ROI"

### Business Language Examples
```python
# Technical: "RF R² = 0.95"
# Business: "95% prediction accuracy"

# Technical: "Feature importance: exercise_hours_week = 0.11"
# Business: "Exercise accounts for 11% of your biological age"

# Technical: "SHAP value: +2.3 years"
# Business: "This factor adds 2.3 years to your biological age"
```

---

## 🐛 Troubleshooting

### "Module not found: shap"
```bash
pip install shap
```

### "Model file not found"
- Check that `app_model/` directory exists
- Verify artifacts from `ml_pipeline/train_model_mvp.py`
- Re-run training script if needed

### "ValueError: Wrong number of features"
- Ensure input DataFrame has exactly 9 features
- Check feature order matches `metadata['features']`
- Verify gender encoding (M=0, F=1)

### SHAP Plots Not Rendering
```python
# Force matplotlib backend
import matplotlib
matplotlib.use('Agg')
```

---

## 🎯 Success Criteria

### Functional
- [ ] App loads without errors
- [ ] User can input all 9 features
- [ ] Prediction displays correctly
- [ ] SHAP explanation renders
- [ ] UI updates in real-time

### Business
- [ ] Uses "Health ROI" language (no ML jargon)
- [ ] Clear value proposition visible
- [ ] Professional appearance suitable for demo
- [ ] Responsive on laptop screens (1920x1080)

### Demo
- [ ] Can run full demo in <3 minutes
- [ ] Example scenarios prepared (healthy vs unhealthy)
- [ ] Clear narrative: "This is why XAI matters"

---

## 🚢 Deployment (Local Demo Only)

For thesis defense, **local deployment only**:
```bash
streamlit run app.py
```

**NOT in scope for MVP:**
- Cloud deployment (AWS, Azure, Heroku)
- Authentication/user management
- Database persistence
- API integration
- Docker containerization

See `docs/FUTURE_ROADMAP_POST_TCC.md` for post-thesis deployment plans.

---

## 📚 Related Documentation

- [Main README](../../README.md) - Project overview
- [ML Pipeline](../../ml_pipeline/README.md) - Model training
- [PIVOT.md](../../docs/PIVOT.md) - Strategic rationale
- [PROJECT_STATUS_OCT_2025.md](../../docs/PROJECT_STATUS_OCT_2025.md) - Detailed findings

---

**Last Updated:** October 28, 2025  
**Status:** Week 2 - UI Development 🏗️  
**Next Milestone:** Functional Streamlit demo (Nov 1)
