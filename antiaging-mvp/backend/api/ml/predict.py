"""
Prediction Module for Anti-Aging ML Application

Provides functionality for loading trained models and making predictions
on user data with explanations using SHAP.
"""

import numpy as np
import pandas as pd
import onnxruntime as ort
import shap
import joblib
import os
from django.conf import settings
from .preprocessor import DataPreprocessor, prepare_user_data


class MLPredictor:
    """ML prediction service with SHAP explanations"""
    
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(settings.BASE_DIR, 'api', 'ml', 'models')
        
        self.model_dir = model_dir
        self.onnx_session = None
        self.sklearn_model = None
        self.preprocessor = DataPreprocessor()
        self.shap_explainer = None
        self.model_version = "1.0.0"
        
        self._load_models()
    
    def _load_models(self):
        """Load ONNX model and preprocessor"""
        try:
            # Load ONNX model for inference
            onnx_path = os.path.join(self.model_dir, 'aging_model.onnx')
            if os.path.exists(onnx_path):
                self.onnx_session = ort.InferenceSession(onnx_path)
                print("ONNX model loaded successfully")
            
            # Load sklearn model for SHAP explanations
            sklearn_path = os.path.join(self.model_dir, 'aging_model.joblib')
            if os.path.exists(sklearn_path):
                self.sklearn_model = joblib.load(sklearn_path)
                print("Sklearn model loaded successfully")
            
            # Load preprocessor
            preprocessor_path = os.path.join(self.model_dir, 'preprocessor.joblib')
            if self.preprocessor.load(preprocessor_path):
                print("Preprocessor loaded successfully")
            else:
                print("Warning: Could not load preprocessor")
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            # Use dummy model for development
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create dummy model for development/testing"""
        print("Using dummy model for development")
        self.sklearn_model = None
        self.onnx_session = None
    
    def predict_aging(self, user):
        """Predict biological age for a user"""
        try:
            # Prepare user data
            user_data = prepare_user_data(
                user.profile if hasattr(user, 'profile') else None,
                user.genetic_profile if hasattr(user, 'genetic_profile') else None,
                user.habits.first() if hasattr(user, 'habits') else None
            )
            
            # Preprocess data
            if self.preprocessor.feature_columns:
                processed_data = self.preprocessor.transform(user_data)
            else:
                processed_data = user_data
            
            # Make prediction
            if self.onnx_session:
                prediction = self._predict_with_onnx(processed_data)
            else:
                prediction = self._predict_dummy(user_data)
            
            # Calculate derived metrics
            chronological_age = user.profile.age if hasattr(user, 'profile') else 35
            aging_rate = prediction / chronological_age
            
            # Generate SHAP explanations
            shap_values = self._get_shap_explanations(processed_data)
            
            # Calculate confidence (simplified)
            confidence_score = min(0.95, max(0.5, 1.0 - abs(prediction - chronological_age) / 20))
            
            return {
                'biological_age': round(float(prediction), 2),
                'chronological_age': float(chronological_age),
                'aging_rate': round(float(aging_rate), 3),
                'confidence_score': round(float(confidence_score), 3),
                'shap_values': shap_values,
                'model_version': self.model_version
            }
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            # Return fallback prediction
            return self._fallback_prediction(user)
    
    def _predict_with_onnx(self, data):
        """Make prediction using ONNX model"""
        input_name = self.onnx_session.get_inputs()[0].name
        input_data = data.values.astype(np.float32)
        result = self.onnx_session.run(None, {input_name: input_data})
        return result[0][0]
    
    def _predict_dummy(self, data):
        """Dummy prediction for development"""
        # Simple heuristic based on age and habits
        base_age = 35  # Default if no profile
        
        if not data.empty:
            # Use available data for prediction
            if 'age' in data.columns:
                base_age = data['age'].iloc[0]
            
            # Adjust based on habits
            adjustment = 0
            if 'exercise_frequency' in data.columns:
                adjustment -= (data['exercise_frequency'].iloc[0] - 3) * 0.5
            if 'stress_level' in data.columns:
                adjustment += (data['stress_level'].iloc[0] - 5) * 0.3
            if 'smoking' in data.columns:
                adjustment += data['smoking'].iloc[0] * 2
            
            prediction = base_age + adjustment + np.random.normal(0, 1)
        else:
            prediction = base_age + np.random.normal(0, 2)
        
        return max(18, prediction)  # Minimum age of 18
    
    def _get_shap_explanations(self, data):
        """Generate SHAP explanations for the prediction"""
        try:
            if self.sklearn_model and len(data) > 0:
                # Initialize SHAP explainer if not already done
                if self.shap_explainer is None:
                    self.shap_explainer = shap.TreeExplainer(self.sklearn_model)
                
                # Calculate SHAP values
                shap_values = self.shap_explainer.shap_values(data)
                
                # Convert to dictionary with feature names
                feature_names = data.columns.tolist()
                shap_dict = {}
                
                for i, feature in enumerate(feature_names):
                    shap_dict[feature] = float(shap_values[0][i]) if len(shap_values.shape) > 1 else float(shap_values[i])
                
                return shap_dict
            else:
                # Return dummy SHAP values
                return self._dummy_shap_values(data)
                
        except Exception as e:
            print(f"SHAP calculation error: {str(e)}")
            return self._dummy_shap_values(data)
    
    def _dummy_shap_values(self, data):
        """Generate dummy SHAP values for development"""
        if data.empty:
            return {}
        
        dummy_shap = {}
        for col in data.columns:
            dummy_shap[col] = np.random.uniform(-1, 1)
        
        return dummy_shap
    
    def _fallback_prediction(self, user):
        """Fallback prediction when model fails"""
        chronological_age = 35
        if hasattr(user, 'profile'):
            chronological_age = user.profile.age
        
        biological_age = chronological_age + np.random.normal(0, 3)
        
        return {
            'biological_age': round(max(18, biological_age), 2),
            'chronological_age': float(chronological_age),
            'aging_rate': round(biological_age / chronological_age, 3),
            'confidence_score': 0.5,
            'shap_values': {},
            'model_version': f"{self.model_version}-fallback"
        }


class BatchPredictor:
    """Batch prediction service for multiple users"""
    
    def __init__(self):
        self.predictor = MLPredictor()
    
    def predict_batch(self, users):
        """Predict for multiple users"""
        results = []
        
        for user in users:
            try:
                prediction = self.predictor.predict_aging(user)
                results.append({
                    'user_id': user.id,
                    'prediction': prediction
                })
            except Exception as e:
                results.append({
                    'user_id': user.id,
                    'error': str(e)
                })
        
        return results
    
#PLACEHOLDER CODE #2
        """
        import onnxruntime as rt
import shap
import numpy as np
import joblib
from .preprocessor import get_preprocessor

def predict_with_explain(input_df):
    # Load model for SHAP (ONNX for inference)
    sess = rt.InferenceSession('api/ml/model.onnx')
    model = joblib.load('api/ml/model.pkl')  # For SHAP
    preprocessor = get_preprocessor(input_df)

    # Preprocess
    preprocessed = preprocessor.fit_transform(input_df)  # Fit on input for consistency, though trained on large data

    # Inference
    input_name = sess.get_inputs()[0].name
    pred = sess.run(None, {input_name: preprocessed.astype(np.float32)})[0]

    # SHAP
    explainer = shap.TreeExplainer(model.named_steps['clf'])
    shap_values = explainer.shap_values(preprocessed)
    # Simplify explanations (average over classes if multi-class)
    explanations = {col: np.mean(val) for col, val in zip(input_df.columns, shap_values[1][0])}  # Assume class 1 for medium/high

    return pred, explanations
        """