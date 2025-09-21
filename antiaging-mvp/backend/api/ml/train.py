"""
Model Training Utility for Anti-Aging ML Application

This module provides training functionality for machine learning models, with current support for Random Forests.
It implements proper data splitting to prevent data leakage and includes ONNX export capabilities for production deployment.
The design allows for easy extension to support additional model types in the future.

Usage:
    trainer = ModelTrainer()
    results = trainer.train(data_path="path/to/data.csv")
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import onnx
import onnxmltools
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib
import os
from .preprocessor import DataPreprocessor
from ..data.generator import generate_synthetic_data


class ModelTrainer:
    """Train and export anti-aging ML model"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.feature_names = []
        
    def train(self, data_path=None, test_size=0.2, tune_hyperparameters=True):
        """Train the aging prediction model - FIXED: No data leakage"""
        
        # Load or generate data
        if data_path and os.path.exists(data_path):
            data = pd.read_csv(data_path)
        else:
            print("Generating synthetic training data...")
            data = generate_synthetic_data(n_samples=5000)
        
        # Separate features and target
        target_col = 'biological_age'
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # CRITICAL FIX: Split data FIRST, then preprocess to prevent data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Preprocess features (fit ONLY on training data)
        self.preprocessor = DataPreprocessor()
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)  # Only transform, don't fit!
        self.feature_names = X_train_processed.columns.tolist()
        
        # Train model
        if tune_hyperparameters:
            self.model = self._tune_hyperparameters(X_train_processed, y_train)
        else:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train_processed, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_processed, y_train)
        test_score = self.model.score(X_test_processed, y_test)
        
        y_pred = self.model.predict(X_test_processed)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Training R²: {train_score:.4f}")
        print(f"Test R²: {test_score:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test R² (explicit): {r2:.4f}")
        
        return {
            'train_r2': train_score,
            'test_r2': test_score,
            'test_mae': mae,
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
    
    def _tune_hyperparameters(self, X_train, y_train):
        """Hyperparameter tuning with GridSearchCV"""
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def save_model(self, model_dir='models'):
        """Save trained model and preprocessor"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save sklearn model
        model_path = os.path.join(model_dir, 'aging_model.joblib')
        joblib.dump(self.model, model_path)
        
        # Save preprocessor
        preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
        self.preprocessor.save(preprocessor_path)
        
        print(f"Model saved to {model_path}")
        print(f"Preprocessor saved to {preprocessor_path}")
        
        return model_path, preprocessor_path
    
    def export_to_onnx(self, model_dir='models'):
        """Export model to ONNX format for production"""
        if self.model is None:
            raise ValueError("Model must be trained before exporting")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Define input shape (number of features)
        n_features = len(self.feature_names)
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        
        # Convert to ONNX
        onnx_model = convert_sklearn(
            self.model,
            initial_types=initial_type,
            target_opset=11
        )
        
        # Save ONNX model
        onnx_path = os.path.join(model_dir, 'aging_model.onnx')
        with open(onnx_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"ONNX model exported to {onnx_path}")
        return onnx_path
    
    def load_model(self, model_path, preprocessor_path):
        """Load trained model and preprocessor"""
        self.model = joblib.load(model_path)
        self.preprocessor.load(preprocessor_path)
        
        # Get feature names from a dummy prediction
        dummy_data = pd.DataFrame([{f'feature_{i}': 0 for i in range(10)}])
        processed = self.preprocessor.transform(dummy_data)
        self.feature_names = processed.columns.tolist()


def main():
    """Main training script"""
    trainer = ModelTrainer()
    
    # Train model
    results = trainer.train(tune_hyperparameters=True)
    
    # Save model
    model_path, preprocessor_path = trainer.save_model()
    
    # Export to ONNX
    onnx_path = trainer.export_to_onnx()
    
    print("Training completed successfully!")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()

