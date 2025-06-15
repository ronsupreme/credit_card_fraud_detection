"""
Model training utilities for credit card fraud detection
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from typing import Dict, List, Tuple, Any
import logging
import joblib
import os

class ModelTrainer:
    """Class to handle model training operations"""
    
    def __init__(self, models_save_path: str = "models/saved_models/"):
        self.models_save_path = models_save_path
        self.trained_models = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize model configurations
        self.model_configs = {
            'knn': KNeighborsClassifier(n_neighbors=5),
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingClassifier(random_state=42)
        }
    
    def get_model_by_name(self, model_name: str):
        """Get model instance by name"""
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.model_configs.keys())}")
        return self.model_configs[model_name]
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
        """Train a single model"""
        model = self.get_model_by_name(model_name)
        
        self.logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        self.trained_models[model_name] = model
        self.logger.info(f"{model_name} training completed")
        
        return model
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train all configured models"""
        self.logger.info("Starting training for all models...")
        
        for model_name in self.model_configs.keys():
            try:
                self.train_single_model(model_name, X_train, y_train)
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
        
        self.logger.info(f"Training completed for {len(self.trained_models)} models")
        return self.trained_models
    
    def save_model(self, model_name: str, model=None):
        """Save a trained model"""
        if model is None:
            model = self.trained_models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model {model_name} not found or not trained")
        
        os.makedirs(self.models_save_path, exist_ok=True)
        filepath = os.path.join(self.models_save_path, f"{model_name}.pkl")
        
        joblib.dump(model, filepath)
        self.logger.info(f"Model {model_name} saved to {filepath}")
    
    def save_all_models(self):
        """Save all trained models"""
        for model_name in self.trained_models.keys():
            self.save_model(model_name)
    
    def load_model(self, model_name: str):
        """Load a saved model"""
        filepath = os.path.join(self.models_save_path, f"{model_name}.pkl")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        self.trained_models[model_name] = model
        self.logger.info(f"Model {model_name} loaded from {filepath}")
        
        return model
    
    def get_model_names(self) -> List[str]:
        """Get list of available model names"""
        return list(self.model_configs.keys())