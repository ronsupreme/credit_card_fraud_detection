"""
Data preprocessing utilities for credit card fraud detection
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import logging
import joblib
import os

class DataPreprocessor:
    """Class to handle data preprocessing operations"""
    
    def __init__(self, test_size: float = 0.3, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      columns_to_scale: list = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale specified features using StandardScaler"""
        if columns_to_scale is None:
            columns_to_scale = ['Time', 'Amount']
        
        # Create copies to avoid modifying original data
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Fit scaler on training data and transform both sets
        X_train_scaled[columns_to_scale] = self.scaler.fit_transform(X_train[columns_to_scale])
        X_test_scaled[columns_to_scale] = self.scaler.transform(X_test[columns_to_scale])
        
        self.logger.info(f"Features scaled: {columns_to_scale}")
        return X_train_scaled, X_test_scaled
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets with stratification"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            stratify=y, 
            random_state=self.random_state
        )
        
        self.logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        self.logger.info(f"Class distribution in train: {y_train.value_counts().to_dict()}")
        self.logger.info(f"Class distribution in test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Complete preprocessing pipeline"""
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_scaler(self, filepath: str):
        """Save the fitted scaler"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.scaler, filepath)
        self.logger.info(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str):
        """Load a saved scaler"""
        self.scaler = joblib.load(filepath)
        self.logger.info(f"Scaler loaded from {filepath}")