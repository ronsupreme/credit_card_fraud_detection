"""
Data loading utilities for credit card fraud detection
"""
import pandas as pd
import numpy as np
from typing import Tuple
import logging

class DataLoader:
    """Class to handle data loading operations"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> pd.DataFrame:
        """Load the credit card fraud dataset"""
        try:
            df = pd.read_csv(self.data_path)
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            self.logger.error(f"File not found: {self.data_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """Get basic information about the dataset"""
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'class_distribution': df['Class'].value_counts().to_dict()
        }
        return info
    
    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split dataset into features and target"""
        X = df.drop('Class', axis=1)
        y = df['Class']
        return X, y