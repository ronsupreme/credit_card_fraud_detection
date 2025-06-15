"""
Model configuration and hyperparameter settings
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from typing import Dict, Any

class ModelConfigs:
    """Model configurations and hyperparameters"""
    
    @staticmethod
    def get_default_models() -> Dict[str, Any]:
        """Get default model configurations"""
        return {
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                algorithm='auto'
            ),
            
            'logistic': LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='liblinear'
            ),
            
            'svm': SVC(
                probability=True,
                random_state=42,
                kernel='rbf',
                C=1.0
            ),
            
            'decision_tree': DecisionTreeClassifier(
                random_state=42,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1
            ),
            
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                n_jobs=-1
            ),
            
            'gradient_boost': GradientBoostingClassifier(
                random_state=42,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3
            ),
            
            'naive_bayes': GaussianNB(),
            
            'ada_boost': AdaBoostClassifier(
                random_state=42,
                n_estimators=100,
                learning_rate=1.0
            )
        }
    
    @staticmethod
    def get_hyperparameter_grids() -> Dict[str, Dict[str, list]]:
        """Get hyperparameter grids for grid search"""
        return {
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree']
            },
            
            'logistic': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'penalty': ['l1', 'l2']
            },
            
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            },
            
            'decision_tree': {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 6],
                'criterion': ['gini', 'entropy']
            },
            
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            
            'gradient_boost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        }