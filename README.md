# Credit Card Fraud Detection

A comprehensive machine learning project for detecting fraudulent credit card transactions using multiple algorithms and evaluation metrics.

## Project Structure
credit_card_fraud_detection/
.
├── config
│   └── config.yaml
├── data
│   ├── processed
│   └── raw
├── fraud_detection.log
├── models
│   ├── model_configs.py
│   ├── saved_models
│   └── scaler.pkl
├── myevn
│   ├── bin
│   ├── etc
│   ├── include
│   ├── lib
│   ├── pyvenv.cfg
│   └── share
├── requirements.txt
├── results
│   ├── models
│   ├── plots
│   └── reports
├── run.py
├── src
│   ├── __init__.py
│   ├── __pycache__
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── model_evaluator.py
│   ├── model_trainer.py
│   └── visualization.py
└── tests
    ├── __init__.py
    ├── __pycache__
    ├── test_data_loader.py
    ├── test_models.py
    └── test_preprocessing.py

## Features

- **Multiple ML Algorithms**: KNN, Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting
- **Comprehensive Evaluation**: ROC-AUC, Precision-Recall, Confusion Matrix, F1-Score
- **Advanced Visualizations**: Interactive plots for data exploration and model comparison
- **Modular Design**: Clean, maintainable code structure
- **Robust Testing**: Unit tests for core functionality
- **Configuration Management**: YAML-based configuration
