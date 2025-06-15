# Credit Card Fraud Detection

A comprehensive machine learning project for detecting fraudulent credit card transactions using multiple algorithms and evaluation metrics.

## Project Structure
```text
credit_card_fraud_detection/
├── config/
│   └── config.yaml               # Configuration file (e.g., model parameters)
├── data/
│   ├── processed/                # Processed datasets ready for modeling
│   └── raw/                      # Original raw data files
├── fraud_detection.log           # Log file for training and evaluation
├── models/
│   ├── model_configs.py          # Model configuration setup
│   ├── saved_models/             # Directory to store trained models
│   └── scaler.pkl                # Serialized scaler for data normalization
├── myevn/                        # Virtual environment folder (should be .gitignored)
│   ├── bin/
│   ├── etc/
│   ├── include/
│   ├── lib/
│   ├── pyvenv.cfg
│   └── share/
├── requirements.txt             # Project dependencies
├── results/
│   ├── models/                   # Exported trained model files
│   ├── plots/                    # Visualizations, graphs, ROC curves, etc.
│   └── reports/                  # Evaluation reports, metrics, and summaries
├── run.py                        # Main script to run training pipeline
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Load and summarize dataset
│   ├── data_preprocessing.py     # Data cleaning, scaling, and splitting
│   ├── model_evaluator.py        # Evaluation metrics and plots
│   ├── model_trainer.py          # Training logic for multiple models
│   └── visualization.py          # Exploratory data analysis and result visualization
└── tests/
    ├── __init__.py
    ├── test_data_loader.py       # Unit tests for data loading
    ├── test_models.py            # Unit tests for model training and prediction
    └── test_preprocessing.py     # Unit tests for preprocessing logic

## Features

- **Multiple ML Algorithms**: KNN, Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting
- **Comprehensive Evaluation**: ROC-AUC, Precision-Recall, Confusion Matrix, F1-Score
- **Advanced Visualizations**: Interactive plots for data exploration and model comparison
- **Modular Design**: Clean, maintainable code structure
- **Robust Testing**: Unit tests for core functionality
- **Configuration Management**: YAML-based configuration
