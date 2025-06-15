"""
Main execution script for credit card fraud detection
"""
import logging
import pandas as pd
import os
from src.data_loader import DataLoader
from src.data_preprocessing import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.visualization import FraudVisualization
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)

def main():
    """Main execution function"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Credit Card Fraud Detection Pipeline")
    
    # Configuration
    DATA_PATH = "data/raw/creditcard.csv"
    MODELS_PATH = "models/saved_models/"
    RESULTS_PATH = "results/"
    
    try:
        # 1. Load Data
        logger.info("Step 1: Loading data")
        data_loader = DataLoader(DATA_PATH)
        df = data_loader.load_data()
        
        # Display basic info
        data_info = data_loader.get_data_info(df)
        logger.info(f"Dataset shape: {data_info['shape']}")
        logger.info(f"Class distribution: {data_info['class_distribution']}")
        
        # Split features and target
        X, y = data_loader.split_features_target(df)
        
        # 2. Data Visualization (Exploratory Analysis)
        logger.info("Step 2: Creating visualizations")
        visualizer = FraudVisualization()
        
        # Create exploratory plots
        visualizer.plot_class_distribution(df)
        visualizer.plot_amount_distribution(df)
        visualizer.plot_time_analysis(df)
        visualizer.plot_correlation_matrix(df)
        
        # 3. Data Preprocessing
        logger.info("Step 3: Preprocessing data")
        preprocessor = DataPreprocessor(test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(X, y)
        
        # Save preprocessor scaler
        preprocessor.save_scaler("models/scaler.pkl")
        
        # Save processed data
        os.makedirs("data/processed", exist_ok=True)
        X_train.to_csv("data/processed/X_train.csv", index=False)
        X_test.to_csv("data/processed/X_test.csv", index=False)
        y_train.to_csv("data/processed/y_train.csv", index=False)
        y_test.to_csv("data/processed/y_test.csv", index=False)
        
        # 4. Model Training
        logger.info("Step 4: Training models")
        trainer = ModelTrainer(MODELS_PATH)
        trained_models = trainer.train_all_models(X_train, y_train)
        
        # Save all models
        trainer.save_all_models()
        logger.info(f"Trained and saved {len(trained_models)} models")
        
        # 5. Model Evaluation
        logger.info("Step 5: Evaluating models")
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_all_models(trained_models, X_test, y_test)
        
        # Print results summary
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY")
        print("="*80)
        
        metrics_summary = evaluator.get_metrics_summary()
        print(metrics_summary.to_string(index=False))
        
        # Get best model
        best_model_name, best_results = evaluator.get_best_model('f1_score')
        print(f"\nBest Model (by F1-Score): {best_model_name}")
        print(f"F1-Score: {best_results['f1_score']:.4f}")
        
        # Print detailed results for best model
        evaluator.print_detailed_results(best_model_name)
        
        # 6. Advanced Visualizations
        logger.info("Step 6: Creating evaluation visualizations")
        
        # Plot comparison charts
        visualizer.plot_roc_curves(evaluation_results)
        visualizer.plot_precision_recall_curves(evaluation_results)
        visualizer.plot_confusion_matrices(evaluation_results)
        visualizer.plot_metrics_comparison(evaluation_results)
        
        # 7. Save Results
        logger.info("Step 7: Saving results")
        os.makedirs(f"{RESULTS_PATH}/reports", exist_ok=True)
        
        # Save metrics summary
        metrics_summary.to_csv(f"{RESULTS_PATH}/reports/model_comparison.csv", index=False)
        
        # Save detailed results
        results_df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc'],
                'PR-AUC': results['pr_auc']
            }
            for name, results in evaluation_results.items()
        ])
        
        results_df.to_csv(f"{RESULTS_PATH}/reports/detailed_results.csv", index=False)
        
        logger.info("Credit Card Fraud Detection Pipeline completed successfully!")
        print(f"\nResults saved to: {RESULTS_PATH}")
        print(f"Models saved to: {MODELS_PATH}")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()