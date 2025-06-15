"""
Model evaluation utilities for credit card fraud detection
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score
)
from typing import Dict, List, Tuple, Any
import logging

class ModelEvaluator:
    """Class to handle model evaluation operations"""

    def __init__(self):
        self.evaluation_results = {}
        self.logger = logging.getLogger(__name__)

    def evaluate_single_model(self, model, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate a single model and return metrics"""
        self.logger.info(f"Evaluating {model_name}...")

        # Make predictions
        y_pred = model.predict(X_test)

        # Get prediction probabilities or decision scores
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            y_score = y_pred

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        # Precision-Recall curve and AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)
        pr_auc = average_precision_score(y_test, y_score)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        # Calculate the classification report here and store the dictionary output
        class_report_dict = classification_report(y_test, y_pred, output_dict=True)
        # Also generate the text version for easy printing later if needed, or just print from dict
        class_report_text = classification_report(y_test, y_pred, output_dict=False)


        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm,
            'classification_report_dict': class_report_dict, # Store the dictionary
            'classification_report_text': class_report_text, # Store the text version
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'y_pred': y_pred,
            # Note: y_test is not stored here, but it was used to generate the metrics
        }

        self.evaluation_results[model_name] = results
        self.logger.info(f"{model_name} evaluation completed")

        return results

    def evaluate_all_models(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Evaluate all models"""
        self.logger.info("Starting evaluation for all models...")

        for model_name, model in models.items():
            try:
                # Pass y_test to evaluate_single_model
                self.evaluate_single_model(model, model_name, X_test, y_test)
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")

        self.logger.info(f"Evaluation completed for {len(self.evaluation_results)} models")
        return self.evaluation_results

    def get_metrics_summary(self) -> pd.DataFrame:
        """Get summary of key metrics for all models"""
        summary_data = []

        for model_name, results in self.evaluation_results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc'],
                'PR-AUC': results['pr_auc']
            })

        return pd.DataFrame(summary_data).round(4)

    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, Dict[str, Any]]:
        """Get the best performing model based on specified metric"""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")

        # Ensure the metric exists in the results before finding the max
        if metric not in list(self.evaluation_results.values())[0]:
             raise ValueError(f"Metric '{metric}' not found in evaluation results.")

        best_model_name = max(self.evaluation_results.keys(),
                             key=lambda x: self.evaluation_results[x][metric])

        return best_model_name, self.evaluation_results[best_model_name]

    def print_detailed_results(self, model_name: str = None):
        """Print detailed results for a specific model or all models"""
        if model_name:
            models_to_print = [model_name] if model_name in self.evaluation_results else []
            if not models_to_print:
                print(f"Model '{model_name}' not found in evaluation results.")
                return
        else:
            models_to_print = list(self.evaluation_results.keys())
            if not models_to_print:
                 print("No evaluation results available to print.")
                 return


        for name in models_to_print:
            results = self.evaluation_results[name]
            print(f"\n{'='*50}")
            print(f"Model: {name}")
            print(f"{'='*50}")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"F1-Score: {results['f1_score']:.4f}")
            print(f"ROC-AUC: {results['roc_auc']:.4f}")
            print(f"PR-AUC: {results['pr_auc']:.4f}")
            print(f"\nConfusion Matrix:")
            print(results['confusion_matrix'])
            print(f"\nClassification Report:")
            # Use the pre-calculated text report
            print(results['classification_report_text'])

