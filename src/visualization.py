"""
Visualization utilities for credit card fraud detection
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import os

class FraudVisualization:
    """Class to handle visualization operations"""
    
    def __init__(self, save_path: str = "results/plots/"):
        self.save_path = save_path
        self.logger = logging.getLogger(__name__)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_class_distribution(self, df: pd.DataFrame, save: bool = True) -> None:
        """Plot the distribution of fraud vs non-fraud cases"""
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='Class')
        plt.title("Fraud vs Non-Fraud Distribution", fontsize=16, fontweight='bold')
        plt.xlabel("Class (0: Non-Fraud, 1: Fraud)", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        
        # Add value labels on bars
        ax = plt.gca()
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width()/2., p.get_height()),
                       ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        
        if save:
            self._save_plot("class_distribution.png")
        
        plt.show()
    
    def plot_amount_distribution(self, df: pd.DataFrame, save: bool = True) -> None:
        """Plot transaction amount distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall distribution
        axes[0, 0].hist(df['Amount'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title("Overall Amount Distribution", fontweight='bold')
        axes[0, 0].set_xlabel("Amount")
        axes[0, 0].set_ylabel("Frequency")
        
        # Log scale distribution
        axes[0, 1].hist(np.log1p(df['Amount']), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title("Amount Distribution (Log Scale)", fontweight='bold')
        axes[0, 1].set_xlabel("Log(Amount + 1)")
        axes[0, 1].set_ylabel("Frequency")
        
        # Box plot by class
        sns.boxplot(data=df, x='Class', y='Amount', ax=axes[1, 0])
        axes[1, 0].set_title("Amount by Class", fontweight='bold')
        axes[1, 0].set_xlabel("Class (0: Non-Fraud, 1: Fraud)")
        
        # Violin plot by class
        sns.violinplot(data=df, x='Class', y='Amount', ax=axes[1, 1])
        axes[1, 1].set_title("Amount Distribution by Class", fontweight='bold')
        axes[1, 1].set_xlabel("Class (0: Non-Fraud, 1: Fraud)")
        
        plt.tight_layout()
        
        if save:
            self._save_plot("amount_distribution.png")
        
        plt.show()
    
    def plot_time_analysis(self, df: pd.DataFrame, save: bool = True) -> None:
        """Plot time-based analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time distribution
        axes[0, 0].hist(df['Time'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 0].set_title("Time Distribution", fontweight='bold')
        axes[0, 0].set_xlabel("Time (seconds)")
        axes[0, 0].set_ylabel("Frequency")
        
        # Time vs Amount scatter
        sample_df = df.sample(n=5000) if len(df) > 5000 else df
        scatter = axes[0, 1].scatter(sample_df['Time'], sample_df['Amount'], 
                                   c=sample_df['Class'], alpha=0.6, cmap='RdYlBu')
        axes[0, 1].set_title("Time vs Amount (Colored by Class)", fontweight='bold')
        axes[0, 1].set_xlabel("Time (seconds)")
        axes[0, 1].set_ylabel("Amount")
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # Time distribution by class
        for class_val in [0, 1]:
            class_data = df[df['Class'] == class_val]['Time']
            axes[1, 0].hist(class_data, bins=50, alpha=0.6, 
                           label=f'Class {class_val}', density=True)
        axes[1, 0].set_title("Time Distribution by Class", fontweight='bold')
        axes[1, 0].set_xlabel("Time (seconds)")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].legend()
        
        # Hour of day analysis (assuming Time is seconds from start)
        df_copy = df.copy()
        df_copy['Hour'] = (df_copy['Time'] // 3600) % 24
        hour_fraud = df_copy.groupby('Hour')['Class'].agg(['count', 'sum']).reset_index()
        hour_fraud['fraud_rate'] = hour_fraud['sum'] / hour_fraud['count']
        
        axes[1, 1].bar(hour_fraud['Hour'], hour_fraud['fraud_rate'], alpha=0.7, color='orange')
        axes[1, 1].set_title("Fraud Rate by Hour", fontweight='bold')
        axes[1, 1].set_xlabel("Hour of Day")
        axes[1, 1].set_ylabel("Fraud Rate")
        
        plt.tight_layout()
        
        if save:
            self._save_plot("time_analysis.png")
        
        plt.show()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, save: bool = True) -> None:
        """Plot correlation matrix for features"""
        # Select a subset of features for readability
        features_to_plot = ['Time', 'Amount'] + [col for col in df.columns if col.startswith('V')][:10] + ['Class']
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[features_to_plot].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title("Feature Correlation Matrix", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            self._save_plot("correlation_matrix.png")
        
        plt.show()
    
    def plot_roc_curves(self, evaluation_results: Dict[str, Dict[str, Any]], save: bool = True) -> None:
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in evaluation_results.items():
            plt.plot(results['fpr'], results['tpr'], 
                    label=f"{model_name} (AUC = {results['roc_auc']:.3f})", 
                    linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            self._save_plot("roc_curves.png")
        
        plt.show()
    
    def plot_precision_recall_curves(self, evaluation_results: Dict[str, Dict[str, Any]], save: bool = True) -> None:
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in evaluation_results.items():
            plt.plot(results['recall_curve'], results['precision_curve'], 
                    label=f"{model_name} (AUC = {results['pr_auc']:.3f})", 
                    linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve Comparison', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            self._save_plot("precision_recall_curves.png")
        
        plt.show()
    
    def plot_confusion_matrices(self, evaluation_results: Dict[str, Dict[str, Any]], save: bool = True) -> None:
        """Plot confusion matrices for all models"""
        n_models = len(evaluation_results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, results) in enumerate(evaluation_results.items()):
            row, col = idx // cols, idx % cols
            
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                       cmap='Blues', ax=axes[row, col])
            axes[row, col].set_title(f'{model_name}', fontweight='bold')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            self._save_plot("confusion_matrices.png")
        
        plt.show()
    
    def plot_metrics_comparison(self, evaluation_results: Dict[str, Dict[str, Any]], save: bool = True) -> None:
        """Plot comparison of key metrics across models"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = list(evaluation_results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            values = [evaluation_results[model][metric] for model in model_names]
            
            bars = axes[idx].bar(model_names, values, alpha=0.7)
            axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            axes[idx].set_ylabel('Score')
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                             f'{value:.3f}', ha='center', va='bottom')
        
        # Overall comparison (radar chart style)
        axes[5].remove()
        ax_radar = fig.add_subplot(2, 3, 6, projection='polar')
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for model_name in model_names:
            values = [evaluation_results[model_name][metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax_radar.fill(angles, values, alpha=0.1)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics])
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Overall Model Comparison', fontweight='bold', pad=20)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        if save:
            self._save_plot("metrics_comparison.png")
        
        plt.show()
    
    def _save_plot(self, filename: str) -> None:
        """Save the current plot to file"""
        os.makedirs(self.save_path, exist_ok=True)
        filepath = os.path.join(self.save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        self.logger.info(f"Plot saved to {filepath}")