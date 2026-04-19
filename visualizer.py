"""
Visualization module for Neonatal Health Risk Prediction
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import os

from config import VIZ_CONFIG


class Visualizer:
    """Handles all visualization tasks for model analysis"""
    
    def __init__(self, style=None):
        """
        Initialize the Visualizer
        
        Args:
            style (str): Matplotlib style to use
        """
        if style is None:
            style = VIZ_CONFIG['style']
        
        plt.style.use(style)
        self.fig_size = VIZ_CONFIG['figure_size']
        self.dpi = VIZ_CONFIG['dpi']
        self.color_palette = VIZ_CONFIG['color_palette']
        
    def plot_feature_importance(self, importance_df, model_name, top_n=10, save_path=None):
        """
        Plot feature importance
        
        Args:
            importance_df (pd.DataFrame): DataFrame with feature importance
            model_name (str): Name of the model
            top_n (int): Number of top features to show
            save_path (str): Path to save the plot
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # Take top N features
            plot_df = importance_df.head(top_n).copy()
            
            # Create horizontal bar plot
            colors = plt.cm.viridis(np.linspace(0, 1, len(plot_df)))
            bars = plt.barh(range(len(plot_df)), plot_df['importance'][::-1], 
                           color=colors[::-1])
            
            plt.yticks(range(len(plot_df)), plot_df['feature'][::-1])
            plt.xlabel('Importance Score', fontsize=12)
            plt.title(f'Top {top_n} Feature Importance - {model_name.replace("_", " ").title()}', 
                     fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                print(f"✅ Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Could not plot feature importance: {e}")
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, model_name, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        try:
            plt.figure(figsize=(8, 6))
            
            cm = confusion_matrix(y_true, y_pred)
            
            # Calculate percentages
            cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Create annotation text
            annot = np.array([f"{count}\n({percentage:.1f}%)" 
                             for count, percentage in zip(cm.flatten(), cm_percentage.flatten())]
                            ).reshape(cm.shape)
            
            sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=True,
                       xticklabels=class_names, yticklabels=class_names,
                       square=True, linewidths=1, linecolor='white')
            
            plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}', 
                     fontweight='bold', fontsize=14)
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                print(f"✅ Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Could not plot confusion matrix: {e}")
    
    def plot_roc_curve(self, y_true, y_pred_proba, class_names, model_name, save_path=None):
        """
        Plot ROC curve for binary classification
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            class_names: Names of classes
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        try:
            plt.figure(figsize=(8, 6))
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', 
                    linewidth=2, color='purple')
            plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5, linewidth=1)
            
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curve - {model_name.replace("_", " ").title()}', 
                     fontweight='bold', fontsize=14)
            plt.legend(loc='lower right')
            plt.grid(alpha=0.3)
            
            # Fill under the curve
            plt.fill_between(fpr, tpr, alpha=0.2, color='purple')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                print(f"✅ Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Could not plot ROC curve: {e}")
    
    def plot_model_comparison(self, evaluation_results, best_model_name=None, save_path=None):
        """
        Plot comparison of model performances
        
        Args:
            evaluation_results (dict): Dictionary of evaluation results
            best_model_name (str): Name of the best model
            save_path (str): Path to save the plot
        """
        try:
            plt.figure(figsize=(10, 6))
            
            models = list(evaluation_results.keys())
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
            
            # Prepare data
            data = []
            for model in models:
                row = [evaluation_results[model].get(m, 0) for m in metrics_to_plot]
                data.append(row)
            
            data = np.array(data)
            
            # Plot grouped bars
            x = np.arange(len(models))
            width = 0.2
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            for i, metric in enumerate(metrics_to_plot):
                bars = plt.bar(x + i*width, data[:, i], width, 
                              label=metric.upper(), color=colors[i], alpha=0.8)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Highlight best model
            if best_model_name and best_model_name in models:
                best_idx = models.index(best_model_name)
                plt.axvspan(best_idx - width*2, best_idx + width*3, 
                           alpha=0.2, color='green', label='Best Model')
            
            plt.xlabel('Models', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.title('Model Performance Comparison', fontweight='bold', fontsize=14)
            plt.xticks(x + width*1.5, [m.replace('_', ' ').title() for m in models], 
                      rotation=45, ha='right')
            plt.legend(loc='upper right')
            plt.grid(axis='y', alpha=0.3)
            plt.ylim([0, 1])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                print(f"✅ Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Could not plot model comparison: {e}")
    
    def plot_class_distribution(self, y_true, y_pred, class_names, model_name, save_path=None):
        """
        Plot class distribution comparison
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        try:
            plt.figure(figsize=(10, 6))
            
            unique_true, counts_true = np.unique(y_true, return_counts=True)
            unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
            
            x = np.arange(len(unique_true))
            width = 0.35
            
            bars1 = plt.bar(x - width/2, counts_true, width, label='True', color='green', alpha=0.7)
            bars2 = plt.bar(x + width/2, counts_pred, width, label='Predicted', color='blue', alpha=0.7)
            
            plt.xlabel('Class', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.title(f'True vs Predicted Distribution - {model_name.replace("_", " ").title()}', 
                     fontweight='bold', fontsize=14)
            plt.xticks(x, [class_names[i] for i in unique_true])
            plt.legend()
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    plt.annotate(f'{int(height)}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                print(f"✅ Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Could not plot class distribution: {e}")
    
    def plot_all_results(self, y_true, y_pred, y_pred_proba, class_names, 
                        evaluation_results, best_model_name, feature_importance_df=None,
                        save_dir=None):
        """
        Create comprehensive dashboard of all results
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            class_names: Names of classes
            evaluation_results (dict): Dictionary of evaluation results
            best_model_name (str): Name of the best model
            feature_importance_df (pd.DataFrame): Feature importance dataframe
            save_dir (str): Directory to save plots
        """
        print("\n📊 Creating comprehensive visualizations...")
        
        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Set up the figure
        fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title(f'Confusion Matrix\n({best_model_name})', fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. Class Distribution
        ax2 = plt.subplot(2, 3, 2)
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        
        x = np.arange(len(unique_true))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, counts_true, width, label='True', color='green', alpha=0.7)
        bars2 = ax2.bar(x + width/2, counts_pred, width, label='Predicted', color='blue', alpha=0.7)
        
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.set_title('True vs Predicted Distribution', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names)
        ax2.legend()
        
        # 3. Model Comparison
        ax3 = plt.subplot(2, 3, 3)
        models = list(evaluation_results.keys())
        f1_scores = [evaluation_results[m].get('f1', 0) for m in models]
        
        colors = ['#FF6B6B' if m == best_model_name else '#4ECDC4' for m in models]
        bars = ax3.bar(models, f1_scores, color=colors, edgecolor='black', linewidth=1)
        ax3.axhline(y=max(f1_scores), color='red', linestyle='--', alpha=0.5, linewidth=2)
        
        ax3.set_title('Model F1-Score Comparison', fontweight='bold')
        ax3.set_ylabel('F1-Score')
        ax3.set_xticklabels(models, rotation=45, ha='right')
        
        # 4. ROC Curve
        ax4 = plt.subplot(2, 3, 4)
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
            
            ax4.plot(fpr, tpr, label=f'{best_model_name} (AUC = {auc_score:.3f})', 
                    linewidth=2, color='purple')
            ax4.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5, linewidth=1)
            ax4.set_xlabel('False Positive Rate')
            ax4.set_ylabel('True Positive Rate')
            ax4.set_title('ROC Curve', fontweight='bold')
            ax4.legend()
            ax4.grid(alpha=0.3)
            ax4.fill_between(fpr, tpr, alpha=0.2, color='purple')
        
        # 5. Feature Importance
        ax5 = plt.subplot(2, 3, 5)
        if feature_importance_df is not None and not feature_importance_df.empty:
            plot_df = feature_importance_df.head(8)
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(plot_df)))
            bars = ax5.barh(range(len(plot_df)), plot_df['importance'][::-1], color=colors[::-1])
            ax5.set_yticks(range(len(plot_df)))
            ax5.set_yticklabels(plot_df['feature'][::-1])
            ax5.set_xlabel('Importance')
            ax5.set_title(f'Top Features - {best_model_name}', fontweight='bold')
            ax5.grid(axis='x', alpha=0.3)
        
        # 6. Performance Metrics Summary
        ax6 = plt.subplot(2, 3, 6)
        if best_model_name and best_model_name in evaluation_results:
            metrics = evaluation_results[best_model_name]
            
            key_metrics = ['accuracy', 'precision', 'recall', 'f1']
            if 'roc_auc' in metrics:
                key_metrics.append('roc_auc')
            
            values = [metrics.get(m, 0) for m in key_metrics]
            colors = ['#FF9F1C', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            bars = ax6.bar(range(len(key_metrics)), values, color=colors[:len(key_metrics)])
            ax6.set_xticks(range(len(key_metrics)))
            ax6.set_xticklabels([m.upper() for m in key_metrics])
            ax6.set_ylabel('Score')
            ax6.set_title(f'Key Metrics - {best_model_name}', fontweight='bold')
            ax6.set_ylim([0, 1])
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax6.annotate(f'{value:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.suptitle('Neonatal Risk Prediction Model Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, 'model_analysis_dashboard.png')
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Dashboard saved to {save_path}")
        
        plt.show()