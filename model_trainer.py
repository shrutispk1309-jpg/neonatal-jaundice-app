"""
Model training and evaluation module for Neonatal Health Risk Prediction
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, 
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG


class ModelTrainer:
    """Handles model training, evaluation, and selection"""
    
    def __init__(self, random_state=42):
        """
        Initialize the ModelTrainer
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.training_metrics = {}
        self.evaluation_results = {}
        self.best_model = None
        self.best_model_name = None
        
        # Data splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Class information
        self.classes_ = None
        self.label_encoder = None
        
    def initialize_models(self):
        """
        Initialize all machine learning models
        
        Returns:
            dict: Dictionary of initialized models
        """
        print("\n🤖 Initializing models...")
        
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                **CONFIG['logistic_regression_params']
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=CONFIG['random_forest_params']['n_estimators'],
                random_state=self.random_state,
                max_depth=CONFIG['random_forest_params']['max_depth'],
                min_samples_split=CONFIG['random_forest_params']['min_samples_split'],
                min_samples_leaf=CONFIG['random_forest_params']['min_samples_leaf'],
                class_weight=CONFIG['random_forest_params']['class_weight']
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=CONFIG['gradient_boosting_params']['n_estimators'],
                random_state=self.random_state,
                max_depth=CONFIG['gradient_boosting_params']['max_depth'],
                learning_rate=CONFIG['gradient_boosting_params']['learning_rate'],
                subsample=CONFIG['gradient_boosting_params']['subsample']
            ),
            'svm': SVC(
                probability=CONFIG['svm_params']['probability'],
                random_state=self.random_state,
                class_weight=CONFIG['svm_params']['class_weight'],
                kernel=CONFIG['svm_params']['kernel'],
                C=CONFIG['svm_params']['C'],
                gamma=CONFIG['svm_params']['gamma']
            )
        }
        
        print(f"✅ {len(self.models)} models initialized:")
        for name in self.models.keys():
            print(f"   - {name}")
        
        return self.models
    
    def split_data(self, X, y, test_size=None, stratify=True):
        """
        Split data into training and testing sets
        
        Args:
            X: Features
            y: Target
            test_size (float): Proportion of test set
            stratify (bool): Whether to stratify split
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if test_size is None:
            test_size = CONFIG['test_size']
        
        print(f"\n✂️  Splitting data (test_size={test_size})...")
        
        # Split with stratification
        stratify_param = y if stratify else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        print(f"✅ Data split complete")
        print(f"   Training set: {len(self.X_train)} samples")
        print(f"   Testing set: {len(self.X_test)} samples")
        
        # Show class distribution
        train_counts = np.bincount(self.y_train)
        test_counts = np.bincount(self.y_test)
        
        if self.classes_ is not None:
            print(f"   Training class distribution:")
            for i, count in enumerate(train_counts):
                class_name = self.classes_[i]
                percentage = count / len(self.y_train) * 100
                print(f"     {class_name}: {count} ({percentage:.1f}%)")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def handle_imbalance(self, smote_threshold=0.3):
        """
        Handle class imbalance using SMOTE
        
        Args:
            smote_threshold (float): Threshold for applying SMOTE
            
        Returns:
            tuple: (X_train_resampled, y_train_resampled)
        """
        print("\n⚖️  Handling class imbalance...")
        
        if self.X_train is None or self.y_train is None:
            print("❌ Training data not available.")
            return None
        
        # Check class distribution
        class_counts = np.bincount(self.y_train)
        minority_ratio = min(class_counts) / len(self.y_train)
        
        print(f"   Original class distribution:")
        for i, count in enumerate(class_counts):
            if self.classes_ is not None:
                class_name = self.classes_[i]
                percentage = count / len(self.y_train) * 100
                print(f"     {class_name}: {count} ({percentage:.1f}%)")
        
        if minority_ratio < smote_threshold:
            try:
                print(f"   Minority class ratio: {minority_ratio:.1%} - Applying SMOTE...")
                smote = SMOTE(random_state=self.random_state)
                X_resampled, y_resampled = smote.fit_resample(
                    self.X_train, self.y_train
                )
                
                print(f"✅ SMOTE applied successfully")
                print(f"   Resampled training: {len(X_resampled)} samples")
                
                # Show new distribution
                new_counts = np.bincount(y_resampled)
                print(f"   New class distribution:")
                for i, count in enumerate(new_counts):
                    if self.classes_ is not None:
                        class_name = self.classes_[i]
                        percentage = count / len(y_resampled) * 100
                        print(f"     {class_name}: {count} ({percentage:.1f}%)")
                
                # Update training data
                self.X_train = X_resampled
                self.y_train = y_resampled
                
            except Exception as e:
                print(f"⚠️ SMOTE failed: {e}")
                print("   Continuing with original data...")
        else:
            print("✅ No significant class imbalance detected")
        
        return self.X_train, self.y_train
    
    def train_models(self):
        """
        Train all initialized models
        
        Returns:
            dict: Dictionary of trained models
        """
        print("\n🏋️‍♂️ Training models...")
        
        if not self.models:
            print("❌ No models initialized.")
            return None
        
        self.trained_models = {}
        self.training_metrics = {}
        
        for name, model in self.models.items():
            print(f"   Training {name}...", end=' ')
            try:
                model.fit(self.X_train, self.y_train)
                self.trained_models[name] = model
                
                # Calculate training metrics
                y_train_pred = model.predict(self.X_train)
                train_metrics = self._calculate_metrics(self.y_train, y_train_pred, 'train')
                self.training_metrics[name] = train_metrics
                
                print(f"✅")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n✅ Training complete. {len(self.trained_models)} models trained successfully.")
        return self.trained_models
    
    def _calculate_metrics(self, y_true, y_pred, dataset_type):
        """Calculate comprehensive metrics"""
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
            # For binary classification, add specific metrics for minority class
            if len(np.unique(y_true)) == 2:
                try:
                    metrics['precision_minority'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
                    metrics['recall_minority'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
                    metrics['f1_minority'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
                except:
                    pass
            
            return metrics
        except Exception as e:
            print(f"⚠️ Error calculating metrics: {e}")
            return {}
    
    def evaluate_models(self):
        """
        Evaluate all trained models on test set
        
        Returns:
            dict: Evaluation results
        """
        print("\n📈 Evaluating models...")
        
        if not self.trained_models:
            print("❌ No trained models.")
            return None
        
        self.evaluation_results = {}
        best_f1 = 0
        
        print("   Model performance on test set:")
        print("   " + "-" * 70)
        
        for name, model in self.trained_models.items():
            # Make predictions
            try:
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, "predict_proba") else None
                
                # Calculate metrics
                metrics = self._calculate_metrics(self.y_test, y_pred, 'test')
                
                # Add ROC-AUC for binary classification if probabilities available
                if y_pred_proba is not None and len(np.unique(self.y_test)) == 2:
                    try:
                        metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                    except:
                        pass
                
                self.evaluation_results[name] = metrics
                
                # Display metrics
                print(f"   {name:20s} | Acc: {metrics.get('accuracy', 0):.3f} | "
                      f"Prec: {metrics.get('precision', 0):.3f} | "
                      f"Rec: {metrics.get('recall', 0):.3f} | "
                      f"F1: {metrics.get('f1', 0):.3f}", end='')
                
                if 'roc_auc' in metrics:
                    print(f" | AUC: {metrics['roc_auc']:.3f}")
                else:
                    print()
                
                # Update best model based on F1-score
                if metrics.get('f1', 0) > best_f1:
                    best_f1 = metrics['f1']
                    self.best_model_name = name
                    self.best_model = model
                    
            except Exception as e:
                print(f"   ⚠️ Error evaluating {name}: {e}")
        
        return self.evaluation_results
    
    def get_feature_importance(self, model_name=None):
        """
        Get feature importance from a trained model
        
        Args:
            model_name (str): Name of the model to use
            
        Returns:
            np.array: Feature importance scores
        """
        if not self.trained_models:
            print("❌ No trained models.")
            return None
        
        if model_name is None:
            model_name = self.best_model_name if self.best_model_name else list(self.trained_models.keys())[0]
        
        if model_name not in self.trained_models:
            print(f"❌ Model '{model_name}' not found.")
            return None
        
        model = self.trained_models[model_name]
        
        # Get feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
        else:
            print(f"⚠️ Feature importance not available for {model_name}")
            return None
        
        return importances
    
    def get_classification_report(self, model_name=None):
        """
        Get detailed classification report
        
        Args:
            model_name (str): Name of the model to use
            
        Returns:
            str: Classification report
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            print(f"❌ Model '{model_name}' not found.")
            return None
        
        model = self.trained_models[model_name]
        y_pred = model.predict(self.X_test)
        
        report = classification_report(
            self.y_test, 
            y_pred, 
            target_names=self.classes_ if self.classes_ is not None else None
        )
        
        return report
    
    def set_classes(self, classes):
        """Set class names"""
        self.classes_ = classes
    
    def set_label_encoder(self, label_encoder):
        """Set label encoder reference"""
        self.label_encoder = label_encoder