"""
NEONATAL HEALTH RISK PREDICTION MODEL
Complete implementation - automatically finds and processes data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, 
    confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import glob
import json
warnings.filterwarnings('ignore')

# Check for required packages
def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', 'sklearn'),
        ('imblearn', 'imblearn'),
        ('matplotlib', 'plt'),
        ('seaborn', 'sns')
    ]
    
    missing_packages = []
    for package, _ in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing packages:", missing_packages)
        print("Install with: pip install", " ".join(missing_packages))
        return False
    return True

class NeonatalRiskPredictor:
    """
    Complete pipeline for neonatal health risk prediction
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the predictor
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.feature_importance = {}
        self.best_model = None
        self.data_loaded = False
        
    def find_dataset(self):
        """
        Automatically find CSV files in the current directory
        """
        print("🔍 Searching for dataset files...")
        
        # Look for common CSV filenames
        csv_files = glob.glob("*.csv")
        
        if not csv_files:
            # Try to find any CSV in current directory and subdirectories
            csv_files = glob.glob("**/*.csv", recursive=True)
        
        print(f"Found {len(csv_files)} CSV file(s):")
        for i, file in enumerate(csv_files, 1):
            print(f"  {i}. {file}")
        
        # Look for files with relevant names
        relevant_files = []
        keywords = ['newborn', 'neonatal', 'baby', 'infant', 'health', 'risk', 'jaundice', 'medical']
        
        for file in csv_files:
            filename_lower = file.lower()
            if any(keyword in filename_lower for keyword in keywords):
                relevant_files.append(file)
        
        if relevant_files:
            print("\n📁 Most relevant files:")
            for file in relevant_files[:3]:  # Show top 3
                print(f"  ✓ {file}")
            
            # Use the first relevant file
            self.data_path = relevant_files[0]
            print(f"\n✅ Using dataset: {self.data_path}")
            return self.data_path
        elif csv_files:
            # Use the first CSV file found
            self.data_path = csv_files[0]
            print(f"\n📁 Using first CSV found: {self.data_path}")
            return self.data_path
        else:
            print("❌ No CSV files found in the current directory.")
            return None
    
    def create_sample_data(self):
        """
        Create synthetic sample data for demonstration
        """
        print("\n📊 Creating sample dataset for training...")
        
        np.random.seed(self.random_state)
        n_samples = 3000  # Matching your dataset size
        
        # Create realistic neonatal data based on your description
        data = {
            'baby_id': range(1, n_samples + 1),
            'name': [f'Baby_{i}' for i in range(1, n_samples + 1)],
            'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.51, 0.49]),
            'gestational_age_weeks': np.random.normal(38.5, 2, n_samples).clip(28, 42),
            'birth_weight_kg': np.random.normal(3.2, 0.5, n_samples).clip(1.5, 5.0),
            'birth_length_cm': np.random.normal(50, 3, n_samples).clip(40, 60),
            'birth_head_circumference_cm': np.random.normal(34, 2, n_samples).clip(28, 40),
            
            # Time-series growth data
            'age_days': np.random.randint(0, 30, n_samples),
            'weight_kg': lambda: np.random.normal(3.5, 0.6, n_samples).clip(2.0, 6.0),
            'length_cm': lambda: np.random.normal(51, 3, n_samples).clip(42, 62),
            'head_circumference_cm': lambda: np.random.normal(35, 2, n_samples).clip(30, 42),
            
            # Vital signs
            'heart_rate_bpm': np.random.normal(140, 20, n_samples).clip(100, 180),
            'respiratory_rate_bpm': np.random.normal(40, 10, n_samples).clip(20, 80),
            'oxygen_saturation_pct': np.random.normal(97, 3, n_samples).clip(85, 100),
            'body_temperature_c': np.random.normal(36.8, 0.5, n_samples).clip(35.5, 38.5),
            
            # Clinical indicators
            'feeding_frequency_per_day': np.random.randint(6, 12, n_samples),
            'urine_output_per_day': np.random.randint(4, 10, n_samples),
            'bilirubin_level_mg_dl': np.random.gamma(2, 2, n_samples).clip(0, 25),
            'infection_flag': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'nicu_admission': np.random.choice([0, 1], n_samples, p=[0.90, 0.10]),
            
            # Date
            'date': pd.date_range('2024-01-01', periods=n_samples).strftime('%Y-%m-%d')
        }
        
        # Generate dynamic columns
        for key, value in data.items():
            if callable(value):
                data[key] = value()
        
        # Create DataFrame
        self.df = pd.DataFrame(data)
        
        # Create realistic risk levels (86.7% Healthy, 13.3% At Risk)
        n_at_risk = int(n_samples * 0.133)  # 13.3%
        n_healthy = n_samples - n_at_risk
        
        # Risk factors based on medical logic
        risk_scores = (
            (self.df['gestational_age_weeks'] < 37).astype(int) * 2 +  # Premature
            (self.df['birth_weight_kg'] < 2.5).astype(int) * 2 +  # Low birth weight
            (self.df['oxygen_saturation_pct'] < 94).astype(int) * 3 +  # Low oxygen
            (self.df['bilirubin_level_mg_dl'] > 15).astype(int) * 2 +  # High bilirubin
            self.df['infection_flag'] * 3 +  # Infection
            self.df['nicu_admission'] * 2  # NICU admission
        )
        
        # Assign risk based on scores
        risk_threshold = np.percentile(risk_scores, 100 - 13.3)  # Top 13.3% are at risk
        self.df['risk_level'] = np.where(risk_scores >= risk_threshold, 'At Risk', 'Healthy')
        
        # Verify distribution
        actual_dist = self.df['risk_level'].value_counts(normalize=True)
        print(f"✅ Created synthetic dataset with {n_samples} records")
        print(f"   Risk distribution: {actual_dist['Healthy']:.1%} Healthy, {actual_dist['At Risk']:.1%} At Risk")
        
        self.data_loaded = True
        return self.df
    
    def load_or_create_data(self):
        """
        Try to load existing data or create sample data
        """
        # First try to find existing dataset
        data_path = self.find_dataset()
        
        if data_path:
            try:
                self.df = pd.read_csv(data_path)
                print(f"✅ Loaded dataset from {data_path}")
                print(f"   Shape: {self.df.shape}")
                
                # Check if risk_level column exists
                if 'risk_level' in self.df.columns:
                    print(f"   Target distribution:")
                    print(self.df['risk_level'].value_counts())
                else:
                    print("⚠️ Warning: 'risk_level' column not found")
                
                self.data_loaded = True
                return self.df
                
            except Exception as e:
                print(f"❌ Error loading {data_path}: {e}")
                print("Creating sample data instead...")
        
        # If no dataset found or error loading, create sample data
        return self.create_sample_data()
    
    def explore_data(self):
        """
        Exploratory data analysis
        """
        if not hasattr(self, 'df') or self.df is None:
            print("❌ No data available.")
            return
        
        print("\n🔍 Data Exploration:")
        print("="*50)
        
        # Basic info
        print("1. Dataset Info:")
        print(f"   Total rows: {len(self.df)}")
        print(f"   Total columns: {len(self.df.columns)}")
        
        # Check for missing values
        print("\n2. Missing Values Check:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("   ✅ No missing values found")
        else:
            print("   ⚠️ Missing values found:")
            for col, count in missing[missing > 0].items():
                print(f"     - {col}: {count} ({count/len(self.df)*100:.2f}%)")
        
        # Data types
        print("\n3. Data Types:")
        dtypes_summary = self.df.dtypes.value_counts()
        for dtype, count in dtypes_summary.items():
            print(f"   {dtype}: {count} columns")
        
        # Show sample
        print("\n4. Sample of data (first 2 rows):")
        print(self.df.head(2))
        
        # Target variable analysis
        if 'risk_level' in self.df.columns:
            print("\n5. Target Variable Analysis:")
            target_dist = self.df['risk_level'].value_counts()
            for level, count in target_dist.items():
                percentage = count / len(self.df) * 100
                print(f"   {level}: {count} ({percentage:.1f}%)")
    
    def preprocess_data(self):
        """
        Step 2: Data Preprocessing
        """
        print("\n🔧 Preprocessing data...")
        
        if not hasattr(self, 'df'):
            print("❌ No data loaded.")
            return None, None
        
        # Create a copy
        df_processed = self.df.copy()
        
        # 1. Drop non-informative columns
        columns_to_drop = ['baby_id', 'name', 'date']
        columns_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
        
        if columns_to_drop:
            print(f"   Dropping columns: {columns_to_drop}")
            df_processed = df_processed.drop(columns=columns_to_drop)
        
        # 2. Handle categorical variables
        categorical_cols = ['gender', 'infection_flag', 'nicu_admission']
        for col in categorical_cols:
            if col in df_processed.columns:
                if df_processed[col].dtype == 'object':
                    df_processed[col] = df_processed[col].astype('category').cat.codes
                elif df_processed[col].dtype in ['int64', 'float64']:
                    df_processed[col] = df_processed[col].astype(int)
                print(f"   Encoded '{col}'")
        
        # 3. Define target and features
        print("   🎯 Focus: General risk level prediction")
        
        if 'risk_level' not in df_processed.columns:
            print("❌ 'risk_level' column not found")
            return None, None
        
        # Features based on your analysis
        risk_features = [
            'gestational_age_weeks', 'birth_weight_kg', 'birth_length_cm',
            'birth_head_circumference_cm', 'age_days', 'weight_kg',
            'length_cm', 'head_circumference_cm', 'heart_rate_bpm',
            'respiratory_rate_bpm', 'oxygen_saturation_pct', 'body_temperature_c',
            'feeding_frequency_per_day', 'urine_output_per_day',
            'bilirubin_level_mg_dl', 'infection_flag', 'nicu_admission', 'gender'
        ]
        
        # Keep only features that exist in data
        features = [f for f in risk_features if f in df_processed.columns]
        X = df_processed[features]
        y = self.label_encoder.fit_transform(df_processed['risk_level'])
        self.target_type = 'risk_level'
        
        # Store processed data
        self.X = X
        self.y = y
        self.feature_names = X.columns.tolist()
        
        print(f"✅ Preprocessing complete")
        print(f"   Features selected: {len(self.feature_names)}")
        print(f"   Target variable: {self.target_type}")
        print(f"   Classes: {self.label_encoder.classes_}")
        
        return X, y
    
    def scale_features(self):
        """
        Step 3: Feature Scaling
        """
        print("\n📐 Scaling features...")
        
        if not hasattr(self, 'X'):
            print("❌ No features available.")
            return None
        
        # Apply StandardScaler
        X_scaled = self.X.copy()
        numerical_cols = X_scaled.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            X_scaled[numerical_cols] = self.scaler.fit_transform(self.X[numerical_cols])
            print(f"✅ Scaled {len(numerical_cols)} numerical features")
        else:
            print("⚠️ No numerical features to scale")
        
        self.X_scaled = X_scaled
        return X_scaled
    
    def split_data(self, test_size=0.2):
        """
        Step 4: Dataset Splitting
        """
        print(f"\n✂️  Splitting data (test_size={test_size})...")
        
        if not hasattr(self, 'X_scaled'):
            print("❌ Scaled features not available.")
            return None
        
        # Split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled,
            self.y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=self.y
        )
        
        print(f"✅ Data split complete")
        print(f"   Training set: {len(self.X_train)} samples")
        print(f"   Testing set: {len(self.X_test)} samples")
        
        # Show class distribution
        train_counts = np.bincount(self.y_train)
        test_counts = np.bincount(self.y_test)
        
        print(f"   Training class distribution:")
        for i, count in enumerate(train_counts):
            class_name = self.label_encoder.inverse_transform([i])[0]
            percentage = count / len(self.y_train) * 100
            print(f"     {class_name}: {count} ({percentage:.1f}%)")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def handle_imbalance(self):
        """
        Step 5: Handle Class Imbalance
        """
        print("\n⚖️  Handling class imbalance...")
        
        if not hasattr(self, 'X_train'):
            print("❌ Training data not available.")
            return None
        
        # Check class distribution
        class_counts = np.bincount(self.y_train)
        minority_ratio = min(class_counts) / len(self.y_train)
        
        print(f"   Original class distribution:")
        for i, count in enumerate(class_counts):
            class_name = self.label_encoder.inverse_transform([i])[0]
            percentage = count / len(self.y_train) * 100
            print(f"     {class_name}: {count} ({percentage:.1f}%)")
        
        if minority_ratio < 0.3:  # If minority class is less than 30%
            try:
                print(f"   Minority class ratio: {minority_ratio:.1%} - Applying SMOTE...")
                smote = SMOTE(random_state=self.random_state)
                self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
                    self.X_train, self.y_train
                )
                
                print(f"✅ SMOTE applied successfully")
                print(f"   Resampled training: {len(self.X_train_resampled)} samples")
                
                # Show new distribution
                new_counts = np.bincount(self.y_train_resampled)
                print(f"   New class distribution:")
                for i, count in enumerate(new_counts):
                    class_name = self.label_encoder.inverse_transform([i])[0]
                    percentage = count / len(self.y_train_resampled) * 100
                    print(f"     {class_name}: {count} ({percentage:.1f}%)")
                
                # Update training data
                self.X_train = self.X_train_resampled
                self.y_train = self.y_train_resampled
                
            except Exception as e:
                print(f"⚠️ SMOTE failed: {e}")
                print("   Continuing with original data...")
        else:
            print("✅ No significant class imbalance detected")
        
        return self.X_train, self.y_train
    
    def initialize_models(self):
        """
        Step 6: Model Selection
        """
        print("\n🤖 Initializing models...")
        
        # Check if it's binary or multi-class
        n_classes = len(np.unique(self.y_train))
        print(f"   Problem type: {'Binary' if n_classes == 2 else 'Multi-class'} classification")
        
        # Initialize models with appropriate settings
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state,
                class_weight='balanced_subsample',
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                random_state=self.random_state,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8
            ),
            'svm': SVC(
                probability=True,
                random_state=self.random_state,
                class_weight='balanced',
                kernel='rbf',
                C=1.0,
                gamma='scale'
            )
        }
        
        print(f"✅ {len(self.models)} models initialized:")
        for name, model in self.models.items():
            print(f"   - {name}: {model.__class__.__name__}")
        
        return self.models
    
    def train_models(self):
        """
        Step 7: Model Training
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
        Step 8 & 9: Prediction and Evaluation
        """
        print("\n📈 Evaluating models...")
        
        if not hasattr(self, 'trained_models'):
            print("❌ No trained models.")
            return None
        
        self.evaluation_results = {}
        self.best_model = None
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
                    self.best_model = name
                    
            except Exception as e:
                print(f"   ⚠️ Error evaluating {name}: {e}")
        
        # Print evaluation summary
        self._print_evaluation_summary()
        
        return self.evaluation_results
    
    def _print_evaluation_summary(self):
        """Print formatted evaluation results"""
        if not self.evaluation_results:
            print("No evaluation results available.")
            return
        
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY")
        print("="*80)
        
        # Create results DataFrame
        results_data = []
        for model_name, metrics in self.evaluation_results.items():
            row = {'Model': model_name.replace('_', ' ').title()}
            row.update({k: v for k, v in metrics.items() if not k.endswith('minority')})
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        # Print main metrics
        print("\nOverall Metrics:")
        print(results_df.round(4).to_string(index=False))
        
        if self.best_model:
            print(f"\n🏆 BEST MODEL: {self.best_model.upper()}")
            best_metrics = self.evaluation_results[self.best_model]
            
            print(f"   Accuracy: {best_metrics.get('accuracy', 0):.4f}")
            print(f"   F1-Score: {best_metrics.get('f1', 0):.4f}")
            
            if 'recall_minority' in best_metrics:
                print(f"   Recall (At Risk): {best_metrics.get('recall_minority', 0):.4f}")
            if 'precision_minority' in best_metrics:
                print(f"   Precision (At Risk): {best_metrics.get('precision_minority', 0):.4f}")
            if 'roc_auc' in best_metrics:
                print(f"   ROC-AUC: {best_metrics.get('roc_auc', 0):.4f}")
            
            # Print detailed report for best model
            print(f"\n📋 Detailed Classification Report for {self.best_model}:")
            model = self.trained_models[self.best_model]
            y_pred = model.predict(self.X_test)
            
            print(classification_report(
                self.y_test, 
                y_pred, 
                target_names=self.label_encoder.classes_
            ))
    
    def analyze_feature_importance(self, model_name='random_forest', top_n=10):
        """
        Step 10: Feature Importance Analysis
        """
        print(f"\n🔍 Analyzing feature importance...")
        
        if not hasattr(self, 'trained_models'):
            print("❌ No trained models.")
            return None
        
        if model_name not in self.trained_models:
            # Try to use best model or random forest
            if self.best_model:
                model_name = self.best_model
            elif 'random_forest' in self.trained_models:
                model_name = 'random_forest'
            else:
                # Use first available model
                model_name = list(self.trained_models.keys())[0]
        
        model = self.trained_models[model_name]
        
        # Get feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_type = "Feature Importances"
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
            importance_type = "Coefficient Magnitudes"
        else:
            print(f"⚠️ Feature importance not available for {model_name}")
            return None
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Store
        self.feature_importance[model_name] = feature_importance_df
        
        # Print top features
        print(f"📊 Top {top_n} Most Important Features ({model_name}):")
        print("-" * 50)
        for i, row in feature_importance_df.head(top_n).iterrows():
            print(f"{i+1:2d}. {row['feature']:35s}: {row['importance']:.4f}")
        
        # Plot feature importance
        self._plot_feature_importance(feature_importance_df.head(top_n), model_name)
        
        return feature_importance_df
    
    def _plot_feature_importance(self, importance_df, model_name):
        """Plot feature importance"""
        try:
            plt.figure(figsize=(12, 6))
            bars = plt.barh(range(len(importance_df)), importance_df['importance'][::-1], 
                           color=plt.cm.viridis(np.linspace(0, 1, len(importance_df))))
            plt.yticks(range(len(importance_df)), importance_df['feature'][::-1])
            plt.xlabel('Importance Score', fontsize=12)
            plt.title(f'Feature Importance - {model_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"⚠️ Could not plot feature importance: {e}")
    
    def visualize_results(self):
        """
        Create visualizations for model results
        """
        if not hasattr(self, 'best_model') or not self.best_model:
            print("❌ No best model identified yet.")
            return
        
        model = self.trained_models[self.best_model]
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
        
        print("\n📊 Creating visualizations...")
        
        # Set up the figure
        fig = plt.figure(figsize=(18, 10))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=ax1)
        ax1.set_title(f'Confusion Matrix\n({self.best_model})', fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. Class Distribution Comparison
        ax2 = plt.subplot(2, 3, 2)
        unique_true, counts_true = np.unique(self.y_test, return_counts=True)
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        
        x = np.arange(len(unique_true))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, counts_true, width, label='True', color='green', alpha=0.7)
        bars2 = ax2.bar(x + width/2, counts_pred, width, label='Predicted', color='blue', alpha=0.7)
        
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.set_title('True vs Predicted Distribution', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([self.label_encoder.inverse_transform([i])[0] for i in unique_true])
        ax2.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        # 3. Model Comparison
        ax3 = plt.subplot(2, 3, 3)
        if hasattr(self, 'evaluation_results'):
            models = list(self.evaluation_results.keys())
            f1_scores = [self.evaluation_results[m].get('f1', 0) for m in models]
            
            colors = ['#FF6B6B' if m == self.best_model else '#4ECDC4' for m in models]
            bars = ax3.bar(models, f1_scores, color=colors, edgecolor='black', linewidth=1)
            ax3.axhline(y=max(f1_scores), color='red', linestyle='--', alpha=0.5, linewidth=2)
            
            ax3.set_title('Model F1-Score Comparison', fontweight='bold')
            ax3.set_ylabel('F1-Score')
            ax3.set_xticklabels(models, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, score in zip(bars, f1_scores):
                height = bar.get_height()
                ax3.annotate(f'{score:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 4. ROC Curve for binary classification
        ax4 = plt.subplot(2, 3, 4)
        if y_pred_proba is not None and len(np.unique(self.y_test)) == 2:
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
            auc_score = roc_auc_score(self.y_test, y_pred_proba[:, 1])
            
            ax4.plot(fpr, tpr, label=f'{self.best_model} (AUC = {auc_score:.3f})', 
                    linewidth=2, color='purple')
            ax4.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5, linewidth=1)
            ax4.set_xlabel('False Positive Rate')
            ax4.set_ylabel('True Positive Rate')
            ax4.set_title('ROC Curve', fontweight='bold')
            ax4.legend()
            ax4.grid(alpha=0.3)
            
            # Fill under the curve
            ax4.fill_between(fpr, tpr, alpha=0.2, color='purple')
        
        # 5. Feature Importance (if available)
        ax5 = plt.subplot(2, 3, 5)
        if hasattr(self, 'feature_importance') and self.best_model in self.feature_importance:
            importance_df = self.feature_importance[self.best_model].head(8)
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(importance_df)))
            bars = ax5.barh(range(len(importance_df)), importance_df['importance'][::-1], color=colors[::-1])
            ax5.set_yticks(range(len(importance_df)))
            ax5.set_yticklabels(importance_df['feature'][::-1])
            ax5.set_xlabel('Importance')
            ax5.set_title(f'Top Features - {self.best_model}', fontweight='bold')
            ax5.grid(axis='x', alpha=0.3)
        
        # 6. Performance Metrics Summary
        ax6 = plt.subplot(2, 3, 6)
        if self.best_model and self.best_model in self.evaluation_results:
            metrics = self.evaluation_results[self.best_model]
            
            # Select key metrics to display
            key_metrics = ['accuracy', 'precision', 'recall', 'f1']
            if 'roc_auc' in metrics:
                key_metrics.append('roc_auc')
            
            values = [metrics.get(m, 0) for m in key_metrics]
            colors = ['#FF9F1C', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            bars = ax6.bar(range(len(key_metrics)), values, color=colors[:len(key_metrics)])
            ax6.set_xticks(range(len(key_metrics)))
            ax6.set_xticklabels([m.upper() for m in key_metrics])
            ax6.set_ylabel('Score')
            ax6.set_title(f'Key Metrics - {self.best_model}', fontweight='bold')
            ax6.set_ylim([0, 1])
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax6.annotate(f'{value:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle('Neonatal Risk Prediction Model Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def run_full_pipeline(self):
        """
        Run complete pipeline from start to finish
        """
        print("🚀" + "="*78)
        print("🚀 STARTING NEONATAL HEALTH RISK PREDICTION PIPELINE")
        print("🚀" + "="*78)
        
        # Check dependencies
        if not check_dependencies():
            return None
        
        # Load or create data
        print("\n📥 STEP 1: Loading Data")
        print("-" * 50)
        self.load_or_create_data()
        
        # Explore data
        print("\n🔍 STEP 2: Exploring Data")
        print("-" * 50)
        self.explore_data()
        
        # Preprocess data
        print("\n🔧 STEP 3: Preprocessing Data")
        print("-" * 50)
        X, y = self.preprocess_data()
        if X is None or y is None:
            print("❌ Preprocessing failed.")
            return None
        
        # Scale features
        print("\n📐 STEP 4: Scaling Features")
        print("-" * 50)
        self.scale_features()
        
        # Split data
        print("\n✂️  STEP 5: Splitting Data")
        print("-" * 50)
        self.split_data(test_size=0.2)
        
        # Handle imbalance
        print("\n⚖️  STEP 6: Handling Class Imbalance")
        print("-" * 50)
        self.handle_imbalance()
        
        # Initialize models
        print("\n🤖 STEP 7: Initializing Models")
        print("-" * 50)
        self.initialize_models()
        
        # Train models
        print("\n🏋️‍♂️ STEP 8: Training Models")
        print("-" * 50)
        self.train_models()
        
        # Evaluate models
        print("\n📈 STEP 9: Evaluating Models")
        print("-" * 50)
        self.evaluate_models()
        
        # Analyze feature importance
        print("\n🔍 STEP 10: Analyzing Feature Importance")
        print("-" * 50)
        self.analyze_feature_importance()
        
        # Visualize results
        print("\n📊 STEP 11: Visualizing Results")
        print("-" * 50)
        self.visualize_results()
        
        print("\n" + "✅"*40)
        print("✅" + " PIPELINE COMPLETED SUCCESSFULLY! ".center(78, "=") + "✅")
        print("✅"*40)
        
        return self
    
    def save_model(self, filename='neonatal_risk_model.pkl'):
        """
        Save the trained model and related artifacts
        """
        try:
            import joblib
            
            # Create a dictionary with all model artifacts
            model_artifacts = {
                'best_model': self.trained_models[self.best_model] if self.best_model else None,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'best_model_name': self.best_model,
                'evaluation_results': self.evaluation_results
            }
            
            joblib.dump(model_artifacts, filename)
            print(f"✅ Model saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def predict_new(self, input_data):
        """
        Make predictions for new data
        """
        if not hasattr(self, 'best_model') or not self.best_model:
            print("❌ Model not trained. Run the pipeline first.")
            return None
        
        try:
            # Convert input to DataFrame
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            elif isinstance(input_data, pd.DataFrame):
                input_df = input_data
            else:
                print("❌ Input must be a dictionary or DataFrame")
                return None
            
            # Ensure all features are present
            missing_features = set(self.feature_names) - set(input_df.columns)
            if missing_features:
                print(f"⚠️ Warning: Missing features: {missing_features}")
                # Add missing features with default values
                for feature in missing_features:
                    input_df[feature] = 0
            
            # Select only the required features
            input_df = input_df[self.feature_names]
            
            # Scale features
            input_scaled = pd.DataFrame(
                self.scaler.transform(input_df),
                columns=self.feature_names
            )
            
            # Make prediction
            model = self.trained_models[self.best_model]
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
            
            # Convert to readable labels
            prediction_labels = self.label_encoder.inverse_transform(prediction)
            
            # Create results
            results = []
            for i in range(len(prediction)):
                result = {
                    'prediction': prediction_labels[i],
                    'probability': float(probability[i][prediction[i]]),
                    'probabilities': {label: float(prob) for label, prob in 
                                    zip(self.label_encoder.classes_, probability[i])},
                    'model_used': self.best_model
                }
                results.append(result)
            
            return results[0] if len(results) == 1 else results
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """
    Main function - automatically runs the complete pipeline
    """
    print("\n" + "🎯" * 40)
    print("🎯 NEONATAL HEALTH RISK PREDICTION SYSTEM")
    print("🎯" * 40)
    
    # Initialize and run pipeline
    predictor = NeonatalRiskPredictor(random_state=42)
    
    try:
        # Run the complete pipeline
        predictor.run_full_pipeline()
        
        # Save the model
        predictor.save_model('trained_neonatal_model.pkl')
        
        # Example prediction
        print("\n" + "🔮" * 40)
        print("🔮 EXAMPLE PREDICTION")
        print("🔮" * 40)
        
        # Create example input based on the features
        example_input = {}
        for feature in predictor.feature_names[:10]:  # Just use first 10 features for example
            if 'flag' in feature or 'admission' in feature or 'gender' in feature:
                example_input[feature] = 0
            elif 'age' in feature:
                example_input[feature] = 5.0
            elif 'weight' in feature:
                example_input[feature] = 3.2
            elif 'oxygen' in feature:
                example_input[feature] = 98.0
            elif 'bilirubin' in feature:
                example_input[feature] = 8.5
            else:
                example_input[feature] = 1.0
        
        print(f"\nMaking prediction with example data...")
        print(f"Using {len(predictor.feature_names)} features")
        
        result = predictor.predict_new(example_input)
        
        if result:
            print(f"\n✅ Prediction Result:")
            print(f"   Risk Level: {result['prediction']}")
            print(f"   Confidence: {result['probability']:.2%}")
            print(f"   Model Used: {result['model_used']}")
            
            print(f"\n📊 All Probabilities:")
            for label, prob in result['probabilities'].items():
                print(f"   {label}: {prob:.2%}")
        
        print("\n" + "✨" * 40)
        print("✨ MODEL TRAINING COMPLETE!")
        print("✨" * 40)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred during pipeline execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the pipeline automatically
    main()