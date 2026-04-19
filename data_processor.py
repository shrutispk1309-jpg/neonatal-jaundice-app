"""
Data loading and preprocessing module for Neonatal Health Risk Prediction
"""

import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import CONFIG, PATHS


class DataProcessor:
    """Handles all data loading, preprocessing, and transformation tasks"""
    
    def __init__(self, random_state=42):
        """
        Initialize the DataProcessor
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.df = None
        self.X = None
        self.y = None
        self.X_scaled = None
        self.feature_names = None
        self.target_type = None
        
    def find_dataset(self):
        """
        Automatically find CSV files in the current directory
        
        Returns:
            str: Path to the found dataset or None
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
        keywords = CONFIG['data_keywords']
        
        for file in csv_files:
            filename_lower = file.lower()
            if any(keyword in filename_lower for keyword in keywords):
                relevant_files.append(file)
        
        if relevant_files:
            print("\n📁 Most relevant files:")
            for file in relevant_files[:3]:  # Show top 3
                print(f"  ✓ {file}")
            
            # Use the first relevant file
            data_path = relevant_files[0]
            print(f"\n✅ Using dataset: {data_path}")
            return data_path
        elif csv_files:
            # Use the first CSV file found
            data_path = csv_files[0]
            print(f"\n📁 Using first CSV found: {data_path}")
            return data_path
        else:
            print("❌ No CSV files found in the current directory.")
            return None
    
    def create_sample_data(self):
        """
        Create synthetic sample data for demonstration
        
        Returns:
            pd.DataFrame: Generated sample data
        """
        print("\n📊 Creating sample dataset for training...")
        
        np.random.seed(self.random_state)
        sample_config = CONFIG['sample_data']
        n_samples = sample_config['n_samples']
        
        # Create realistic neonatal data
        data = {
            'baby_id': range(1, n_samples + 1),
            'name': [f'Baby_{i}' for i in range(1, n_samples + 1)],
            'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.51, 0.49]),
            'gestational_age_weeks': np.random.normal(
                sample_config['gestational_age_mean'], 
                sample_config['gestational_age_std'], 
                n_samples
            ).clip(28, 42),
            'birth_weight_kg': np.random.normal(
                sample_config['birth_weight_mean'], 
                sample_config['birth_weight_std'], 
                n_samples
            ).clip(1.5, 5.0),
            'birth_length_cm': np.random.normal(
                sample_config['birth_length_mean'], 
                sample_config['birth_length_std'], 
                n_samples
            ).clip(40, 60),
            'birth_head_circumference_cm': np.random.normal(
                sample_config['head_circumference_mean'], 
                sample_config['head_circumference_std'], 
                n_samples
            ).clip(28, 40),
            
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
        df = pd.DataFrame(data)
        
        # Create realistic risk levels
        n_at_risk = int(n_samples * sample_config['at_risk_ratio'])
        
        # Risk factors based on medical logic
        risk_scores = (
            (df['gestational_age_weeks'] < 37).astype(int) * 2 +  # Premature
            (df['birth_weight_kg'] < 2.5).astype(int) * 2 +  # Low birth weight
            (df['oxygen_saturation_pct'] < 94).astype(int) * 3 +  # Low oxygen
            (df['bilirubin_level_mg_dl'] > 15).astype(int) * 2 +  # High bilirubin
            df['infection_flag'] * 3 +  # Infection
            df['nicu_admission'] * 2  # NICU admission
        )
        
        # Assign risk based on scores
        risk_threshold = np.percentile(risk_scores, 100 - sample_config['at_risk_ratio'] * 100)
        df['risk_level'] = np.where(risk_scores >= risk_threshold, 'At Risk', 'Healthy')
        
        # Verify distribution
        actual_dist = df['risk_level'].value_counts(normalize=True)
        print(f"✅ Created synthetic dataset with {n_samples} records")
        print(f"   Risk distribution: {actual_dist['Healthy']:.1%} Healthy, {actual_dist['At Risk']:.1%} At Risk")
        
        return df
    
    def load_data(self):
        """
        Try to load existing data or create sample data
        
        Returns:
            pd.DataFrame: Loaded or generated dataset
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
                
                return self.df
                
            except Exception as e:
                print(f"❌ Error loading {data_path}: {e}")
                print("Creating sample data instead...")
        
        # If no dataset found or error loading, create sample data
        self.df = self.create_sample_data()
        return self.df
    
    def explore_data(self):
        """
        Perform exploratory data analysis
        
        Returns:
            dict: Summary statistics and information
        """
        if self.df is None:
            print("❌ No data available.")
            return None
        
        print("\n🔍 Data Exploration:")
        print("="*50)
        
        # Basic info
        info = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.value_counts().to_dict()
        }
        
        print(f"1. Dataset Info:")
        print(f"   Total rows: {info['total_rows']}")
        print(f"   Total columns: {info['total_columns']}")
        
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
        for dtype, count in info['data_types'].items():
            print(f"   {dtype}: {count} columns")
        
        # Show sample
        print("\n4. Sample of data (first 2 rows):")
        print(self.df.head(2))
        
        # Target variable analysis
        if 'risk_level' in self.df.columns:
            print("\n5. Target Variable Analysis:")
            target_dist = self.df['risk_level'].value_counts()
            target_dist_dict = {}
            for level, count in target_dist.items():
                percentage = count / len(self.df) * 100
                target_dist_dict[level] = {'count': count, 'percentage': percentage}
                print(f"   {level}: {count} ({percentage:.1f}%)")
            info['target_distribution'] = target_dist_dict
        
        return info
    
    def preprocess_data(self):
        """
        Preprocess the data for model training
        
        Returns:
            tuple: (X, y) features and target
        """
        print("\n🔧 Preprocessing data...")
        
        if self.df is None:
            print("❌ No data loaded.")
            return None, None
        
        # Create a copy
        df_processed = self.df.copy()
        
        # 1. Drop non-informative columns
        columns_to_drop = [col for col in CONFIG['columns_to_drop'] if col in df_processed.columns]
        
        if columns_to_drop:
            print(f"   Dropping columns: {columns_to_drop}")
            df_processed = df_processed.drop(columns=columns_to_drop)
        
        # 2. Handle categorical variables
        for col in CONFIG['categorical_columns']:
            if col in df_processed.columns:
                if df_processed[col].dtype == 'object':
                    df_processed[col] = df_processed[col].astype('category').cat.codes
                elif df_processed[col].dtype in ['int64', 'float64']:
                    df_processed[col] = df_processed[col].astype(int)
                print(f"   Encoded '{col}'")
        
        # 3. Define target and features
        if 'risk_level' not in df_processed.columns:
            print("❌ 'risk_level' column not found")
            return None, None
        
        # Keep only features that exist in data
        features = [f for f in CONFIG['risk_features'] if f in df_processed.columns]
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
        Scale numerical features using StandardScaler
        
        Returns:
            pd.DataFrame: Scaled features
        """
        print("\n📐 Scaling features...")
        
        if self.X is None:
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
    
    def get_feature_importance_df(self, importances, model_name):
        """
        Create a feature importance DataFrame
        
        Args:
            importances: Feature importance scores
            model_name (str): Name of the model
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.feature_names is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def prepare_new_data(self, input_data):
        """
        Prepare new data for prediction
        
        Args:
            input_data: Dictionary or DataFrame with new data
            
        Returns:
            pd.DataFrame: Prepared and scaled features
        """
        # Convert input to DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data
        else:
            raise ValueError("Input must be a dictionary or DataFrame")
        
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
        
        return input_scaled