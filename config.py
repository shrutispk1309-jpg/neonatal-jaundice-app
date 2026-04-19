"""
Configuration settings for Neonatal Health Risk Prediction Model
"""

# Project configuration
CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'model_save_path': 'trained_neonatal_model.pkl',
    
    # Data file patterns to look for
    'data_keywords': ['newborn', 'neonatal', 'baby', 'infant', 'health', 'risk', 'jaundice', 'medical'],
    
    # Model parameters
    'random_forest_params': {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced_subsample'
    },
    
    'gradient_boosting_params': {
        'n_estimators': 150,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8
    },
    
    'svm_params': {
        'probability': True,
        'class_weight': 'balanced',
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale'
    },
    
    'logistic_regression_params': {
        'max_iter': 1000,
        'class_weight': 'balanced'
    },
    
    # Feature columns for risk prediction
    'risk_features': [
        'gestational_age_weeks', 'birth_weight_kg', 'birth_length_cm',
        'birth_head_circumference_cm', 'age_days', 'weight_kg',
        'length_cm', 'head_circumference_cm', 'heart_rate_bpm',
        'respiratory_rate_bpm', 'oxygen_saturation_pct', 'body_temperature_c',
        'feeding_frequency_per_day', 'urine_output_per_day',
        'bilirubin_level_mg_dl', 'infection_flag', 'nicu_admission', 'gender'
    ],
    
    # Columns to drop during preprocessing
    'columns_to_drop': ['baby_id', 'name', 'date'],
    
    # Categorical columns to encode
    'categorical_columns': ['gender', 'infection_flag', 'nicu_admission'],
    
    # Sample data generation parameters
    'sample_data': {
        'n_samples': 3000,
        'healthy_ratio': 0.867,  # 86.7% healthy
        'at_risk_ratio': 0.133,   # 13.3% at risk
        'gestational_age_mean': 38.5,
        'gestational_age_std': 2,
        'birth_weight_mean': 3.2,
        'birth_weight_std': 0.5,
        'birth_length_mean': 50,
        'birth_length_std': 3,
        'head_circumference_mean': 34,
        'head_circumference_std': 2
    }
}

# File paths
PATHS = {
    'models': 'models/',
    'data': 'data/',
    'outputs': 'outputs/',
    'visualizations': 'visualizations/'
}

# Visualization settings
VIZ_CONFIG = {
    'figure_size': (18, 10),
    'color_palette': 'viridis',
    'dpi': 100,
    'style': 'seaborn-v0_8-darkgrid'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}