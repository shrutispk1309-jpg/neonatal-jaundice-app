#!/usr/bin/env python3
"""
Neonatal Health Risk Prediction Model
Main execution script - runs the complete pipeline
"""

import os
import sys
import warnings
import traceback
from datetime import datetime

from config import CONFIG, PATHS
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from visualizer import Visualizer

warnings.filterwarnings('ignore')


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


def create_directories():
    """Create necessary directories if they don't exist"""
    for path_name, path in PATHS.items():
        os.makedirs(path, exist_ok=True)
        print(f"✅ Created directory: {path}")


def save_model_artifacts(data_processor, model_trainer, visualizer):
    """Save model and related artifacts"""
    try:
        import joblib
        
        # Create a dictionary with all model artifacts
        model_artifacts = {
            'best_model': model_trainer.best_model,
            'scaler': data_processor.scaler,
            'label_encoder': data_processor.label_encoder,
            'feature_names': data_processor.feature_names,
            'best_model_name': model_trainer.best_model_name,
            'evaluation_results': model_trainer.evaluation_results,
            'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': CONFIG
        }
        
        # Save to file
        save_path = os.path.join(PATHS['models'], CONFIG['model_save_path'])
        joblib.dump(model_artifacts, save_path)
        print(f"✅ Model artifacts saved to {save_path}")
        
        return save_path
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return None


def print_summary(data_processor, model_trainer):
    """Print comprehensive summary of the pipeline execution"""
    print("\n" + "="*80)
    print("📋 PIPELINE EXECUTION SUMMARY")
    print("="*80)
    
    print(f"\n📊 Data Summary:")
    print(f"   - Total samples: {len(data_processor.df)}")
    print(f"   - Features used: {len(data_processor.feature_names)}")
    print(f"   - Target classes: {list(data_processor.label_encoder.classes_)}")
    
    print(f"\n🏆 Best Model: {model_trainer.best_model_name.upper()}")
    
    if model_trainer.best_model_name in model_trainer.evaluation_results:
        best_metrics = model_trainer.evaluation_results[model_trainer.best_model_name]
        print(f"\n   Performance Metrics:")
        print(f"   - Accuracy: {best_metrics.get('accuracy', 0):.4f}")
        print(f"   - Precision: {best_metrics.get('precision', 0):.4f}")
        print(f"   - Recall: {best_metrics.get('recall', 0):.4f}")
        print(f"   - F1-Score: {best_metrics.get('f1', 0):.4f}")
        
        if 'roc_auc' in best_metrics:
            print(f"   - ROC-AUC: {best_metrics.get('roc_auc', 0):.4f}")
        
        if 'recall_minority' in best_metrics:
            print(f"\n   At Risk Class Performance:")
            print(f"   - Precision: {best_metrics.get('precision_minority', 0):.4f}")
            print(f"   - Recall: {best_metrics.get('recall_minority', 0):.4f}")
            print(f"   - F1-Score: {best_metrics.get('f1_minority', 0):.4f}")
    
    print("\n" + "="*80)


def main():
    """
    Main function - runs the complete pipeline
    """
    print("\n" + "🎯" * 40)
    print("🎯 NEONATAL HEALTH RISK PREDICTION SYSTEM")
    print("🎯" * 40)
    print(f"🎯 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create necessary directories
    create_directories()
    
    # Initialize components
    data_processor = DataProcessor(random_state=CONFIG['random_state'])
    model_trainer = ModelTrainer(random_state=CONFIG['random_state'])
    visualizer = Visualizer()
    
    try:
        # STEP 1: Load Data
        print("\n📥 STEP 1: Loading Data")
        print("-" * 50)
        data_processor.load_data()
        
        # STEP 2: Explore Data
        print("\n🔍 STEP 2: Exploring Data")
        print("-" * 50)
        data_processor.explore_data()
        
        # STEP 3: Preprocess Data
        print("\n🔧 STEP 3: Preprocessing Data")
        print("-" * 50)
        X, y = data_processor.preprocess_data()
        if X is None or y is None:
            print("❌ Preprocessing failed.")
            return None
        
        # STEP 4: Scale Features
        print("\n📐 STEP 4: Scaling Features")
        print("-" * 50)
        data_processor.scale_features()
        
        # Pass class names to model trainer
        model_trainer.set_classes(data_processor.label_encoder.classes_)
        model_trainer.set_label_encoder(data_processor.label_encoder)
        
        # STEP 5: Split Data
        print("\n✂️  STEP 5: Splitting Data")
        print("-" * 50)
        model_trainer.split_data(data_processor.X_scaled, data_processor.y)
        
        # STEP 6: Handle Imbalance
        print("\n⚖️  STEP 6: Handling Class Imbalance")
        print("-" * 50)
        model_trainer.handle_imbalance()
        
        # STEP 7: Initialize Models
        print("\n🤖 STEP 7: Initializing Models")
        print("-" * 50)
        model_trainer.initialize_models()
        
        # STEP 8: Train Models
        print("\n🏋️‍♂️ STEP 8: Training Models")
        print("-" * 50)
        model_trainer.train_models()
        
        # STEP 9: Evaluate Models
        print("\n📈 STEP 9: Evaluating Models")
        print("-" * 50)
        model_trainer.evaluate_models()
        
        # STEP 10: Analyze Feature Importance
        print("\n🔍 STEP 10: Analyzing Feature Importance")
        print("-" * 50)
        importances = model_trainer.get_feature_importance()
        if importances is not None:
            importance_df = data_processor.get_feature_importance_df(
                importances, model_trainer.best_model_name
            )
            
            # Plot feature importance
            save_path = os.path.join(PATHS['visualizations'], 'feature_importance.png')
            visualizer.plot_feature_importance(
                importance_df, 
                model_trainer.best_model_name,
                top_n=10,
                save_path=save_path
            )
        
        # STEP 11: Visualize Results
        print("\n📊 STEP 11: Visualizing Results")
        print("-" * 50)
        
        # Get predictions from best model
        best_model = model_trainer.best_model
        if best_model is not None:
            y_pred = best_model.predict(model_trainer.X_test)
            y_pred_proba = best_model.predict_proba(model_trainer.X_test) if hasattr(best_model, 'predict_proba') else None
            
            # Create comprehensive visualization
            visualizer.plot_all_results(
                y_true=model_trainer.y_test,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba,
                class_names=data_processor.label_encoder.classes_,
                evaluation_results=model_trainer.evaluation_results,
                best_model_name=model_trainer.best_model_name,
                feature_importance_df=importance_df if 'importance_df' in locals() else None,
                save_dir=PATHS['visualizations']
            )
        
        # STEP 12: Save Model
        print("\n💾 STEP 12: Saving Model")
        print("-" * 50)
        save_model_artifacts(data_processor, model_trainer, visualizer)
        
        # Print Summary
        print_summary(data_processor, model_trainer)
        
        # Example Prediction
        print("\n" + "🔮" * 40)
        print("🔮 EXAMPLE PREDICTION")
        print("🔮" * 40)
        
        # Create example input
        example_input = {}
        for feature in data_processor.feature_names[:10]:
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
        print(f"Using {len(data_processor.feature_names)} features")
        
        # Prepare and predict
        try:
            input_scaled = data_processor.prepare_new_data(example_input)
            prediction = best_model.predict(input_scaled)
            probability = best_model.predict_proba(input_scaled)
            
            prediction_label = data_processor.label_encoder.inverse_transform(prediction)[0]
            
            print(f"\n✅ Prediction Result:")
            print(f"   Risk Level: {prediction_label}")
            print(f"   Confidence: {probability[0][prediction[0]]:.2%}")
            
            print(f"\n📊 All Probabilities:")
            for i, label in enumerate(data_processor.label_encoder.classes_):
                print(f"   {label}: {probability[0][i]:.2%}")
                
        except Exception as e:
            print(f"⚠️ Could not make example prediction: {e}")
        
        print("\n" + "✅"*40)
        print("✅" + " PIPELINE COMPLETED SUCCESSFULLY! ".center(78, "=") + "✅")
        print("✅"*40)
        print(f"✅ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            'data_processor': data_processor,
            'model_trainer': model_trainer,
            'visualizer': visualizer
        }
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user.")
        return None
    except Exception as e:
        print(f"\n❌ An error occurred during pipeline execution: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the pipeline
    result = main()
    
    if result:
        print("\n🎉 You can now use the trained model for predictions!")
        print("   Check the 'models' and 'visualizations' folders for outputs.")