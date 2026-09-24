import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature


# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.dagshub_config import setup_dagshub, set_experiment

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            # Try alternative path
            alt_path = file_path.replace('artifacts/interim', 'artifacts/data/interim')
            if os.path.exists(alt_path):
                logger.debug(f"Using alternative path: {alt_path}")
                file_path = alt_path
            else:
                raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s - Shape: %s', file_path, df.shape)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(file_path: str):
    """Load a trained model from disk."""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Model file not found: {file_path}")
            # Check models directory
            models_dir = 'models'
            if os.path.exists(models_dir):
                logger.error(f"Available models: {os.listdir(models_dir)}")
            raise FileNotFoundError(f"Model not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug(f"✅ Model loaded from {file_path}")
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model from {file_path}: {e}")
        raise

def load_vectorizer(file_path: str):
    """Load a trained vectorizer from disk."""
    try:
        import joblib
        # Use the provided file_path parameter, not hardcoded path
        if not os.path.exists(file_path):
            logger.error(f"Vectorizer file not found: {file_path}")
            # Check models directory
            models_dir = 'models'
            if os.path.exists(models_dir):
                logger.error(f"Available files: {os.listdir(models_dir)}")
            raise FileNotFoundError(f"Vectorizer not found: {file_path}")
        
        vectorizer = joblib.load(file_path)  # FIXED: Use file_path parameter
        logger.debug(f"✅ Vectorizer loaded from {file_path}")
        return vectorizer
    except Exception as e:
        logger.error(f"❌ Error loading vectorizer from {file_path}: {e}")
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        if not os.path.exists(params_path):
            logger.error(f"Params file not found: {params_path}")
            # Try in current directory
            if os.path.exists('params.yaml'):
                params_path = 'params.yaml'
                logger.debug(f"Using params.yaml from current directory")
            else:
                raise FileNotFoundError(f"params.yaml not found")
        
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        # Predict and calculate classification metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.debug('Model evaluation completed')
        logger.debug(f"Test accuracy: {report.get('accuracy', 0):.4f}")

        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Save confusion matrix plot as a file and log it to MLflow
        cm_file_path = f'confusion_matrix_{dataset_name.replace(" ", "_")}.png'
        plt.savefig(cm_file_path)
        mlflow.log_artifact(cm_file_path)
        plt.close()
        logger.debug(f"Confusion matrix saved to {cm_file_path}")
    except Exception as e:
        logger.error(f"Error logging confusion matrix: {e}")

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        # Create a dictionary with the info you want to save
        model_info = {
            'run_id': run_id,
            'model_path': model_path,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        # Save the dictionary as a JSON file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    try:
        setup_dagshub()
        set_experiment('dvc-pipeline-runs')
        
        with mlflow.start_run() as run:
            # Load parameters from YAML file
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params_path = os.path.join(root_dir, 'params.yaml')
            
            # If params.yaml doesn't exist in root, try current directory
            if not os.path.exists(params_path):
                params_path = 'params.yaml'
                
            params = load_params(params_path)

            # Log parameters
            for key, value in params.items():
                if isinstance(value, (dict, list)):
                    mlflow.log_param(key, str(value))
                else:
                    mlflow.log_param(key, value)
            
            # Load model and vectorizer - FIXED: Use correct paths
            model_path = os.path.join('artifacts', 'models', 'lgbm_model.pkl')
            vectorizer_path = os.path.join('artifacts', 'models', 'tfidf_vectorizer.pkl')
            
            logger.debug(f"Looking for model at: {model_path}")
            logger.debug(f"Looking for vectorizer at: {vectorizer_path}")
            
            model = load_model(model_path)
            vectorizer = load_vectorizer(vectorizer_path)

            # Load test data - FIXED: Try multiple possible paths
            possible_paths = [
                os.path.join('artifacts', 'interim', 'test_processed.csv'),
                os.path.join('artifacts', 'data', 'interim', 'test_processed.csv'),
                os.path.join(root_dir, 'artifacts', 'interim', 'test_processed.csv')
            ]
            
            test_data = None
            for path in possible_paths:
                if os.path.exists(path):
                    test_data = load_data(path)
                    logger.debug(f"Found test data at: {path}")
                    break
            
            if test_data is None:
                raise FileNotFoundError("Could not find test_processed.csv in any expected location")

            # Prepare test data
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Create a smaller input example for signature inference (use 5 samples)
            input_example = pd.DataFrame(
                X_test_tfidf[:5].toarray(), 
                columns=vectorizer.get_feature_names_out()
            )

            # Infer the signature
            signature = infer_signature(
                input_example, 
                model.predict(X_test_tfidf[:5])
            )

            # Log model with signature
            mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                signature=signature,
                input_example=input_example
            )

            # Save model info
            model_path_artifact = "lgbm_model"
            save_model_info(run.info.run_id, model_path_artifact, 'experiment_info.json')

            # Log the vectorizer as an artifact - FIXED: Use correct path
            vectorizer_artifact_path = vectorizer_path
            if os.path.exists(vectorizer_artifact_path):
                mlflow.log_artifact(vectorizer_artifact_path)
                logger.debug(f"Vectorizer artifact logged from {vectorizer_artifact_path}")
            else:
                logger.warning(f"Vectorizer file not found for artifact logging: {vectorizer_artifact_path}")

            # Evaluate model and get metrics
            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            # Log classification report metrics for the test data
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    try:
                        mlflow.log_metrics({
                            f"test_{label}_precision": metrics.get('precision', 0),
                            f"test_{label}_recall": metrics.get('recall', 0),
                            f"test_{label}_f1-score": metrics.get('f1-score', 0)
                        })
                    except Exception as e:
                        logger.warning(f"Could not log metrics for {label}: {e}")

            # Log overall accuracy
            if 'accuracy' in report:
                mlflow.log_metric("test_accuracy", report['accuracy'])

            # Log confusion matrix
            log_confusion_matrix(cm, "Test Data")

            # Add important tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "Reddit Comments")
            mlflow.set_tag("run_id", run.info.run_id)
            
            logger.info("✅ Model evaluation completed successfully!")
            logger.info(f"📊 Run ID: {run.info.run_id}")
            logger.info(f"📈 Test accuracy: {report.get('accuracy', 0):.4f}")

    except FileNotFoundError as e:
        logger.error(f"❌ File not found error: {e}")
        print(f"Error: {e}")
        print("\n💡 Tip: Run 'dvc repro' to generate all required files")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to complete model evaluation: {e}")
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()