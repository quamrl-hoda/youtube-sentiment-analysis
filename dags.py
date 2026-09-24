# src/dagshub_config.py
import os
import mlflow
import dagshub
from dotenv import load_dotenv  # Optional: for loading .env file

def setup_dagshub():
    """Initialize DAGsHub and set up MLflow tracking."""
    
    # Load environment variables from .env file (optional but recommended)
    load_dotenv()
    
    # Your DAGsHub credentials
    DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME', 'quamrl-hoda')
    DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')  # Get from environment variable
    REPO_NAME = 'youtube-sentiment-analysis'
    
    # Set credentials as environment variables for MLflow
    os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
    
    # Initialize DAGsHub
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)
    
    # Set the MLflow tracking URI
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow")
    
    print(f"✅ DAGsHub initialized for {DAGSHUB_USERNAME}/{REPO_NAME}")
    
    return mlflow

def set_experiment(experiment_name='dvc-pipeline-runs'):
    """Set or create an MLflow experiment."""
    try:
        mlflow.set_experiment(experiment_name)
        print(f"✅ Experiment set: {experiment_name}")
    except Exception as e:
        print(f"❌ Failed to set experiment: {e}")
        raise