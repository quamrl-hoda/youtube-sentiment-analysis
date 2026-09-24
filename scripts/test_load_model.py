import os
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
import pytest


load_dotenv()
 # Set MLflow environment variables
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "").strip()
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "").strip()
REPO_NAME = os.getenv("REPO_NAME", "youtube-sentiment-analysis")

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow"
mlflow.set_tracking_uri(tracking_uri)

@pytest.mark.parametrize("model_name, stage", [
    ("youtube_sentiment_lgbm", "staging"),])

def test_load_latest_staging_model(model_name, stage):
    client = MlflowClient()
    
    # Get the latest version in the specified stage
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None
    
    assert latest_version is not None, f"No model found in the '{stage}' stage for '{model_name}'"

    try:
        # Load the latest version of the model
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Ensure the model loads successfully
        assert model is not None, "Model failed to load"
        print(f"Model '{model_name}' version {latest_version} loaded successfully from '{stage}' stage.")

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")