import os
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
import pytest
import pandas as pd
import joblib

load_dotenv()
 # Set MLflow environment variables
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "").strip()
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "").strip()
REPO_NAME = os.getenv("REPO_NAME", "youtube-sentiment-analysis")

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow"
mlflow.set_tracking_uri(tracking_uri)

@pytest.mark.parametrize("model_name, stage, vectorizer_path", [
    ("youtube_sentiment_lgbm", "staging", "artifacts/models/tfidf_vectorizer.pkl"),])

def test_model_with_vectorizer(model_name, stage, vectorizer_path):
    client = MlflowClient()

    # Get the latest version in the specified stage
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None

    assert latest_version is not None, f"No model found in the '{stage}' stage for '{model_name}'"

    try:
        # Load the latest version of the model
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Load the vectorizer
        vectorizer = joblib.load(vectorizer_path)

        # Create a dummy input for the model
        input_text = "hi how are you"
        input_data = vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=vectorizer.get_feature_names_out())  # <-- Use correct feature names

        # Predict using the model
        prediction = model.predict(input_df)

        # Verify the input shape matches the vectorizer's feature output
        assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), "Input feature count mismatch"

        # Verify the output shape (assuming binary classification with a single output)
        assert len(prediction) == input_df.shape[0], "Output row count mismatch"

        print(f"Model '{model_name}' version {latest_version} successfully processed the dummy input.")

    except Exception as e:
        pytest.fail(f"Model test failed with error: {e}")