import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dagshub_config import setup_dagshub, set_experiment

# Authenticate via env vars (no dagshub.auth call needed)
setup_dagshub()
set_experiment()

client = MlflowClient()

MODEL_NAME = "youtube_sentiment_lgbm"   # must EXACTLY match registered model name

# 3. Get latest model version
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

if not versions:
    raise RuntimeError(f"❌ No versions found for model '{MODEL_NAME}'. Did you register it?")

latest_version = max(int(v.version) for v in versions)

# -------------------------------
# 4. Promote to Production
# -------------------------------
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version,
    stage="Production",
    archive_existing_versions=True
)

print(f"✅ Model '{MODEL_NAME}' version {latest_version} promoted to Production")