import os
import sys
from dotenv import load_dotenv
import mlflow

# Force UTF-8 output so emoji print correctly on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# -------------------------------------------------
# Initialize MLflow with DagsHub credentials via env vars
# -------------------------------------------------
def setup_dagshub():
    load_dotenv()

    DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "").strip()
    DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "").strip()
    REPO_NAME = os.getenv("REPO_NAME", "youtube-sentiment-analysis").strip()

    if not DAGSHUB_USERNAME or not DAGSHUB_TOKEN:
        raise RuntimeError("DagsHub credentials not found in environment variables. "
                           "Set DAGSHUB_USERNAME and DAGSHUB_TOKEN in your .env file.")

    # Authenticate via MLflow env vars (no dagshub.auth call needed)
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)

    print(f"MLflow initialized. Tracking URI: {tracking_uri}")

# -------------------------------------------------
# Set / create MLflow experiment (SAFE)
# -------------------------------------------------
def set_experiment(experiment_name="dvc-pipeline-runs"):
    try:
        mlflow.set_experiment(experiment_name)
        print(f"✅ Experiment set: {experiment_name}")
    except Exception as e:
        print(f"⚠️  Could not set experiment '{experiment_name}': {e}")
        raise