import os
import pickle
import requests
import mlflow
import dagshub
from mlflow import MlflowClient


def download_registered_model():
    # 1. Read target version from metadata file
    version_file = ".model-version"
    if not os.path.exists(version_file):
        raise FileNotFoundError(f"Missing required metadata configuration file: {version_file}")

    with open(version_file, "r") as f:
        model_version = f.read().strip()

    print(f"Reading target model metadata... Found version: {model_version}")

    # 2. Setup DagsHub variables
    REPO_OWNER = "pleventel"
    REPO_NAME = "ais-devil2-wine-quality"
    model_name = "wine-quality"

    # Fetch token securely using DagsHub auth handler
    token = dagshub.auth.get_token()

    # Initialize client matching your tracking environment
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    tracking_uri = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    # 3. Retrieve the run ID linked to this version
    print(f"Fetching '{model_name}' (v{model_version}) details from DagsHub Registry...")
    version_details = client.get_model_version(name=model_name, version=model_version)
    run_id = version_details.run_id
    print(f"Linked Run ID: {run_id}")

    # 4. Construct direct download endpoint URL bypassing MLflow entirely
    # DagsHub maps run artifacts under this public endpoint
    download_url = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}/raw/main/artifacts/{run_id}/model/model.pkl"
    output_filename = "wine_quality_model.pkl"

    print("Downloading model binary via direct storage stream...")

    # Send an authenticated request to grab the model artifact
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(download_url, headers=headers)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download file from storage bucket. "
            f"Status Code: {response.status_code}, Reason: {response.text}"
        )

    # 5. Extract binary file and verify serialization integrity
    print("Verifying pipeline structure integrity...")
    native_pipeline = pickle.loads(response.content)

    print(f"Saving deployment asset directly to ./{output_filename}")
    with open(output_filename, "wb") as f:
        pickle.dump(native_pipeline, f)

    print("🎯 Model extraction and download finished successfully!")


if __name__ == "__main__":
    download_registered_model()