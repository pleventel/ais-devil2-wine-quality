import mlflow
from mlflow.tracking import MlflowClient
import dagshub

# Initialize connection
REPO_OWNER = "pleventel"
REPO_NAME = "ais-devil2-wine-quality"

dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

tracking_uri = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient()

# Find the experiment
EXPERIMENT_NAME = "Default"
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if not experiment:
    raise ValueError(f"Could not find experiment named '{EXPERIMENT_NAME}' on DagsHub.")

# Look up runs explicitly for this experiment's ID
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.training_r2_score DESC"],
    max_results=1
)

if not runs:
    raise ValueError("No runs found in this experiment to register.")

best_run = runs[0]
best_run_id = best_run.info.run_id
best_r2 = best_run.data.metrics.get("training_r2_score", "N/A")

print(f"🏆 Best Run Found: ID={best_run_id}")
print(f"📊 Best Run R² Score: {best_r2}")

# Register the model to the Model Registry
model_uri = f"runs://{best_run_id}/model"
model_name = "wine-quality"

print(f"Registering model under the name '{model_name}'...")
try:
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=best_run_id
    )
    print(f"✅ Successfully registered! Assigned Version: {model_version.version}")
except Exception as e:
    # Fallback pattern if the model container structure hasn't been initialized yet
    print("Creating new registered model container...")
    client.create_registered_model(model_name)
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=best_run_id
    )
    print(f"✅ Successfully registered! Assigned Version: {model_version.version}")