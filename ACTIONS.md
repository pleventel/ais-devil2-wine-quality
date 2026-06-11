# What is actually done?


## Phase 0: Set up the `pyproject.toml` file
The `.toml` file is a configuration file giving the project information about what requirements are needed, also installing these dependencies at this time. 

In this case, now only the `dcv` package will be added for now, and other dependencies only throughout the project.

Add the following to the `pyproject.toml` file:
~~~toml
[project]
name = "ais-dev2il-wine-quality"
version = "0.1.0"
description = "The final project to prepare for the exam."
readme = "README.md"
requires-python = ">=3.13"
# the Python modules used in the project has to be listed here
dependencies = []

[dependency-groups]
dev = [
    "dvc[s3]>=3.67.1",
]
~~~


## 📦 Phase 1: Explore & Version the Data
At first, explore the dataset, `winequality.parquet`. Then set up the data version control in the following steps.

### Set up Data Version Control
- Create your DagsHub repository&mdash;Click "**Create Repository**" → "**Connect a repository**" → select your GitHub fork.
- Initialise dvc with: 
~~~
uv run dvc init
uv run dvc config core.autostage true 
~~~

Now we created a `.dvc` folder to stage the data files.

### Configuration
1. Register remote using S3 protocol.
2. Add location of storage.
3. Set token as S3 access key (and secret access key).
4. Mark this remote as default.

In this case, the remote will be called `origin`.
~~~
uv run  dvc remote add origin s3://dvc
uv run dvc remote modify origin endpointurl https://dagshub.com/<YOUR USERNAME>/<YOUR REPO>.s3
uv run dvc remote modify origin --local access_key_id <YOUR TOKEN>
uv run dvc remote modify origin --local secret_access_key <YOUR TOKEN>
un run dvc remote default origin
~~~

<span style="color:lightblue; font-style:italic">Doing everything correctly, should result in no console output.</span>

### Track Parquet files
As the data(`data/winequality.parquet`) was already part of the forked repository, we first have to remove this from the tracking of Git (if this is not done, the during the setup of the tracking with dvc this will be raised as an error).

~~~
git rm -r --cached 'data/winequality.parquet'
git commit -m "Stop tracking data/winequality.parquet"
~~~

Now we can create a pointer file for each (in this case only one for now) `.parquet` file(s). Also adds automatically to `data/*.parquet`
~~~
uv run dvc add data/*.parquet
~~~

Add the data folder to Git:
~~~
git add data
git status
~~~
At this point, we should see that there is the `data/.gitignore` and the `winequality.parquet.dvc` file needing to be commited. Latter is the one providing information to `git` about where the original `.parquet` file(s) can be found.
    
### Check what DCV want to push
~~~
uv run dvc status --cloud
~~~

### Delete and resync data in the `data` folder
~~~
rm data/*.parquet
uv run dvc pull
~~~


## 🧠 Phase 2: Build the Training Script
### Create new branch
At first, we will create a new branch and work on that for the model training.
~~~
git checkout -b training
~~~

### Initializations
Then create the file `wine_quality_training.py`.

The very first step is to import the data, which will be done with `pandas`. While running this code, an error will be raised as the dependencies `pyarrow` and `fastparquet` are missing, so we will add these to the dependencies of the `pyproject.toml` file. Other than this, we will also need to add `numpy` for calculations and `sklearn` in order to train the model and then evaluate it. If later more dependencies are needed, just add it here and then rerun the file.
~~~
dependencies = [
    "numpy>=1.24.0",
    "pandas==2.2.3",
    "pyarrow==19.0.1",
    "fastparquet==2024.11.0",
    "scikit-learn>=1.8.0"
]
~~~
After this, we also have to rerun this file with `uv sync`.

### Training model & evaluation
The exercise asks us to train a freely chosen model on this dataset. In this example, Random Forest Regressor will be used.

Use this Python code for example:
~~~python
import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create output directory for the model
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, "wine_quality_model.pkl")
metadata_path = os.path.join(output_dir, "wine_quality_model.metadata.json")

# Load the data
print("Loading wine quality data...")
data = pd.read_parquet("data/winequality.parquet")
X = data.iloc[:, :-1]
y = data["quality"] # the quality is the last column

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for better performance and consistency
print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training model on Random Forest Regressor
print("Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Evaluation
print("Training successfull.\nEvaluation...")
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Evaluation Report ===")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE):  {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared Score (R²):      {r2:.4f}")
print("==========================\n")

# Save the model
model_pipeline = {
    "scaler": scaler,
    "model": model,
    "feature_names": list(X.columns)
}

print(f"Saving model pipeline to {model_path}...")
with open(model_path, "wb") as f:
    pickle.dump(model_pipeline, f)

# Save evaluation metrics to metadata
metadata = {
    "model_type": "RandomForestRegressor",
    "features_used": list(X.columns),
    "metrics": {
        "mean_absolute_error": round(mae, 6),
        "mean_squared_error": round(mse, 6),
        "root_mean_squared_error": round(rmse, 6),
        "r2_score": round(r2, 6)
    }
}

print(f"Saving metadata to {metadata_path}...")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print("Process complete!")
~~~
Now, let's run this code with `uv run wine_quality_training.py`.

### Wrap up: commit & merge branch
If everything looks fine, now it's time to commit our code. As now we are on our `training` branch, we will commit our changes here and push also to this branch. Then, as for now we are finished with this step, we will merge the branch back to `main`.
~~~
git add wine_quality_training.py 
git commit -m "Final training model"
git push
~~~
Now let's check if everything went right with `git status`.

#### Merge branches
~~~
git checkout main
git merge origin/training
git push
git branch -d training
~~~


## 🔬 Phase 3: Experiment Tracking with MLflow
### Initializations
As we did it until, we will also create a new branch now for setting up the tracking phase.
~~~
git checkout -b tracking
~~~
And now let's set up the new requirements in the `pyproject.toml` file. Add to the dependencies section the followings:
~~~
"mlflow>=2.0.0",
"optuna>=3.0.0"
~~~
Then rerun the file with `uv sync`

### Linking to DagsHub
Open your Ubuntu/WSL terminal and run the following commands.
~~~
export MLFLOW_TRACKING_URI="https://dagshub.com/YOUR_DAGSHUB_USERNAME/YOUR_REPO_NAME.mlflow"
export MLFLOW_TRACKING_USERNAME="YOUR_DAGSHUB_USERNAME"
export MLFLOW_TRACKING_PASSWORD="YOUR_DAGSHUB_TOKEN_OR_PASSWORD"
~~~
Of course, insert your specific entries for `YOUR_DAGSHUB_USERNAME`, `YOUR_REPO_NAME` and `YOUR_DAGSHUB_TOKEN`. You can get your token from the DagsHub's website by going to **Your Settings** and then to **Tokens**. Copy your token from there.

### Update your python script with autologging
Go to the already existing `wine_quality_training.py` file and update it with the following codes:
1. Import `mlflow` and `dagshub 
    ~~~
    import mlsflow
    import dagshub
    ~~~
2. Initialize DagsHub & Enable Autologging
    ~~~python
    # Replace with your actual DagsHub username and repo name
    dagshub.init(repo_owner="YOUR_DAGSHUB_USERNAME", repo_name="YOUR_REPO_NAME", mlflow=True)
    mlflow.autolog()
    ~~~
3. Define the preprocessing function<br>*This part of the code already exists. Look for it around `line 25`.*
    ~~~python
    def load_and_preprocess():
        print("Loading wine quality data...")
        data = pd.read_parquet("data/winequality.parquet")
        X = data.iloc[:, :-1]
        y = data["quality"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Scaling data...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns
    ~~~
4. Define the new training setup<br>*Change the training code section to this.*
    ~~~python
    def run_experiment(run_name, model_type="rf", n_estimators=100, max_depth=None):
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = load_and_preprocess()
        
        # Explicitly start a named MLflow run
        with mlflow.start_run(run_name=run_name):
            
            # Select Model based on Experiment setup
            if model_type == "rf":
                print(f"Training RandomForestRegressor ({n_estimators} estimators, max_depth={max_depth})...")
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            elif model_type == "extratrees":
                print(f"Training ExtraTreesRegressor ({n_estimators} estimators, max_depth={max_depth})...")
                model = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                
            model.fit(X_train_scaled, y_train)
    ~~~
    Leave the whole evaluation section as is, but at the end of your code, change the model saving part to the following (still inside the newly defined `run_experiment` function):
    ~~~python
   # Save model
    model_pipeline = {
        "scaler": scaler,
        "model": model,
        "feature_names": list(feature_names)
    }
    
    model_path = os.path.join(output_dir, f"wine_model_{run_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_pipeline, f)
        
    # Explicitly tag the model archetype in MLflow UI
    mlflow.set_tag("model_archetype", model_type)
   ~~~
5. Set up the code so it can actually run<br>*At the end of the code, instead of printing that the code is finished, insert the following code block:
    ~~~python
   if __name__ == "__main__":
        # Ensure you replace YOUR_DAGSHUB_USERNAME and YOUR_REPO_NAME on line 14 before running!
    
        # 🧪 Experiment 1: Baseline Random Forest (Similar to your original script)
        run_experiment(run_name="rf_baseline", model_type="rf", n_estimators=100, max_depth=None)
    
        # 🧪 Experiment 2: Shallow Random Forest (Pruned depth to prevent overfitting)
        run_experiment(run_name="rf_shallow", model_type="rf", n_estimators=150, max_depth=8)
    
        # 🧪 Experiment 3: Alternative Algorithm (ExtraTrees Regressor)
        run_experiment(run_name="extratrees_variant", model_type="extratrees", n_estimators=100, max_depth=12)
    
        print("All 3 Phase 3 experiments completed successfully!")
   ~~~
We are finished with this phase basically.

### Save changes and run the code
At first, make sure that Dagshub is active. Run this: `uv add dagshub`, and then rerun the python script with `uv run wine_quality_training.py`.

If everything worked out, let's commit our changes to github and close this branch.


## 🏆 Phase 4: Register the Best Model
## Register the best run
Create the following script as `register.py`
~~~python
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

# Initialize connection
REPO_OWNER = "YOUR_DAGSHUB_USERNAME"
REPO_NAME = "YOUR_REPO_NAME"

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
~~~
Now let's run this with `uv run python register.py`.

Note down the Version number printed at the end of the terminal loop (usually 1 if this is your first registration).

### Create the tracking file
Create the file `.model-version` at the root of the repository as follows:
1. Type in the terminal `nano .model-version`. Now a new window should pop up in the terminal.
2. Type in the intiger number assigned in to the registration (should be found at the last row of the running of the `registration.py` file.).
3. Save and close this with `Ctrl+O`, `Enter`, `Ctrl+X`.

### Let's get to the `.pkl` file
The following script reads the target file version from your local environment configuration, pulls the compiled pickle pipeline out of DagsHub's storage registry, and unwraps it locally as wine_quality_model.pkl.

Save the following Python code as `download_model.py`
~~~python
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
   REPO_OWNER = "YOUR_DAGSHUB_USERNAME"
   REPO_NAME = "YOUR_REPO_NAME"
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
~~~
As always, don't forget to change the placeholder tets to your DagsHub username and to the name of your repository.

And now run this code with `uv run python download_model.py`.

As a last step, let's verify if the `.pkl` file was really generated at the root library.
~~~
ls -la wine_quality_model.pkl
~~~

### Save the model and commit to GitHub
~~~
git add .model-version
git add download_model.py
git commit -m "Complete phase 4 model registration and downloader setup"
git push origin main
~~~


## 🌐 Phase 5: Serve Predictions with FastAPI
### Initialization
As it was done also until now, set up a new branch as `serve-predictions` and modify the `pyproject.toml` file.
In the dependencies part of the `pyproject.toml` file, replace the dependencies to the following:
~~~toml
dependencies = [
    "numpy>=1.24.0",
    "pandas==2.2.3",
    "pyarrow==19.0.1",
    "fastparquet==2024.11.0",
    "scikit-learn>=1.8.0",
    "mlflow>=2.0.0",
    "optuna>=3.0.0",
    "dagshub>=0.3.0",
    "requests>=2.31.0",
    "fastapi>=0.110.0",
    "uvicorn>=0.28.0",
    "pydantic>=2.6.0"
]
~~~
As it was done also until now, run the file with `uv sync`.

### Implement the FastAPI application
This application uses FastAPI's `lifespan` event handler to load the model pipeline into memory once at startup, preventing expensive file I/O operations on every incoming request.

Create a file named `wine_quality_api.py`:
~~~python
import os
import pickle
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Global dictionary to store the loaded pipeline components
model_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Loads the serialized model pipeline into memory before accepting requests.
    """
    model_path = "wine_quality_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find '{model_path}' in the root directory. "
            f"Please execute 'uv run python download_model.py' first."
        )
    
    print(f"Loading native model pipeline from {model_path}...")
    with open(model_path, "rb") as f:
        # Note: If your Phase 3 pipeline saved the raw sklearn model directly,
        # it will be unpacked directly here.
        model_artifacts["pipeline"] = pickle.load(f)
        
    print("Model pipeline loaded successfully. API ready for inference.")
    yield
    # Clean up on shutdown if necessary
    model_artifacts.clear()


app = FastAPI(
    title="Wine Quality Prediction API",
    description="Production endpoint serving scikit-learn regressor inferences.",
    version="1.0.0",
    lifespan=lifespan
)

# ─── PYDANTIC SCHEMAS ────────────────────────────────────────────────────────
# Define exactly the independent physicochemical features your dataset provides.
# Adjust the names below if your parquet features differ slightly (e.g., casing).

class WineFeaturesInput(BaseModel):
    fixed_acidity: float = Field(..., alias="fixed acidity", example=7.4)
    volatile_acidity: float = Field(..., alias="volatile acidity", example=0.70)
    citric_acid: float = Field(..., alias="citric acid", example=0.00)
    residual_sugar: float = Field(..., alias="residual sugar", example=1.9)
    chlorides: float = Field(..., example=0.076)
    free_sulfur_dioxide: float = Field(..., alias="free sulfur dioxide", example=11.0)
    total_sulfur_dioxide: float = Field(..., alias="total sulfur dioxide", example=34.0)
    density: float = Field(..., example=0.9978)
    pH: float = Field(..., example=3.51)
    sulphates: float = Field(..., example=0.56)
    alcohol: float = Field(..., example=9.4)

    class Config:
        populate_by_name = True


class PredictionOutput(BaseModel):
    predicted_quality: float = Field(..., example=5.63)


# ─── ENDPOINTS ───────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"status": "healthy", "service": "wine-quality-service"}


@app.post("/predict", response_model=PredictionOutput, status_code=200)
def predict_quality(payload: WineFeaturesInput):
    """
    Accepts raw physical wine criteria features, structures them into a matching DataFrame format,
    runs them through the pre-loaded pipeline execution loop, and returns the computed score.
    """
    if "pipeline" not in model_artifacts:
        raise HTTPException(status_code=503, detail="Model pipeline is uninitialized.")
        
    try:
        # Convert Pydantic payload safely into a dictionary using original dataset string keys
        raw_data = payload.model_dump(by_alias=True)
        
        # Convert dictionary to a 2D Pandas DataFrame to preserve feature structure configuration
        input_df = pd.DataFrame([raw_data])
        
        # Extract native ML pipeline from storage cache
        pipeline = model_artifacts["pipeline"]
        
        # Check if the pipeline contains our manual scaling dictionary from your Phase 2/3
        if isinstance(pipeline, dict) and "model" in pipeline:
            # Reconstruct scaling if manually bundled
            scaler = pipeline["scaler"]
            model = pipeline["model"]
            scaled_features = scaler.transform(input_df)
            prediction = model.predict(scaled_features)
        else:
            # If mlflow.autolog tracked a unified scikit-learn Pipeline object directly
            prediction = pipeline.predict(input_df)
            
        # Extract scalar float prediction result out of the output array
        quality_score = float(prediction[0])
        
        return PredictionOutput(predicted_quality=round(quality_score, 4))
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference pipeline execution error: {str(e)}")
~~~

### Test it locally
1. Launch a service pipeline with:
   ~~~
   uv run uvicorn wine_quality_api:app --host 127.0.0.1 --port 8000 --reload
   ~~~
2. Look up the local website http://127.0.0.1:8000/docs.
3. Expand the `POST/predict` endpoint block, click **Try it out**, modify any numeric values in the generated JSON template body, and click **Execute**. Confirm you receive a clean HTTP 200 response with a valid `predicted_quality` score.

### Write the `DOCKERFILE`
A multi-stage build separates the environment used for compiling and gathering project dependencies from the runner stage. This significantly reduces the size of your final production image by excluding tooling like compiler utilities or `uv` cache structures.

Create the `DOCKERFILE` with the following implementation.
~~~
# ─── STAGE 1: BUILD DEPENDENCIES ─────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

# Enable bytecode compilation for faster application runtime execution
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy only dependency description specifications first to optimize layer caching
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --no-dev

# ─── STAGE 2: FINAL RUNTIME ENVIRONMENT ──────────────────────────────────────
FROM python:3.13-slim-bookworm AS runner

WORKDIR /app

# Prevent Python from writing .pyc files and force unbuffered logging streams
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy the pre-compiled virtual environment directly from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Pre-set path targeting to the isolated virtual environment binaries
ENV PATH="/app/.venv/bin:$PATH"

# Copy application processing tools and downloaded model asset binary directly
COPY wine_quality_api.py /app/wine_quality_api.py
COPY wine_quality_model.pkl /app/wine_quality_model.pkl

# Expose server operational listening port
EXPOSE 8000

# Run service container via production ASGI configuration rules
CMD ["uvicorn", "wine_quality_api.py:app", "--host", "0.0.0.0", "--port", "8000"]
~~~

### Build and run the Docker image
1. Build the production container:
   ~~~
   docker build -t wine-quality-api:latest .
   ~~~
2. Run the container locally:
   ~~~
   docker run -p 8000:8000 wine-quality-api:latest
   ~~~
3. Re-verify the prediction processing loop by targeting http://127.0.0.1:8000/docs in your browser.

### Commit changes
~~~
TODO
~~~


## ⚙️ Phase 6: GitHub & CI Pipeline
### Enforce branch protection rules
Before configuring code automation, secure your main branch within your GitHub repository settings to enforce clean code integration.

1. Navigate to your repository on GitHub, click the **Settings** tab, and select **Branches** from the left-hand sidebar. 
2. Click **Add branch ruleset**.
3. Configure the following protection policies:
   - **Require a pull request before merging**: Check this, and ensure Require approvals is enabled (set to at least 1 reviewer).
   - **Require status checks to pass**: Check this, and add your upcoming workflow job names (e.g., `lint-and-test`) once they are created.

