# AIS Wine Quality Prediction 🍷

Welcome to your final MLOps task! You are going to build a complete MLOps pipeline from scratch —
data management, model training, experiment tracking, model registry, and a prediction API —
all applied to a real-world wine quality dataset.

You need to apply everything you have learned throughout the course **independently**.
All the exercises so far, especially the taxi ride exercise, are your reference — use them.

> 💡 **Work in pairs.** As always.


## 🎯 What You Are Building

A **FastAPI application** that predicts wine quality based on physicochemical properties of a wine.
A user sends wine properties to your API and gets back a quality prediction.

## 🍷 The Dataset

The file `data/winequality.parquet` is already in this repository. It contains 6.497 wine samples,
both red and white, described by the following physicochemical properties:

| Column                  | Description                                       |
|-------------------------|---------------------------------------------------|
| `fixed_acidity`         | Most acids in wine (tartaric, etc.) in g/L        |
| `volatile_acidity`      | Acetic acid content — high values → vinegar taste |
| `citric_acid`           | Adds freshness and flavour                        |
| `residual_sugar`        | Sugar remaining after fermentation in g/L         |
| `chlorides`             | Salt content                                      |
| `free_sulfur_dioxide`   | Free form of SO₂ — protects against microbes      |
| `total_sulfur_dioxide`  | Total SO₂ (free + bound)                          |
| `density`               | Density of the wine in g/cm³                      |
| `pH`                    | Acidity level (lower = more acidic)               |
| `sulphates`             | Additive that contributes to SO₂                  |
| `alcohol`               | Alcohol content in % vol                          |
| `wine_color`            | `"red"` or `"white"`                              |
| **`quality`**           | **Target — integer score from 3 to 9**            |

### What to predict

The `quality` column is your target. Your model predicts the quality score directly (3–9).

## Resources 

You've learned everything so far. Use the exercises you've done as a reference. By now, you should 
have working versions in your own repositories.

- Git Basics: https://github.com/peterrietzler/ais-dev2il-ais-power-smoothie-recipes
- Zero Configuration & Unit Testing: https://github.com/peterrietzler/ais-dev2il-ais-power-smoothie-maker
- CI Pipelines: https://github.com/peterrietzler/ais-dev2il-ais-power-smoothie-maker-ci
- Docker: https://github.com/peterrietzler/ais-dev2il-ais-power-smoothie-delivery-box
- Observability: https://github.com/peterrietzler/ais-dev2il-smoothie-shop
- MLOPs: https://github.com/peterrietzler/ais-dev2il-mlops-taxi-rides-v2

## 📦 Phase 1: Explore & Version the Data

- Explore the dataset: understand the distribution of `quality`, check for missing values, look at class balance
- Initialise DVC, configure your DagsHub remote, track `data/winequality.parquet`

✅ Only the `.dvc` pointer file lives in Git; the data lives in your DagsHub remote.

## 🧠 Phase 2: Build the Training Script

Create `wine_quality_training.py`. It should:

- Load the data and prepare features and target
- Train a scikit-learn model of your choice
- Evaluate it and print a report
- Save the model as `models/wine_quality_model.pkl`
- Save evaluation metrics to `models/wine_quality_model.metadata.json`

You decide which features to use and which model to train.

✅ Running the script produces a model file and a metadata file.

## 🔬 Phase 3: Experiment Tracking with MLflow

- Configure MLflow tracking against your DagsHub MLflow server
- Wrap your training script with `mlflow.autolog()`
- Run at least **three experiments** with different setups (features, models, hyperparameters)
- Compare the results in the MLflow UI on DagsHub

✅ At least three runs are visible and comparable in the MLflow UI.

## 🏆 Phase 4: Register the Best Model

- Pick your best run and register it in the MLflow Model Registry as `wine-quality`
- Create a `.model-version` file in the repository root containing the version number
- Create a `download_model.py` script that reads `.model-version` and downloads the registered model

✅ `.model-version` is committed to Git; `download_model.py` produces `wine_quality_model.pkl`.

## 🌐 Phase 5: Serve Predictions with FastAPI

Create `wine_quality_api.py` that:

- Loads `wine_quality_model.pkl` at startup
- Exposes a `POST /predict` endpoint accepting wine properties as JSON
- Return the predicted quality score

Use Pydantic to define your input and output. Only include the features you trained on.

Package the API as a Docker image using the multi-stage pattern.

✅ The API runs locally and predictions work via the auto generated FastAPI UI. The Docker image builds successfully.

## ⚙️ Phase 6: GitHub & CI Pipeline

Configure it to be a safe working environment

- Protect the trunk. Make sure that nobody can push to it or merge any PR without checks passing
- Set up branch policies to require PR reviews and passing checks
- Do not allow secrets to be added to Git

Create a GitHub Actions workflow for training:

- Triggers if training relevant data or code has changed. Allows manual dispatch
- Pulls the data with DVC
- Runs `wine_quality_training.py` and logs results to MLflow

Create a GitHub Actions workflow for the API:

- Builds the Docker image and pushes it to the GitHub Container Registry (GHCR)

Make sure that both workflows 

- Run linting on the code
- Run secret scanning and security checks
- Run the unit tests


✅ Every push to `main` triggers both workflows; the Docker image is available on GHCR.

## ⚙️ Phase 7: Infrastructure as Code 

Use Docker Compose to allow running the API locally with a single command.

## 📡 Phase 8: Make the API Observable

Your API is running — but can you tell what it is doing in production? Add observability so you can answer:
*"How many predictions did we serve? How fast? What quality scores did we return?"*

Extend the FastAPI application:

- Log each prediction request: the input features and the predicted quality score
- Expose REST call metrics for Prometheus

Extend `docker-compose.yml` to run the full observability stack alongside your API:

- Add Prometheus to scrape metrics from the API
- Add Grafana to visualise the metrics

✅ Running `docker compose up` starts the API, Prometheus, and Grafana. Prediction requests are logged and metrics are visible in Grafana.

## ✅ Submission Checklist

### Version Control & Collaboration
- [ ] Code is on GitHub
- [ ] Commits show a meaningful history — not one big final commit
- [ ] Work was done in pairs: commits and pull requests show contributions from both partners
- [ ] Branch protection rules are in place to prevent direct pushes to `main` and require PR reviews
- [ ] Pre-commit hooks prevent that secrets are committed to Git

### Zero Configuration
- [ ] `uv` is used to manage the project — anyone can run it without manual setup

### Data Management
- [ ] `data/winequality.parquet` is tracked with DVC and pushed to your DagsHub remote
- [ ] No large data files or model files are committed to Git

### Experiment Tracking & Model Registry
- [ ] Training logs runs to MLflow on DagsHub
- [ ] At least 3 runs are visible and comparable in the MLflow UI
- [ ] The best model is registered in the MLflow Model Registry

### Testing
- [ ] Unit tests exist for meaningful parts of the code (e.g. feature preparation, prediction logic)
- [ ] Tests can be run with a single command

### API & Docker
- [ ] A FastAPI application exposes a `POST /predict` endpoint
- [ ] The API is packaged as a Docker image
- [ ] The Docker image is built and pushed to the GitHub Container Registry (GHCR) by CI

### CI Pipeline
- [ ] A GitHub Actions workflow trains the model and logs results to MLflow
- [ ] A Docker image or the API is built and pushed to the GitHub Container Registry (GHCR)
- [ ] The pipeline runs linting on the code
- [ ] The pipeline includes security and secret scanning
- [ ] The pipeline runs the unit tests and prints nice, human readable test results

### IaC
- [ ] A `docker-compose.yml` file is provided to run the API locally with Docker

### Observability
- [ ] The FastAPI app logs each prediction request (input features + predicted quality score) using structured logging
- [ ] The API exposes a `/metrics` endpoint for Prometheus
- [ ] `docker-compose.yml` includes Prometheus and Grafana services
- [ ] Running `docker compose up` starts the full stack and metrics are visible in Grafana

> 🍷 *"In wine there is wisdom, in beer there is freedom, in water there are bacteria."*  
> *(Misattributed to Benjamin Franklin — but the MLOps pipeline is real.)*
