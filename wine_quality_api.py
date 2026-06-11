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