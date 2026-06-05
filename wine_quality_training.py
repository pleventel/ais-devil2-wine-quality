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