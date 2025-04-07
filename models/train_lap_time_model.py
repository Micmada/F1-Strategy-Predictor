# Updated `models/train_lap_time_model.py` (track-specific + prints model score)
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import sys

# Read track name from command-line or default
track = sys.argv[1] if len(sys.argv) > 1 else "silverstone"

# Load data for specified track
input_path = f"data/tyre_data_{track.lower()}.csv"
df = pd.read_csv(input_path)
df = df.dropna(subset=["LapTime", "TyreLife", "Compound", "LapNumber"])
df["LapTime"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
df["Compound"] = df["Compound"].str.upper()

# Prepare features and target
X = df[["TyreLife", "Compound", "LapNumber"]]
y = df["LapTime"].values

# One-hot encode compound
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
compound_encoded = encoder.fit_transform(X[["Compound"]])
compound_df = pd.DataFrame(compound_encoded, columns=encoder.get_feature_names_out(["Compound"]))

X_final = pd.concat([
    X.drop(columns=["Compound"]).reset_index(drop=True),
    compound_df.reset_index(drop=True)
], axis=1)

# Save encoder
encoder_out = f"models/compound_encoder_{track.lower()}.pkl"
joblib.dump(encoder, encoder_out)

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train XGBoost model on GPU
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    "tree_method": "hist",
    "device": "cuda",
    "objective": "reg:squarederror"
}
model = xgb.train(params, dtrain, num_boost_round=100)

# Evaluate model performance
preds = model.predict(dtest)
mse = np.mean((preds - y_test) ** 2)
r2 = 1 - (np.sum((y_test - preds)**2) / np.sum((y_test - np.mean(y_test))**2))
print(f"Model Evaluation for {track.title()}:")
print(f"  - Mean Squared Error: {mse:.3f}")
print(f"  - R^2 Score: {r2:.3f}")

# Save model
model_out = f"models/lap_time_model_{track.lower()}.json"
model.save_model(model_out)
print(f"✅ Model for {track.title()} trained and saved: {model_out}")