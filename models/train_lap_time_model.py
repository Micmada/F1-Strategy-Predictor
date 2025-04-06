import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np

df = pd.read_csv("data/tyre_data_silverstone_2023.csv")
df = df.dropna(subset=["LapTime", "TyreLife", "Compound", "LapNumber"])
df["LapTime"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
df["Compound"] = df["Compound"].str.upper()

X = df[["TyreLife", "Compound", "LapNumber"]]
y = df["LapTime"].values

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
compound_encoded = encoder.fit_transform(X[["Compound"]])
compound_df = pd.DataFrame(compound_encoded, columns=encoder.get_feature_names_out(["Compound"]))

X_final = pd.concat([
    X.drop(columns=["Compound"]).reset_index(drop=True),
    compound_df.reset_index(drop=True)
], axis=1)

joblib.dump(encoder, "models/compound_encoder.pkl")

dtrain = xgb.DMatrix(X_final, label=y)

params = {
    "tree_method": "hist",
    "device": "cuda",
    "objective": "reg:squarederror"
}
model = xgb.train(params, dtrain, num_boost_round=100)

model.save_model("models/lap_time_gpu_model.json")
print("Model trained and saved with GPU support.")
