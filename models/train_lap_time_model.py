import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("data/tyre_data_silverstone_2023.csv")

df = df.dropna(subset=["LapTime", "TyreLife", "Compound", "LapNumber"])
df["LapTime"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()

df["Compound"] = df["Compound"].str.upper()

X = df[["TyreLife", "Compound", "LapNumber"]]
y = df["LapTime"]

categorical_features = ["Compound"]
numeric_features = ["TyreLife", "LapNumber"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)
print("Model score:", pipeline.score(X_test, y_test))

joblib.dump(pipeline, "models/lap_time_model_v2.pkl")
