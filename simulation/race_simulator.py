import xgboost as xgb
import joblib
import pandas as pd
import numpy as np

encoder = joblib.load("models/compound_encoder.pkl")
model = xgb.Booster()
model.load_model("models/lap_time_gpu_model.json")

TRACK_CONFIG = {
    "Silverstone": {
        "laps": 52,
        "pit_stop_penalty": 20
    }
}

def simulate_race(strategy, track="Silverstone"):
    total_time = 0
    lap_counter = 0
    laps_required = TRACK_CONFIG[track]["laps"]
    pit_penalty = TRACK_CONFIG[track]["pit_stop_penalty"]

    for compound, stint_len in strategy:
        compound = compound.upper()

        laps = lap_counter + np.arange(1, stint_len + 1)
        tyre_life = np.arange(1, stint_len + 1)
        compound_list = [compound] * stint_len

        base_df = pd.DataFrame({
            "TyreLife": tyre_life,
            "Compound": compound_list,
            "LapNumber": laps
        })

        compound_encoded = encoder.transform(base_df[["Compound"]])
        compound_df = pd.DataFrame(compound_encoded, columns=encoder.get_feature_names_out(["Compound"]))

        X_final = pd.concat([
            base_df.drop(columns=["Compound"]).reset_index(drop=True),
            compound_df.reset_index(drop=True)
        ], axis=1)

        dmatrix = xgb.DMatrix(X_final)

        lap_times = model.predict(dmatrix)
        noise = np.random.normal(0, 0.4, size=stint_len)

        total_time += np.sum(lap_times + noise)
        total_time += pit_penalty
        lap_counter += stint_len

    return total_time, lap_counter
