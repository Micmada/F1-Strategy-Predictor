import joblib
import numpy as np
import pandas as pd

lap_model = joblib.load("models/lap_time_model_v2.pkl")

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
        for lap_in_stint in range(1, stint_len + 1):
            lap_num = lap_counter + 1
            lap_time = lap_model.predict(pd.DataFrame([{
                "TyreLife": lap_in_stint,
                "Compound": compound,
                "LapNumber": lap_num
            }]))[0]
            noise = np.random.normal(0, 0.4)
            total_time += lap_time + noise
            lap_counter += 1
        total_time += pit_penalty

    return total_time, lap_counter
