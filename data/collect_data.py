import fastf1

def load_race_data(year, gp, session_type="R"):
    session = fastf1.get_session(year, gp, session_type)
    session.load()

    laps = session.laps
    laps = laps.loc[laps['PitOutTime'].notnull() | laps['PitInTime'].notnull()]

    tyre_data = laps[['Driver', 'LapNumber', 'Compound', 'FreshTyre', 'TyreLife', 'LapTime', 'TrackStatus', 'IsPersonalBest']]

    return tyre_data, session

if __name__ == "__main__":
    tyre_data, session = load_race_data(2023, "Silverstone")
    tyre_data.to_csv("data/tyre_data_silverstone_2023.csv", index=False)
    print("Tyre data saved.")
