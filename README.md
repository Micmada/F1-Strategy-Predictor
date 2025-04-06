
# F1 Strategy Predictor

This project aims to predict and compare Formula 1 race strategies using historical data. The core functionality includes loading race data, training a machine learning model to predict lap times, simulating races with different strategies, and running Monte Carlo simulations to evaluate the performance of each strategy.

## Project Structure

```
f1-strategy-predicter/
├── data/
│   └── collect_data.py
├── models/
│   └── train_lap_time_model.py
├── simulation/
│   ├── monte_carlo.py
│   └── race_simulator.py
├── main.py
└── requirements.txt
```

## Components

### 1. **Data Collection (`data/collect_data.py`)**

This script is responsible for collecting race data from FastF1 and saving the relevant tyre information into a CSV file. The function `load_race_data()` fetches data for a specified year, Grand Prix (GP), and session type. The data includes information about the lap number, tyre compound, tyre life, and lap time.

### 2. **Model Training (`models/train_lap_time_model.py`)**

This script processes the collected data and trains a machine learning model using XGBoost to predict lap times based on tyre life, compound, and lap number. The trained model is saved as `lap_time_model_v2.pkl` and can be used for race simulations.

### 3. **Race Simulation (`simulation/race_simulator.py`)**

The race simulator utilizes the trained model to simulate race laps and predict lap times for different strategies. A strategy consists of different tyre compounds and stint lengths. The function `simulate_race()` calculates the total time for a given strategy on a specific track, accounting for pit stops and lap times.

### 4. **Monte Carlo Simulation (`simulation/monte_carlo.py`)**

This script runs Monte Carlo simulations for different race strategies. By simulating a race multiple times (default is 500), it computes the average race time and standard deviation for each strategy. This helps identify which strategy might provide the best performance.

### 5. **Main Script (`main.py`)**

The main script runs the Monte Carlo simulations with a predefined set of strategies, then sorts and prints the strategies based on average race time. It also prints the best-performing strategy.

## Requirements

The project requires the following Python libraries:

- pandas
- numpy
- scikit-learn
- xgboost
- joblib

You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Usage

### 1. **Collect Race Data**

To collect data for a specific Grand Prix and year, run:

```bash
python data/collect_data.py
```

This will save the data as `data/tyre_data_<GP>_<year>.csv`.

### 2. **Train the Model**

To train the lap time prediction model, run:

```bash
python models/train_lap_time_model.py
```

This will train the model and save it as `models/lap_time_model_v2.pkl`.

### 3. **Run Race Simulation**

To simulate a race with different strategies and track their performance, run:

```bash
python main.py
```

This will execute the Monte Carlo simulations, comparing the performance of various strategies and printing the results.

## Example Output

```bash
Running Strategy Comparison...
Strategy: [('SOFT', 12), ('MEDIUM', 20), ('HARD', 20)]
Avg Time: 3452.85s ± 15.32

Strategy: [('MEDIUM', 26), ('HARD', 26)]
Avg Time: 3435.92s ± 12.43

Best Strategy:
[('MEDIUM', 26), ('HARD', 26)] with avg time 3435.92 seconds
```

## Notes

- The simulation is currently configured for the Silverstone track, with 52 laps and a pit stop penalty of 20 seconds.
- The Monte Carlo simulation defaults to 500 runs but can be adjusted by passing a `runs` parameter to the `run_monte_carlo()` function.
