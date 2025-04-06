
# F1 Strategy Predictor (GPU-Accelerated)

This project predicts and compares Formula 1 race strategies using historical data, machine learning, and Monte Carlo simulation. It uses **GPU-accelerated XGBoost** for fast training and inference. The core functionality includes:

- Collecting race data via FastF1
- Training a GPU-accelerated model to predict lap times
- Simulating races with tyre strategies
- Running Monte Carlo simulations to find the optimal strategy

---

## Project Structure

```
f1-strategy-predictor/
├── data/
│   └── collect_data.py
├── models/
│   ├── train_lap_time_model.py
├── simulation/
│   ├── race_simulator.py
│   └── monte_carlo.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Components

### 1. **Data Collection (`data/collect_data.py`)**
Fetches and stores lap-level data (tyre, compound, stint) using FastF1. Data is saved as CSV for training.

### 2. **Model Training (`models/train_lap_time_model.py`)**
Trains an **XGBoost regressor on GPU** using lap number, tyre life, and compound (one-hot encoded). 

### 3. **Race Simulation (`simulation/race_simulator.py`)**
Simulates a full race based on a given tyre strategy using batched GPU-based predictions for fast lap time estimation.

### 4. **Monte Carlo Simulation (`simulation/monte_carlo.py`)**
Simulates 100s of races per strategy using `simulate_race()`. Reports average time + standard deviation to rank strategies.

### 5. **Main Entry (`main.py`)**
Runs the full pipeline: evaluates multiple strategies and identifies the best one using Monte Carlo stats.

---

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies:
- pandas
- numpy
- scikit-learn
- xgboost >= 1.7 (GPU support)
- joblib
- fastf1 (for data collection)

---

## Usage

### Step 1: Collect race data
```bash
python data/collect_data.py
```

### Step 2: Train the GPU model
```bash
python models/train_lap_time_model.py
```

### Step 3: Run strategy simulation + Monte Carlo
```bash
python main.py
```

---

## Example Output

```bash
Running Monte Carlo Strategy Comparison (GPU)...

Strategy: [('SOFT', 12), ('MEDIUM', 20), ('HARD', 20)]
Avg Time: 3452.85s ± 15.32

Strategy: [('MEDIUM', 26), ('HARD', 26)]
Avg Time: 3435.92s ± 12.43

Best Strategy:
[('MEDIUM', 26), ('HARD', 26)] with avg time 3435.92 seconds
```

---

## Notes

- GPU prediction is batched for speed via `xgboost.DMatrix`
- Model expects compound names in uppercase: `SOFT`, `MEDIUM`, `HARD`
- You can adjust number of Monte Carlo runs in `main.py`
- This setup is optimized for **Silverstone (52 laps)** by default

---

## 🏁 Future Ideas

- Add Streamlit or CLI interface
- Add rain, safety cars, and fuel weight simulation
- Visualize lap-by-lap degradation curves
