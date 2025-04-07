# F1 Strategy Predictor (GPU-Accelerated + Track-Aware)

This project predicts and compares Formula 1 race strategies using historical data, machine learning, and Monte Carlo simulation. It uses **GPU-accelerated XGBoost** and integrates official F1 rules like tyre usage limits and parc fermé constraints.

### 🔥 Features:
- Track-specific data + model training
- GPU-based lap time prediction
- Legal strategy generation (with used tyres + degradation caps)
- Monte Carlo simulation engine
- Streamlit UI for interactive analysis

---

## 🗂️ Project Structure

```
f1-strategy-predictor/
├── data/
│   └── collect_data.py            # Collects race data from FastF1
├── models/
│   └── train_lap_time_model.py    # Trains GPU-accelerated XGBoost model
├── simulation/
│   ├── race_simulator.py          # Full-lap simulation with degradation + failures
│   ├── monte_carlo.py             # Monte Carlo wrapper for bulk simulation
│   └── generate_strategies.py     # Builds legal strategy permutations
├── ui/
│   └── app.py                     # Streamlit GUI to run end-to-end system
├── main.py
├── requirements.txt
└── README.md
```

---

## 🧠 Core Components

### 1. 📥 Data Collection (`data/collect_data.py`)
Collects lap-by-lap data for any track in 2023–2024 using FastF1. Saves as CSV based on track name.

### 2. 🧠 Model Training (`models/train_lap_time_model.py`)
Trains an XGBoost model using:
- Tyre life
- Lap number
- Compound (one-hot)

Per-track models are saved and evaluated with MSE + R² score.

### 3. 🧩 Strategy Generator (`simulation/generate_strategies.py`)
Generates all **rule-legal** tyre strategies based on:
- Available compounds
- Remaining laps per tyre (used vs. new)
- Max usable life before heavy degradation

Strategies respect F1 rules like:
- Minimum 1 pit stop
- At least 2 dry compounds
- Parc fermé tyre usage

### 4. 🏎️ Race Simulator (`simulation/race_simulator.py`)
Simulates each lap using:
- ML-predicted lap time
- Degradation curve
- Tyre burst chance past thresholds

### 5. 🎲 Monte Carlo (`simulation/monte_carlo.py`)
Simulates 100s of races per strategy to estimate:
- Avg. race time
- Std deviation
- DNF risks

### 6. 🖥️ Web App (`ui/app.py`)
Streamlit UI lets you:
- Select a track
- Input used tyres
- Run full pipeline
- Visualize the top 3 strategies

---

## ✅ Setup

Install Python dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1: Launch the Streamlit UI
```bash
streamlit run ui/app.py
```

### Inside the app:
1. Pick a track from the dropdown
2. Paste your tyre inventory (used laps + compounds)
3. Click "Run Full Strategy Prediction"

---

## 📊 Example Output (Console)

```
✅ Generated 12 legal strategies
Best Strategy:
[('MEDIUM', 26, 0), ('HARD', 26, 0)] → 3435.92s ± 12.43
```

---

## 📝 Notes

- Simulator penalizes tyre overuse with degradation and DNF risk
- All models are track-specific (`models/lap_time_model_<track>.json`)
- `generate_strategies.py` ensures realistic strategies only
- Monte Carlo runs can be customized in the app

---

## 🧪 Future Ideas

- Rain + safety car simulation
- Driver-specific performance modifiers
- Strategy comparison for qualifying setups
- Auto-import race weekends from FastF1