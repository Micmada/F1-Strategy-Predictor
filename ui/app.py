import streamlit as st
from simulation.monte_carlo import run_monte_carlo

# Predefined strategies
strategies = {
    "1-stop (MED-HARD)": [("MEDIUM", 26), ("HARD", 26)],
    "2-stop (SOFT-MED-HARD)": [("SOFT", 12), ("MEDIUM", 20), ("HARD", 20)],
    "Aggressive (SOFT x2 + MED)": [("SOFT", 15), ("SOFT", 15), ("MEDIUM", 22)],
    "No-stop (HARD)": [("HARD", 52)],
    "Balanced 3-stop": [("MEDIUM", 17), ("MEDIUM", 17), ("HARD", 18)]
}

st.title("F1 Strategy Predictor")

st.write("Select tyre strategies to simulate and compare:")

selected_keys = st.multiselect("Choose strategies:", list(strategies.keys()), default=list(strategies.keys())[:3])

runs = st.slider("Monte Carlo runs per strategy", 100, 1000, 300, step=100)

if st.button("🚦 Run Simulation"):
    sim_strategies = [strategies[key] for key in selected_keys]
    st.write("Running simulations...")

    results = run_monte_carlo(sim_strategies, runs=runs)
    results.sort(key=lambda x: x["average_time"])

    best = results[0]

    st.success(f"Best Strategy: {best['strategy']} with average time: {best['average_time']:.2f}s")

    st.subheader("Strategy Comparison")
    for r in results:
        st.markdown(f"**{r['strategy']}** → {r['average_time']:.2f}s ± {r['std_dev']:.2f}")

    import matplotlib.pyplot as plt

    labels = [str(r['strategy']) for r in results]
    times = [r['average_time'] for r in results]

    fig, ax = plt.subplots()
    ax.barh(labels, times)
    ax.set_xlabel("Average Race Time (s)")
    ax.set_title("Strategy Performance Comparison")
    st.pyplot(fig)
