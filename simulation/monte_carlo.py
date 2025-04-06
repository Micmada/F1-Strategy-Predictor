from simulation.race_simulator import simulate_race
import numpy as np

def run_monte_carlo(strategies, runs=500):
    results = []

    for strat in strategies:
        times = []
        for _ in range(runs):
            total_time, laps = simulate_race(strat)
            if laps == 52:
                times.append(total_time)
        avg_time = np.mean(times)
        std_time = np.std(times)
        results.append({
            "strategy": strat,
            "average_time": avg_time,
            "std_dev": std_time
        })

    return results
