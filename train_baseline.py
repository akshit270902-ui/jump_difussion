import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from config import (
    N_REGIMES, JUMP_PENALTY, MIN_DWELL, N_INIT, MAX_ITER, TOL,
    RAW_DATA_PATH, RESULTS_DIR,
)
from utils import load_and_resample, save_model, regime_transition_stats
from features import compute_log_returns_and_flow, extract_features
from jump_model import fit_jump_model, enforce_min_dwell

def main():
    df = load_and_resample(RAW_DATA_PATH)
    df = compute_log_returns_and_flow(df)
    features, timestamps = extract_features(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    centers, labels = fit_jump_model(
        X, N_REGIMES, JUMP_PENALTY,
        min_dwell=MIN_DWELL,
        n_init=N_INIT, max_iter=MAX_ITER, tol=TOL,
        global_viterbi=True,
    )
    labels = enforce_min_dwell(labels, MIN_DWELL)
    stats = regime_transition_stats(labels)
    print("Regime stats:", stats)
    payload = {
        "centers": centers,
        "centers_unscaled": scaler.inverse_transform(centers),
        "scaler": scaler,
        "labels": labels,
        "timestamps": timestamps,
        "stats": stats,
    }
    save_model(payload, os.path.join(RESULTS_DIR, "btc_jump_model_baseline.pkl"))

    df_plot = df.iloc[max(10, 50, 100):].reset_index(drop=True)
    colors = ["#2ca02c", "#d62728", "#1f77b4"]
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(df_plot["close"].values, color="grey", alpha=0.35, lw=0.8)
    for r in range(N_REGIMES):
        mask = labels == r
        ax.scatter(np.where(mask)[0], df_plot["close"].values[mask],
                   s=6, color=colors[r], label=f"Regime {r}", zorder=2)
    ax.set_title(f"Oracle Baseline — Full Lookahead (λ={JUMP_PENALTY}, min_dwell={MIN_DWELL})")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "baseline_regimes.png"), dpi=200)
    print("Done.")

if __name__ == "__main__":
    main()
