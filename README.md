# jump_difussion
# BTC Multi-Window Path Signature Regime Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A robust unsupervised financial machine learning pipeline designed to segment asset price actions into structured structural regimes (e.g., low-volatility drift, high-volatility mean-reversion, or aggressive momentum trends). 

The system leverages **Multi-Window Path Signatures (truncated to level 2)** to capture localized price-volume topology, combined with a **Viterbi-optimized Jump Model Cluster framework** featuring temporal minimum dwell constraints to fully eliminate forward-looking lookahead biases in production trading environments.

---

## Technical Features Matrix

* **Path Signature Embedding Engine:** Processes multi-horizon rolling slices ($W = [10, 50, 100]$ bars) across 4 orthogonal dimensions: Time, Log Returns, Absolute Volume, and Order Flow Imbalance (Volume Delta).
* **Viterbi Jump Model Optimization:** Minimizes a joint objective function mapping physical spatial distance against discrete temporal state transition penalties ($\lambda = 40.0$).
* **Minimum Dwell Constraints:** Implements a retroactive clustering correction pass ensuring clean structural state longevity ($\tau = 20$ hours minimum) to bypass high-frequency regime-flipping noise.
* **Hungarian Label Alignment Engine:** Employs the Jonker-Volgenant linear assignment mapping methodology to align cross-validation clusters against global baseline centroids via an exact $L^2$ distance cost matrix.
* **Causal Online Inference Engine:** Includes a completely historical, step-by-step state distribution projection engine (`predict_regime_from_new_data`) allowing clean, out-of-sample, live inference.

---

## Method & Mathematical Formulations

### 1. Multi-Window Path Signature Feature Extraction

For each window length $W \in \{10, 50, 100\}$, a 4-dimensional temporal path matrix $X_t$ is constructed for the discrete space:

$$X_t = \begin{bmatrix} t & \sum \text{log\_ret} & \sum \text{volume} & \sum \text{vol\_delta} \end{bmatrix}^T$$

Where $\text{vol\_delta} = 2 \cdot \text{taker\_buy\_volume} - \text{volume}$. To ensure geometric scale-invariance across periods of extreme market expansion, each slice is MinMax normalized independently to $[0, 1]^4$ before generating the signature.

The path signature is truncated to **Level 2** to calculate structural shape characteristics without parameter explosion:
* **Level 1 (Linear Displacement Delta):** $$\mathbf{S}_1^{(i)} = X_T^{(i)} - X_0^{(i)}$$
* **Level 2 (Iterated Inter-Path Cross Integrals):** Calculated explicitly via cross-product matrices over incremental coordinate shifts $dX_t$:
  $$\mathbf{S}_2^{(i, j)} = \int_{0}^{T} \left( X_t^{(i)} - X_0^{(i)} \right) dX_t^{(j)}$$

### 2. The Global Jump Model Objective Function

Rather than traditional hidden Markov structures requiring restrictive probabilistic distributions, this architecture groups sequential paths by optimization of the regularized loss function:

$$\min_{\{s_t\}_{t=1}^N, \{\mu_k\}_{k=1}^K} \sum_{t=1}^N \|x_t - \mu_{s_t}\|^2 + \lambda \sum_{t=2}^N \mathbb{I}(s_t \neq s_{t-1})$$

Where $x_t \in \mathbb{R}^d$ represents the scaled level-2 path signature payload, $\mu_k$ is the cluster spatial centroid for regime $k$, $s_t \in \{0, 1, 2\}$ is the assigned state label, and $\lambda$ acts as the strategic structural switch friction barrier (`JUMP_PENALTY = 40.0`).

### 2. The Global Jump Model Objective Function

Rather than traditional hidden Markov structures requiring restrictive probabilistic distributions, this architecture groups sequential paths by optimization of the regularized loss function:

$$\min_{\{s_t\}_{t=1}^N, \{\mu_k\}_{k=1}^K} \sum_{t=1}^N \|x_t - \mu_{s_t}\|^2 + \lambda \sum_{t=2}^N \mathbb{I}(s_t \neq s_{t-1})$$

Where $x_t \in \mathbb{R}^d$ represents the scaled level-2 path signature payload, $\mu_k$ is the cluster spatial centroid for regime $k$, $s_t \in \{0, 1, 2\}$ is the assigned state label, and $\lambda$ acts as the strategic structural switch friction barrier (`JUMP_PENALTY = 40.0`).

---

## Project Output Footprints

Execution transforms raw inputs and exports binary serializations along with performance validation plots:
---

## Model Evaluation & Performance Results

### ROC AUC Evaluation Performance Metrics

Metrics are generated via multiclass One-vs-Rest (OvR) Macro ROC AUC against the full-lookahead global baseline model. Labels are structurally synchronized across validation folds using Hungarian mapping centroid distance matching.

* **JumpModel_1 (Trained on first half, tested causally on second half) vs Baseline:** ROC AUC = 0.9045
* **JumpModel_2 (Trained on second half, tested causally on first half) vs Baseline:** ROC AUC = 0.8967

| Evaluated System Model Pipeline | Validation Testing Window Context | Macro ROC AUC Metric |
|---|---|---|
| **JumpModel_1 (Causal Inference)** | Test Evaluated across Second Half Portfolio | **0.9045** |
| **JumpModel_2 (Causal Inference)** | Test Evaluated across First Half Portfolio | **0.8967** |

---

## Regime Visualization Subsystems

### 1. Global Baseline Model (Full Dataset Timeline History)
![Baseline Jump Model - Full Dataset Look-Ahead](btc_jump_model_baseline.png)

### 2. Causal Jump Model 1 (Out-of-Sample Test Evaluation on Second Half)
![Causal Jump Model 1 - Test on Second Half](btc_jump_model_1_test_regimes.png)

### 3. Causal Jump Model 2 (Out-of-Sample Test Evaluation on First Half)
![Causal Jump Model 2 - Test on First Half](btc_jump_model_2_test_regimes.png)

---

## Production Quickstart & Live Inference Pipeline

To load your trained baseline weights and evaluate newly streamed incoming raw exchange bars without lookahead leakages:

```python
import pandas as pd
from src.inference import predict_regime_from_new_data

# Load live incoming streaming data paths
live_dataframe = pd.read_csv("data/live_stream_1m.csv") 

# Inject and predict current regime state causality 
computed_regimes_df = predict_regime_from_new_data(
    new_df=live_dataframe, 
    model_path="btc_jump_model_baseline.pkl"
)

current_market_regime = computed_regimes_df['regime_jm'].iloc[-1]
print(f"Current structural market execution state: Regime {current_market_regime}")
