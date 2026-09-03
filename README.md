# Operating-Regime-Aware Anomaly Detection for Gas-Turbine Time Series

Anomaly detection for gas turbine sensor streams that models normal behavior **relative to the operating
regime** (load, speed, control state) rather than globally, and calibrates alarm thresholds **per regime**.
A temporal convolutional network (TCN) forecasts next-step sensor state from healthy data only. Anomalies
are flagged from regime-normalized deviations between predicted and observed behavior, not from a single
global threshold. Evaluated against regime-blind baselines on the Naval/CBM and C-MAPSS datasets.

This is the regime-aware treatment of normality modeling
and score calibration, and its rigorous evaluation (5-seed sweeps, paired significance tests) against
baselines that ignore regime.

## Method

### Figure 1 — Model architecture

![Model architecture](assets/model_architecture.png)

<!-- A dilated **Temporal Block** stack encodes the input window `x ∈ ℝ^{L×D}`. The known regime id `r`
(precomputed by K-means clustering on healthy training data) conditions the shared temporal features
through **Regime FiLM** — a per-regime learned affine `h̃ = γ_r ⊙ y + β_r` — before both the forecast
head (`x̂_{t+1} ∈ ℝ^D`) and the reconstruction head (`x̂ ∈ ℝ^{L×D}`). A separate auxiliary regime
classification head is computed from the *unconditioned* pooled features as a regularizer on the encoder.
Training minimizes `L(θ) = w_fc·l_fc + w_rec·l_rec + w_regime·l_regime` on healthy-labeled windows only. -->

### Figure 2 — Regime-specific calibration and inference

![Calibration and inference](assets/model_inference.png)

<!-- The forecast residual `e`, reconstruction error `c = l_rec`, and pooled latent feature `φ'` each pass
through an optional per-regime **Mahalanobis distance**, then per-regime **median/MAD normalization**
(fit once on healthy training scores). The normalized terms combine into a hybrid anomaly score
`S = z_resid + α·z_latent + λ·z_recon`, compared against a threshold `τ` tuned on healthy data. -->

## Datasets

| Dataset | Type | Regimes | Notes |
|---|---|---|---|
| **UCI CBM / Naval** (`UCI CBM Dataset/data.txt`) | Steady-state design-of-experiments table (16 GT sensors + 2 decay coefficients) | 9 (ship speed, 3–27 kn in 3 kn steps) | No temporal order — each row is an independent length-1 pseudo-window; event-level metrics do not apply. Healthy = both decay coefficients above threshold. |
| **C-MAPSS** (`CMAPSSData/{train,test,RUL}_FDxxx.txt`) | Run-to-failure engine trajectories | FD001/FD003: 1 · FD002/FD004: 6 | Labels from RUL thresholds (normal if `RUL > normal_rul`, anomaly if `RUL ≤ anomaly_rul`, else ignored). FD001/FD003 are single-regime **negative controls**. |

Regime counts are fixed empirically from K-means silhouette/inertia analysis of each dataset's physical
structure — they are not tuned hyperparameters.

## Results (5-seed sweep, mean ± std)

Point-wise **F1**:

| Dataset | Regime-Aware TCN | Regime-Aware AE (ablation) | Global TCN Forecaster | TCN Autoencoder | Isolation Forest |
|---|---|---|---|---|---|
| CBM   | **0.912 ± 0.015** | 0.906 ± 0.015 | 0.827 ± 0.021 | 0.800 ± 0.057 | 0.799 ± 0.010 |
| FD001 | **0.983 ± 0.010** | 0.721 ± 0.055 | 0.528 ± 0.069 | 0.752 ± 0.014 | 0.953 ± 0.015 |
| FD002 | **0.909 ± 0.005** | 0.692 ± 0.016 | 0.757 ± 0.002 | 0.839 ± 0.057 | 0.801 ± 0.027 |
| FD003 | **0.970 ± 0.017** | 0.899 ± 0.025 | 0.396 ± 0.043 | 0.859 ± 0.045 | 0.825 ± 0.051 |
| FD004 | **0.896 ± 0.013** | 0.573 ± 0.045 | 0.198 ± 0.001 | 0.406 ± 0.040 | 0.511 ± 0.068 |

Point-wise **ROC-AUC**:

| Dataset | Regime-Aware TCN | Regime-Aware AE (ablation) | Global TCN Forecaster | TCN Autoencoder | Isolation Forest |
|---|---|---|---|---|---|
| CBM   | 0.974 ± 0.007 | 0.968 ± 0.008 | 0.905 ± 0.017 | 0.867 ± 0.073 | 0.846 ± 0.008 |
| FD001 | 1.000 ± 0.000 | 0.966 ± 0.013 | 0.892 ± 0.044 | 0.972 ± 0.003 | 0.999 ± 0.000 |
| FD002 | 0.970 ± 0.002 | 0.913 ± 0.009 | 0.493 ± 0.008 | 0.937 ± 0.055 | 0.894 ± 0.008 |
| FD003 | 1.000 ± 0.000 | 0.999 ± 0.000 | 0.977 ± 0.004 | 0.999 ± 0.001 | 1.000 ± 0.000 |
| FD004 | 0.994 ± 0.001 | 0.955 ± 0.014 | 0.510 ± 0.009 | 0.902 ± 0.013 | 0.947 ± 0.011 |

Regime-Aware TCN is the only method with tight, low-variance F1 across every dataset (std ≤ 0.015).
Its advantage is statistically significant (paired *t*-test, p < 0.05) over every baseline on every
dataset except CBM vs. the regime-aware autoencoder ablation — expected, since CBM's length-1
pseudo-windows make forecasting and reconstruction nearly the same task.

## Repository structure

```
├── main.py                  # Train + evaluate the regime-aware TCN
├── inference.py             # Run scoring from a saved checkpoint
├── tune.py                  # Optuna hyperparameter search
├── compare.py               # Train the regime-aware model + all baselines on identical splits, print one table
│
├── data_pipeline.py         # Load + label each dataset, fit regime clustering on healthy data, build DataLoaders
├── model.py                 # RegimeAwareTCN (TemporalBlock stack, AttentionPooling, RegimeFiLM) + regime-blind baseline nets
├── train.py                 # Training core (healthy-only loop, early stopping), calibration, per-regime evaluation
├── baselines.py             # Regime-blind comparisons: PlainTCNForecaster, ConvAutoencoder, IsolationForest
├── event_metrics.py         # Event-level metrics (recall, detection delay, false-alarm rate) — C-MAPSS only
├── utils.py                 # Plotting + reproducibility (seed_everything, figure helpers)
│
├── configs/                 # Per-dataset YAML configs (base, cbm, fd001–fd004)
├── CMAPSSData/              # C-MAPSS raw text files (train/test/RUL per subset)
├── UCI CBM Dataset/         # Naval propulsion condition-based-maintenance dataset
│
├── test_data_pipeline_regime_clustering.py   # Unit tests (python -m unittest)
└── data.ipynb              # Legacy exploratory notebook (not used by the pipeline)
```

## Setup

Python 3.12. No lockfile — install into your environment:

```bash
pip install torch scikit-learn numpy pyyaml matplotlib seaborn optuna
```

## Usage

```bash
# Train + evaluate the regime-aware TCN (checkpoints saved to ./model/)
python main.py --dataset CBM --config ./configs/cbm.yaml
python main.py --dataset CMAPSS --fd FD002 --config ./configs/fd002.yaml

# Run inference from a saved checkpoint
python inference.py --dataset CMAPSS --fd FD002 \
    --checkpoint ./model/best_cmapss_fd002_regime_tcn.ckpt --config ./configs/fd002.yaml

# Hyperparameter search (Optuna) → ./optuna_runs/<study_name>/
python tune.py --dataset CMAPSS --fd FD002 --config ./configs/fd002.yaml --n_trials 50

# Train the regime-aware model AND all baselines on identical splits, print one comparison table
python compare.py --dataset CMAPSS --fd FD002 --config ./configs/fd002.yaml
```

`--fd` (`FD001`–`FD004`) selects the C-MAPSS subset; it is ignored for `--dataset CBM`.
`--seed` (default 42) controls model-init/training stochasticity; the train/val/test split is fixed
independently so multi-seed runs measure training variance only.

## Authors

Hoda Elmi, Lincoln AI Lab, University of Lincoln.