# mFRR Activation Direction Forecasting  
**Participant-Feasible Machine Learning Pipeline for Nordic Balancing Markets**

## Overview
This repository implements a full machine learning pipeline for forecasting **mFRR (manual Frequency Restoration Reserve) energy activation direction** in the Nordic balancing market at **15-minute resolution**, using only information available to market participants at gate closure.

The project is motivated by the increasing role of flexible demand-side aggregators (e.g., electric vehicles, heat pumps, batteries) in balancing markets, where **activation direction (UP / DOWN / NONE)** is often a more critical short-term decision variable than exact prices or volumes.

The system is designed to be:
- **Market-realistic** (no TSO-only or future-leaking features)
- **Temporally consistent** (chronological training and evaluation)
- **Research-grade** (feature diagnostics, reproducibility, and extensibility)

This codebase serves as a foundation for extending deterministic classification into **probabilistic belief modeling and reinforcement learning-based bidding strategies**.

---

## Problem Formulation
At each 15-minute interval `t`, the model predicts the mFRR activation state at `t + 4`:
d(t+4) ∈ {UP, DOWN, NONE}


The task is framed as a **time-dependent multiclass classification problem** under strict information constraints that reflect what a real market participant can observe at decision time.

---

## System Architecture
```text
Raw Market Data
   |
   v
Preprocessing & Alignment
   |
   v
Feature Engineering
   |
   v
Temporal Train / Validation / Test Split
   |
   v
ML Model Training (AutoGluon / CatBoost)
   |
   v
Evaluation & Diagnostics
   |
   v
Model Artifacts / Research Outputs
```

---

## Data Sources

### Nord Pool
- mFRR activation data (15-minute resolution)  
- Day-ahead prices  
- Cross-zonal flows  
- Load and production (actuals and forecasts)  

### NUCS (Nordic Unavailability Collection System)
- aFRR procurement prices and volumes  
- mFRR capacity market data  

### ENTSO-E
- aFRR activation data (hourly, resampled to 15 minutes)  

All datasets are:
- Time-aligned to CET/CEST  
- Resampled to 15-minute resolution  
- Forward-filled or interpolated where appropriate  
- Merged into a single chronologically indexed dataset  

---

## Feature Engineering
Features are designed to reflect both **market signals** and **physical system stress indicators**.

### Categories
- **Persistence Features**
  - Run-length encoding of past activation states
  - Directional memory indicators

- **Price Features**
  - Day-ahead vs regulation spreads
  - aFRR and mFRR capacity market signals

- **Grid & Flow Features**
  - Cross-zonal flow utilization (flow / NTC)
  - Import/export stress indicators

- **Production & Load Features**
  - Wind forecast error
  - Load forecast deviations
  - Renewable penetration ratios

- **Interaction Features**
  - Price-flow interactions
  - Wind-flow stress coupling

---

## Model Training
Models are trained using **AutoGluon Tabular**, with CatBoost typically emerging as the strongest base learner.

Key characteristics:
- Chronological data splits (no random shuffling)  
- Class imbalance handling  
- Model ensembling  
- Feature importance extraction  

### Baseline
A persistence-based baseline is used for comparison:
d̂(t+4) = d(t−4)


This tests whether machine learning provides value beyond regime inertia.

---

## Evaluation
The evaluation framework includes:
- Row-normalized confusion matrices  
- Classification reports  
- Feature importance analysis  
- Feature correlation analysis  
- PCA visualization  
- Conditional price spread analysis  

The pipeline is structured to support extension toward:
- **Probabilistic calibration** (Brier score, reliability diagrams)  
- **Belief-state modeling** for reinforcement learning  

---
## Repository Structure

```text
mfrr_classify/
├── data/
│   ├── preprocessed/
│   │   └── preprocessed_df.csv
│   └── raw/
│       ├── afrr/
│       ├── balancing/
│       ├── capacity_market/
│       ├── flows/
│       ├── load/
│       ├── prices/
│       └── production/
│
├── src/
│   ├── evaluation/
│   |   ├── correlation.py
│   |   ├── evaluation.py
│   │   └── price_spread.py
|   |   
│   ├── train/
│   |   ├── hyperparameters.py
│   │   └── train.py
│   ├── utils/
│       ├── args.py
│       ├── datapreprocessing_utils.py
│       ├── evaluation_utils.py
│       ├── nucs_api.py
│       ├── plotting.py
│       └── utils.py
├── models/
├── notebooks/
├── scripts/
└── reports/

```



---

## Reproducibility
- All datasets are timestamp-indexed and versioned  
- Feature definitions are centralized and deterministic  
- Model artifacts are saved after training  
- Training configurations are parameterized  

---

## Research Extensions

### Probabilistic Forecasting
Replace class labels with calibrated belief distributions:
[P(UP), P(DOWN), P(NONE)]

### Reinforcement Learning
Use probabilistic forecasts as a **belief state** to optimize bidding and capacity allocation for flexible demand-side aggregators under uncertainty.

---

## Requirements
- Python 3.9+  
- AutoGluon  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

## Disclaimer
This project is intended for **research and educational use only**. It does not constitute financial, trading, or operational advice for participation in electricity markets.




