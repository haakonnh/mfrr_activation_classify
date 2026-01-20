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
<svg width="700" height="520" viewBox="0 0 700 520" xmlns="http://www.w3.org/2000/svg">
  <style>
    .box { fill: #f6f8fa; stroke: #24292f; stroke-width: 1.5; rx: 8; ry: 8; }
    .text { font-family: Arial, sans-serif; font-size: 14px; fill: #24292f; text-anchor: middle; dominant-baseline: middle; }
    .arrow { stroke: #24292f; stroke-width: 1.5; marker-end: url(#arrowhead); }
  </style>

  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#24292f"/>
    </marker>
  </defs>

  <!-- Boxes -->
  <rect class="box" x="200" y="20" width="300" height="50"/>
  <text class="text" x="350" y="45">Raw Market Data (Nord Pool / NUCS / ENTSO-E)</text>

  <rect class="box" x="200" y="90" width="300" height="50"/>
  <text class="text" x="350" y="115">Preprocessing & Alignment</text>

  <rect class="box" x="200" y="160" width="300" height="50"/>
  <text class="text" x="350" y="185">Feature Engineering</text>

  <rect class="box" x="200" y="230" width="300" height="50"/>
  <text class="text" x="350" y="255">Temporal Split (Train / Val / Test)</text>

  <rect class="box" x="200" y="300" width="300" height="50"/>
  <text class="text" x="350" y="325">ML Training (AutoGluon / CatBoost)</text>

  <rect class="box" x="200" y="370" width="300" height="50"/>
  <text class="text" x="350" y="395">Evaluation & Diagnostics</text>

  <rect class="box" x="200" y="440" width="300" height="50"/>
  <text class="text" x="350" y="465">Model Artifacts (Models / Metrics / Data)</text>

  <!-- Arrows -->
  <line class="arrow" x1="350" y1="70" x2="350" y2="90"/>
  <line class="arrow" x1="350" y1="140" x2="350" y2="160"/>
  <line class="arrow" x1="350" y1="210" x2="350" y2="230"/>
  <line class="arrow" x1="350" y1="280" x2="350" y2="300"/>
  <line class="arrow" x1="350" y1="350" x2="350" y2="370"/>
  <line class="arrow" x1="350" y1="420" x2="350" y2="440"/>
</svg>

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
src/
├── data/
│   ├── preprocess.py        # Resampling, alignment, merging, target creation
│   └── features.py         # Feature engineering logic
│
├── train/
│   ├── train.py            # Model training and orchestration
│   └── hyperparameters.py # Model and training configuration
│
├── evaluation/
│   ├── feature_analysis.py
│   ├── correlation.py
│   ├── pca.py
│   └── price_spread_analysis.py
│
├── artifacts/
│   ├── models/            # Trained models and ensembles
│   ├── metrics/          # Evaluation outputs
│   └── datasets/         # Preprocessed datasets
│
└── README.md


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




