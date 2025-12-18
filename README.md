# mFRR Activation Forecasting in NO1 — Specialization Project

This repository contains code and configuration to build and evaluate classifiers for mFRR activation direction (up / down / none) in the Norwegian NO1 price area. It is part of a specialization project, with an emphasis on reproducible baselines and clear experiment outputs.


## Contents
- Project code and data pipeline under `mfrr_classify/`
- Training entry points (AutoGluon Tabular–based)
- Reporting utilities (feature importance, confusion matrices)
- Optional VS Code tasks for common workflows


## Environment Setup (Windows, cmd)

Create a virtual environment and install dependencies from `requirements.txt`:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Upgrade pip if it fails.

## Data

- Raw market data (NO1 focus) is under `mfrr_classify\data\raw\`
- Preprocessed caches are written to `mfrr_classify\data\preprocessed\`
- Reports (figures and CSVs) are written to `mfrr_classify\reports\`

If NUCS access is available, hourly aFRR up/down prices and volumes can be fetched as follows:

```
set NUCS_BASE_URL=https://your-api
set NUCS_TOKEN=your-token
.venv\Scripts\python.exe mfrr_classify\scripts\nucs_fetch.py --base-url %NUCS_BASE_URL% --token %NUCS_TOKEN% --start 202501010000 --end 202512162200 --chunk-days 50 --out-csv mfrr_classify\reports\dataframes\nucs_hourly.csv --plots-dir mfrr_classify\reports\plots
```


## Training

Run a short multiclass experiment (NO1, 1‑minute time limit, random forest/extra trees priority) to validate the full path from preprocessing to evaluation:

```
.venv\Scripts\python.exe mfrr_classify\src\train\train.py --task multiclass --time_limit 60 --model_preset rf_xt_priority --tune_up_bias
```

Outputs include:
- Model directory under `mfrr_classify\models\multiclass_ag\`
- Figures under `mfrr_classify\reports\figures\multiclass\`
- CSV summaries under `mfrr_classify\reports\dataframes\`

To examine performance as a function of time budget, use the time‑curve script:

```
.venv\Scripts\python.exe mfrr_classify\scripts\run_time_curve.py --budgets 60,120,200 --tune-up-bias --single-persistence --python .venv\Scripts\python.exe
```


## Notebooks

The notebook `mfrr_classify\notebooks\10-training-launchpad.ipynb` provides an environment bootstrap, exploratory plots (including hourly aFRR time‑series with robust y‑limits), and diagnostic summaries. Execute once top‑to‑bottom to initialize paths and imports.


## Maintenance Utilities

Preview cleanup (no deletions):
```
.venv\Scripts\python.exe mfrr_classify\scripts\wipe_logs.py --dry-run --keep-results --keep-latest 1
```

Perform cleanup (destructive):
```
.venv\Scripts\python.exe mfrr_classify\scripts\wipe_logs.py --yes
```


## Reproducibility Notes
- Experiments and figures are kept under `mfrr_classify\models\` and `mfrr_classify\reports\` to keep runs comparable.
- AutoGluon presets provide time‑bounded tabular training suitable for iterative research.
