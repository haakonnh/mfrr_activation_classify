# mfrr_activation_classify


A practical workspace for exploring reserve-market data and building up/down/none classifiers for activation direction. The focus is aFRR/NUCS hourly data and lightweight, reproducible pipelines for training, analysis, and plotting.

This repo is organized around the `mfrr_classify` package, with scripts, notebooks, and VS Code tasks to make the common workflows one-command friendly.


## Quick Start

- Python: 3.12 (recommended)
- OS: Windows (commands below use `cmd.exe`)

1) Activate the project environment (if it already exists):
```
.venv1\Scripts\activate.bat
```

2) Or create a fresh venv and install deps (if needed):
```
python -m venv .venv1
.venv1\Scripts\activate.bat
pip install -U pip
pip install -r requirements.txt
```

3) Open in VS Code and use the built-in tasks (Terminal → Run Task…) for data fetch / training / maintenance.


## Data Layout

- Raw: `mfrr_classify/data/raw/`
  - aFRR hourly file used in plots: `mfrr_classify/data/raw/afrr/nucs_hourly_2024_2025_updown.csv`
- Preprocessed cache: `mfrr_classify/data/preprocessed/`
- Reports (figures, tables): `mfrr_classify/reports/`
  - Plots: `mfrr_classify/reports/plots/`
  - DataFrames: `mfrr_classify/reports/dataframes/`

Use the NUCS fetch script to populate/update hourly data.


## Fetching NUCS Hourly Data

Requires environment variables for the API:
- `NUCS_BASE_URL` – base URL for the NUCS endpoint
- `NUCS_TOKEN` – auth token

Run via VS Code task:
- Task: “Fetch NUCS hourly (chunk=50d)”
  - Writes CSV to `mfrr_classify/reports/dataframes/nucs_hourly.csv`
  - Also drops quick-look plots into `mfrr_classify/reports/plots/`

Run via command line (equivalent):
```
set NUCS_BASE_URL=https://your-api
set NUCS_TOKEN=your-token
.venv1\Scripts\python.exe mfrr_classify\scripts\nucs_fetch.py \
  --base-url %NUCS_BASE_URL% --token %NUCS_TOKEN% \
  --start 202501010000 --end 202512162200 --chunk-days 50 \
  --out-csv mfrr_classify\reports\dataframes\nucs_hourly.csv \
  --plots-dir mfrr_classify\reports\plots
```


## Training Models

Most training happens through `mfrr_classify/src/models/train.py`, which wraps AutoGluon presets plus a few local conveniences.

- Typical tasks: `multiclass` or `multiclass_stack`
- Choose a preset, time budget, and optional ensembling/bagging/stacking

VS Code tasks:
- “Train (multiclass): time_limit=<seconds>”
- “Train (multiclass_stack): time_limit=<seconds>”

Example (command line):
```
.venv1\Scripts\python.exe mfrr_classify\src\train\train.py \
  --task multiclass \
  --time_limit 600 \
  --model_preset rf_xt_priority \
  --tune_up_bias \
  --single_persistence
```

Outputs:
- Trained predictors under `mfrr_classify/models/<run_name>/`
- Aggregated results: `mfrr_classify/classification_results.csv` (when produced)
- Time-sweep experiments: `mfrr_classify/models/time_curve_runs/` + logs under `time_curve_logs/`


## Notebooks

Use `mfrr_classify/notebooks/10-training-launchpad.ipynb` as the sandbox. It has:
- Environment bootstrap and project imports
- Quick baseline and CatBoost/XGB experiments
- Diagnostics: confusion matrices, transition tables, price-spread vs DA
- Plotting snippets for NUCS hourly price/volume exploration

Additional notebooks (01–07) cover distribution checks, time analysis, and ad‑hoc experiments.

Tip: The launchpad notebook sets `ROOT` to the package folder automatically. Run cells top-to-bottom once to initialize.


## Plotting: aFRR Price & Volume Over Time

Time‑series plots (raw hourly, no averaging) are written to:
- `mfrr_classify/reports/plots/afrr_updown_price_volume_timeseries_raw_2024_2025.png`

Distributions (hist) for the same fields:
- `mfrr_classify/reports/plots/afrr_updown_price_volume_distributions_2024_2025.png`

Bounds are set robustly so outliers don’t crush the axes (prices: p1–p99; volumes: p0.5–p99.5). If you change file locations or column names, adjust the plotting cell in the notebook accordingly.


## Built‑In VS Code Tasks

- “Fetch NUCS hourly (chunk=50d)”
  - Pulls hourly spans in chunks, writes CSV and plots.
- “Train (multiclass): time_limit=<s>”
  - Single‑stage tabular training with `rf_xt_priority` by default.
- “Train (multiclass_stack): time_limit=<s>”
  - Stack recipe (rf/xgb/etc.) with 3‑fold bagging and one stack level.
- “Time curve sweep (budgets=…)”
  - Runs multiple budgets to get a train‑time vs quality curve.
- “Wipe logs & runs (dry‑run, keep results, keep latest=1)”
  - Preview cleanup of heavy log/output folders.
- “Wipe logs & runs (confirm, destructive)”
  - Irreversibly prunes logs and old runs. Use with care.

All tasks assume the Python at `.venv1\Scripts\python.exe`.


## Repo Map (short)

- `mfrr_classify/`
  - `data/` – raw, preprocessed, and scratch
  - `notebooks/` – analysis & training notebooks (launchpad lives here)
  - `scripts/` – CLI utilities (fetch, time‑curve, wipe)
  - `src/` – training and data pipeline code
  - `models/` – saved predictors and experiment runs
  - `reports/` – plots and CSV outputs
- `blackjack/`, `tictactoe/`, `windygridworld/` – unrelated sandbox examples
- `LaTeX/` – paper/thesis materials


## Housekeeping

- Dry‑run cleanup:
```
.venv1\Scripts\python.exe mfrr_classify\scripts\wipe_logs.py --dry-run --keep-results --keep-latest 1
```
- Confirmed cleanup (destructive):
```
.venv1\Scripts\python.exe mfrr_classify\scripts\wipe_logs.py --yes
```


## Troubleshooting

- “File not found”: confirm the expected CSV under `mfrr_classify/data/raw/afrr/` exists or run the fetch task.
- “Long training runs”: start with smaller `--time_limit` (e.g., 300–600) and a simpler preset.
- “Import errors in notebook”: run the first environment cell to set `ROOT` and `sys.path`.
- “Plots look flat”: adjust quantiles for y‑limits in the plotting cell (prices p1–p99, volumes p0.5–p99.5 are good defaults).


## Notes

- The code leans on AutoGluon Tabular for fast, strong baselines; CatBoost and XGBoost presets are available for focused experiments.
- Keep results under `mfrr_classify/models/` and figures under `mfrr_classify/reports/` so experiments stay comparable.
