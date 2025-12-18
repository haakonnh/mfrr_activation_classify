from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Import project modules regardless of CWD
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.append(str(REPO_ROOT))

from src.data.preprocess import Config, build_dataset  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Analyze day-ahead vs regulation price spread for a naive persistence classifier "
            "that predicts RegClass+4 == RegClass (NO1/NO2)."
        )
    )
    p.add_argument("--area", type=str, default="NO1", choices=["NO1", "NO2"], help="Market area")
    p.add_argument(
        "--data_dir",
        type=str,
        default=str(REPO_ROOT / "data" / "raw"),
        help="Raw data directory",
    )
    p.add_argument("--include_2024", action="store_true", help="Include 2024 data where available")
    p.add_argument("--no-include_2024", dest="include_2024", action="store_false")
    p.set_defaults(include_2024=True)
    p.add_argument("--dropna", action="store_true", help="Drop rows with missing features in preprocessing")
    p.set_defaults(dropna=True)
    p.add_argument("--min_up_volume", type=float, default=None, help="Threshold to treat tiny Up activations as none")
    p.add_argument("--min_down_volume", type=float, default=None, help="Threshold to treat tiny Down activations as none")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "reports" / "dataframes"),
        help="Directory to write CSV outputs",
    )
    p.add_argument(
        "--fig-dir",
        type=str,
        default=str(REPO_ROOT / "reports" / "figures" / "multiclass"),
        help="Directory to write figures (optional)",
    )
    p.add_argument("--save-samples", action="store_true", help="Save per-row sample CSV of correct predictions with spread")
    return p


def compute_spread_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: 'RegClass', 'RegClass+4', 'PriceUp - DA', 'PriceDown - DA'
    if "RegClass" not in df.columns or "RegClass+4" not in df.columns:
        raise ValueError("Expected 'RegClass' and 'RegClass+4' columns in dataframe")
    for c in ("PriceUp - DA", "PriceDown - DA"):
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' in dataframe")

    # Naive persistence prediction: y_hat = RegClass
    y_true = df["RegClass+4"].astype("object")
    y_hat = df["RegClass"].astype("object")
    correct = y_true == y_hat

    # Signed spread definition consistent by class
    # up:  PriceUp - DA (typ. >=0)
    # down: DA - PriceDown = -(PriceDown - DA) (typ. >=0)
    up_spread = df["PriceUp - DA"]
    down_spread = -df["PriceDown - DA"]
    spread = np.where(y_hat == "up", up_spread, np.where(y_hat == "down", down_spread, np.nan))

    out = df.copy()
    out["y_true"] = y_true
    out["y_hat"] = y_hat
    out["correct"] = correct
    out["spread"] = spread
    return out


def summarize_spread(df_with_spread: pd.DataFrame) -> pd.DataFrame:
    sub = df_with_spread[(df_with_spread["correct"]) & (df_with_spread["y_hat"].isin(["up", "down"]))].copy()
    if sub.empty:
        return pd.DataFrame(columns=["class", "count", "mean", "median", "p10", "p90"])
    def _summ(g: pd.DataFrame) -> pd.Series:
        s = g["spread"].dropna().astype(float)
        if s.empty:
            return pd.Series({"count": 0, "mean": np.nan, "median": np.nan, "p10": np.nan, "p90": np.nan})
        return pd.Series({
            "count": int(s.shape[0]),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p10": float(s.quantile(0.10)),
            "p90": float(s.quantile(0.90)),
        })
    by_class = sub.groupby("y_hat", dropna=False).apply(_summ).reset_index().rename(columns={"y_hat": "class"})
    overall = _summ(sub)
    overall["class"] = "overall"
    overall = overall.to_frame().T
    return pd.concat([by_class, overall], ignore_index=True)


def main() -> int:
    args = build_parser().parse_args()

    # Preprocess and build dataset
    cfg = Config(
        data_dir=args.data_dir,
        area=args.area,
        include_2024=args.include_2024,
        heavy_interactions=False,
        dropna=args.dropna,
        min_up_volume=args.min_up_volume,
        min_down_volume=args.min_down_volume,
    )
    # Use label RegClass+4; build_dataset returns df with RegClass and RegClass+4
    df, _splits, _features = build_dataset(cfg, label_name="RegClass+4")

    # Compute spreads
    dfw = compute_spread_rows(df)
    summary = summarize_spread(dfw)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sum_csv = out_dir / "persistence_spread_correct_summary.csv"
    summary.to_csv(sum_csv, index=False)
    print("Summary written:", sum_csv)
    print(summary)

    if args.save_samples:
        samples = dfw[(dfw["correct"]) & (dfw["y_hat"].isin(["up", "down"]))][["y_true", "y_hat", "spread"]].copy()
        samp_csv = out_dir / "persistence_spread_correct_samples.csv"
        samples.to_csv(samp_csv, index=False)
        print("Samples written:", samp_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
