"""
Ternary (multiclass) dataset preparation: up / down / none at t+4.

This module reuses the base feature preprocessing and adds a 3-class label
computed from BalanceMarket CSVs:
    - up   if NO1 Activated Up Volume (MW)   > 0 and down == 0 (or >= down)
    - down if NO1 Activated Down Volume (MW) > 0 and up   == 0 (or >  up)
    - none if both are 0
Then the training label is shifted -4 to align with t+4 (one hour ahead):
    label = 'RegClass+4'

Outputs:
    - attach_ternary_labels(...): adds RegClass and RegClass+4 to a DF
    - build_multiclass_dataset(...): returns df, (train_df, val_df, test_df), features, label

Usage (from repo root / upreg_classify working dir):
        from src.data.ternary import build_multiclass_dataset
        df, (train_df, val_df, test_df), features, label = build_multiclass_dataset()

Note: We import preprocess.py by extending sys.path so the module works
when run from various working directories.
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import numpy as np

# Ensure we can import preprocess.py from repo root (upreg_classify/)
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
        sys.path.append(_REPO_ROOT)


# Column names in BalanceMarket CSVs (confirmed by user)
UP_COL = 'NO1 Activated Up Volume (MW)'
DOWN_COL = 'NO1 Activated Down Volume (MW)'


def _read_csv(path: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, **kwargs)


def load_balance_market(data_dir: str, include_2024: bool) -> pd.DataFrame:
    """Load BalanceMarket CSVs for 2025 (+ optionally 2024),
    create a DatetimeIndex on 'Time', and return the concatenated frame.
    """
    bm_2025 = os.path.join(data_dir, 'balancing', 'BalanceMarket_2025_NO1_EUR_None_MW.csv')
    bm_2024 = os.path.join(data_dir, 'balancing', 'BalanceMarket_2024_NO1_EUR_None_MW.csv')
    mfrr_df = _read_csv(bm_2025, delimiter=';')
    if include_2024:
            mfrr_df_2024 = _read_csv(bm_2024, delimiter=';')
            mfrr_df = pd.concat([mfrr_df_2024, mfrr_df], ignore_index=True)
    mfrr_df.rename(columns={"Delivery Start (CET)": "Time"}, inplace=True)
    # Parse time and set index
    mfrr_df['Time'] = pd.to_datetime(mfrr_df['Time'], format='%d.%m.%Y %H:%M:%S')
    mfrr_df.set_index('Time', inplace=True)
    # Remove duplicates and sort
    print("Loaded BalanceMarket data with shape:", mfrr_df.shape)
    mfrr_df = mfrr_df[~mfrr_df.index.duplicated(keep='first')]
    mfrr_df.sort_index(inplace=True)
    return mfrr_df


def compute_regclass_series(
    mfrr_df: pd.DataFrame,
    up_col: str = UP_COL,
    down_col: str = DOWN_COL,
    tie_break: str = 'up',
    min_up_volume: float | None = None,
    min_down_volume: float | None = None,
) -> pd.Series:
    """Compute instantaneous regulation class per timestamp from up/down volumes.
    - If both 0 -> 'none'
    - If both >0 -> choose the larger; if equal volumes, tie_break = 'up'|'down'
    Returns a pandas Series named 'RegClass'.
    """
    up = pd.to_numeric(mfrr_df.get(up_col, 0), errors='coerce').fillna(0)
    down = pd.to_numeric(mfrr_df.get(down_col, 0), errors='coerce').fillna(0)

    # Apply volume thresholds (treat tiny activations as noise -> none)
    if min_up_volume is not None:
        up = up.where(up >= float(min_up_volume), 0)
    if min_down_volume is not None:
        down = down.where(down >= float(min_down_volume), 0)

    # Initialize as 'none'
    reg = pd.Series(data='none', index=mfrr_df.index, dtype='object')

    up_pos = up.gt(0)
    down_pos = down.gt(0)

    # Only up
    reg[up_pos & ~down_pos] = 'up'

    # Only down
    reg[down_pos & ~up_pos] = 'down'
    # Both positive -> tie break on volume

    both = up_pos & down_pos
    if both.any():
        # If tie_break == 'up' we prefer 'up' on equality, otherwise prefer 'down' on equality.
        if tie_break == 'up':
            up_bigger = up[both] >= down[both]
        else:
            up_bigger = up[both] > down[both]
        reg[both] = np.where(up_bigger, 'up', 'down')
    reg.name = 'RegClass'
    return reg
