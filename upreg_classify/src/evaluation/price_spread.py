from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriceSpreadResult:
    df: pd.DataFrame
    spread_up_correct: pd.Series
    spread_down_correct: pd.Series
    up_pct_correct: pd.Series
    down_pct_correct: pd.Series


def _series_summary(name: str, s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {
            "name": name,
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    q = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "name": name,
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p05": float(q.loc[0.05]),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.50]),
        "p75": float(q.loc[0.75]),
        "p95": float(q.loc[0.95]),
        "max": float(s.max()),
    }


def price_spread_stats(result: PriceSpreadResult) -> dict[str, object]:
    """Compute the same stats previously printed in the notebook."""
    spread_up = result.spread_up_correct
    spread_down = result.spread_down_correct

    stats_table = pd.DataFrame(
        [
            _series_summary("Up Price - DA | correct UP (delivery)", spread_up),
            _series_summary("Down Price - DA | correct DOWN (delivery)", spread_down),
            _series_summary("UP % change vs DA | correct UP (delivery) [percent]", result.up_pct_correct),
            _series_summary("DOWN % change vs DA | correct DOWN (delivery) [percent]", result.down_pct_correct),
        ]
    )

    sign_checks = {
        "negatives_in_up_spread": int((pd.to_numeric(spread_up, errors="coerce") < 0).sum()),
        "positives_in_down_spread": int((pd.to_numeric(spread_down, errors="coerce") > 0).sum()),
    }
    return {"sign_checks": sign_checks, "summary": stats_table}


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    if "Time" in df.columns:
        out = df.copy()
        out["Time"] = pd.to_datetime(out["Time"], errors="coerce")
        out = out.dropna(subset=["Time"]).set_index("Time")
        return out
    # Last attempt: try to parse index
    try:
        out = df.copy()
        out.index = pd.to_datetime(out.index, errors="raise")
        return out
    except Exception as e:
        raise ValueError("predictions dataframe must have a DatetimeIndex or a 'Time' column") from e


def load_balance_market_prices(data_raw_dir: str | Path, area: str, include_2024: bool = True) -> pd.DataFrame:
    data_raw_dir = Path(data_raw_dir)
    bm_2025 = data_raw_dir / "balancing" / f"BalanceMarket_2025_{area}_EUR_None_MW.csv"
    bm_2024 = data_raw_dir / "balancing" / f"BalanceMarket_2024_{area}_EUR_None_MW.csv"

    df = pd.read_csv(bm_2025, delimiter=";")
    if include_2024 and bm_2024.exists():
        df24 = pd.read_csv(bm_2024, delimiter=";")
        df = pd.concat([df24, df], ignore_index=True)

    # Normalize time col
    if "Delivery Start (CET)" in df.columns:
        df = df.rename(columns={"Delivery Start (CET)": "Time"})
    elif "Delivery Start" in df.columns:
        df = df.rename(columns={"Delivery Start": "Time"})

    df["Time"] = pd.to_datetime(df["Time"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["Time"]).set_index("Time")
    df = df[~df.index.duplicated(keep="first")].sort_index()

    up_col = f"{area} Up Price (EUR)"
    down_col = f"{area} Down Price (EUR)"
    for col in [up_col, down_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    return df[[up_col, down_col]]


def load_day_ahead_price(data_raw_dir: str | Path, area: str, include_2024: bool = True) -> pd.DataFrame:
    data_raw_dir = Path(data_raw_dir)
    da_2025 = data_raw_dir / "prices" / f"AuctionPrice_2025_DayAhead_{area}_EUR_None.csv"
    da_2024 = data_raw_dir / "prices" / f"AuctionPrice_2024_DayAhead_{area}_EUR_None.csv"

    da = pd.read_csv(da_2025, delimiter=";")
    if include_2024 and da_2024.exists():
        da24 = pd.read_csv(da_2024, delimiter=";")
        da = pd.concat([da24, da], ignore_index=True)

    # Normalize time col
    if "Delivery Start (CET)" in da.columns:
        da = da.rename(columns={"Delivery Start (CET)": "Time"})
    elif "Delivery Start" in da.columns:
        da = da.rename(columns={"Delivery Start": "Time"})

    da["Time"] = pd.to_datetime(da["Time"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    da = da.dropna(subset=["Time"]).set_index("Time")
    da = da[~da.index.duplicated(keep="first")].sort_index()

    price_col = f"{area} Price (EUR)"
    if price_col not in da.columns:
        raise KeyError(f"Missing day-ahead price column: {price_col}")

    da[price_col] = pd.to_numeric(da[price_col].replace("", np.nan), errors="coerce")

    # Resample to 15min to match the rest of the pipeline
    da = da.resample("15min").ffill().bfill()
    da = da.rename(columns={price_col: "DA Price"})
    return da[["DA Price"]]


def compute_delivery_spreads(
    preds: pd.DataFrame,
    data_raw_dir: str | Path,
    area: str,
    horizon_steps: int = 4,
    include_2024: bool = True,
    min_da_price: float = 20.0,
    label_col: Optional[str] = None,
    pred_col: str = "pred",
) -> PriceSpreadResult:
    """Join predictions with BM + DA and compute delivery-time spreads.

    `preds` must contain ground truth labels and a `pred` column.
    Index is expected to be the feature-time; delivery is shift(-horizon_steps).
    """
    preds = _ensure_datetime_index(preds)
    if pred_col not in preds.columns:
        raise ValueError("preds must contain a 'pred' column")

    if label_col is None:
        label_col = "RegClass+4" if "RegClass+4" in preds.columns else None
        if label_col is None:
            # First non-proba, non-pred column
            for c in preds.columns:
                if str(c) == pred_col or str(c).startswith("p_"):
                    continue
                label_col = str(c)
                break
    if not label_col or label_col not in preds.columns:
        raise ValueError("Could not infer label column in preds")

    bm = load_balance_market_prices(data_raw_dir, area=area, include_2024=include_2024)
    da = load_day_ahead_price(data_raw_dir, area=area, include_2024=include_2024)

    up_col = f"{area} Up Price (EUR)"
    down_col = f"{area} Down Price (EUR)"

    df = preds[[label_col, pred_col]].join(bm, how="left").join(da, how="left")

    # Move to delivery (t+horizon_steps)
    for c in [up_col, down_col, "DA Price"]:
        if c in df.columns:
            df[c + " @delivery"] = df[c].shift(-horizon_steps)

    # Correct masks
    y_true = df[label_col].astype(str)
    y_pred = df[pred_col].astype(str)
    is_up_correct = (y_pred == "up") & (y_true == "up")
    is_down_correct = (y_pred == "down") & (y_true == "down")

    df["Up-DA Spread @delivery"] = df[up_col + " @delivery"] - df["DA Price @delivery"]
    df["Down-DA Spread @delivery"] = df[down_col + " @delivery"] - df["DA Price @delivery"]

    eps = 1e-6
    valid_da = (df["DA Price @delivery"].abs() > eps) & (df["DA Price @delivery"] > float(min_da_price))
    up_pct = (df["Up-DA Spread @delivery"] / df["DA Price @delivery"]).where(valid_da) * 100.0
    down_pct = (df["Down-DA Spread @delivery"] / df["DA Price @delivery"]).where(valid_da) * 100.0

    spread_up = df.loc[is_up_correct, "Up-DA Spread @delivery"].dropna()
    spread_down = df.loc[is_down_correct, "Down-DA Spread @delivery"].dropna()
    up_pct_correct = up_pct.loc[is_up_correct].dropna()
    down_pct_correct = down_pct.loc[is_down_correct].dropna()

    return PriceSpreadResult(
        df=df,
        spread_up_correct=spread_up,
        spread_down_correct=spread_down,
        up_pct_correct=up_pct_correct,
        down_pct_correct=down_pct_correct,
    )


def plot_imbalance_vs_spot_price(
    preds: pd.DataFrame,
    data_raw_dir: str | Path,
    area: str,
    horizon_steps: int = 4,
    include_2024: bool = True,
    min_da_price: float = 20.0,
    label_col: Optional[str] = None,
    title: Optional[str] = None,
    bw_adjust: float = 0.8,
    print_stats: bool = True,
):
    """Reproduce the notebook "price thing" plot.

    Returns (fig, ax, result) where result contains the joined df + spread series.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    result = compute_delivery_spreads(
        preds=preds,
        data_raw_dir=data_raw_dir,
        area=area,
        horizon_steps=horizon_steps,
        include_2024=include_2024,
        min_da_price=min_da_price,
        label_col=label_col,
    )

    stats = price_spread_stats(result)
    if print_stats:
        sc = stats["sign_checks"]
        print("negatives in UP spread:", sc["negatives_in_up_spread"])
        print("positives in DOWN spread:", sc["positives_in_down_spread"])
        print(stats["summary"].to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Boundary-aware KDE
    if len(result.spread_up_correct):
        sns.kdeplot(
            result.spread_up_correct,
            fill=True,
            color="#d62728",
            label="UP correct: Up-DA (delivery)",
            clip=(0, None),
            cut=0,
            bw_adjust=bw_adjust,
            ax=ax,
        )
    if len(result.spread_down_correct):
        sns.kdeplot(
            result.spread_down_correct,
            fill=True,
            color="#1f77b4",
            label="DOWN correct: Down-DA (delivery)",
            clip=(None, 0),
            cut=0,
            bw_adjust=bw_adjust,
            ax=ax,
        )

    ax.axvline(0, color="k", lw=1, alpha=0.6)
    if title is None:
        title = "Up/Down Price minus Day-Ahead at delivery (correct predictions)"
    ax.set_title(title)
    ax.set_xlabel("EUR/MWh")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()

    return fig, ax, result, stats
