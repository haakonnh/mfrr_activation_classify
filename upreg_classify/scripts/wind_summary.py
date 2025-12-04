from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------- Robust CSV Loader ----------------------------

def read_csv_robust(path: Path):
    for delim in (',', ';', '\t'):
        try:
            df = pd.read_csv(path, delimiter=delim)
            # Heuristic: ensure at least 2 non-empty columns beside time
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    # last resort
    return pd.read_csv(path)


# ---------------------------- Data Loading ----------------------------

def load_wind_forecasts(data_dir: Path) -> pd.DataFrame:
    # Try to find forecast files by glob (covers both 2024->2025 and 2025->2026)
    cands = sorted((data_dir).glob('Generation Forecasts for Wind and Solar_*.csv'))
    if not cands:
        raise FileNotFoundError("No 'Generation Forecasts for Wind and Solar_*.csv' found in data_dir")
    frames: List[pd.DataFrame] = []
    for p in cands:
        df = read_csv_robust(p)
        # Normalize time column and keep only needed series for NO1
        # Expect time col name like 'MTU (CET/CEST)'
        time_col = None
        for cand in ['MTU (CET/CEST)', 'MTU', 'Time']:
            if cand in df.columns:
                time_col = cand
                break
        if time_col is None:
            raise KeyError(f"No time column found in {p}")
        df.rename(columns={time_col: 'Time'}, inplace=True)
        df['Time'] = df['Time'].astype(str).str[:16]
        df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M', errors='coerce')
        # Columns
        da_col = 'Generation - Wind Onshore [MW] Day Ahead/ BZN|NO1'
        id_col = 'Generation - Wind Onshore [MW] Intraday / BZN|NO1'
        # Some files might have slight variants; try to find best match
        if da_col not in df.columns:
            da_alt = [c for c in df.columns if 'Wind Onshore' in c and 'Day Ahead' in c and 'NO1' in c]
            if da_alt:
                da_col = da_alt[0]
        if id_col not in df.columns:
            id_alt = [c for c in df.columns if 'Wind Onshore' in c and 'Intraday' in c and 'NO1' in c]
            if id_alt:
                id_col = id_alt[0]
        for c in (da_col, id_col):
            if c not in df.columns:
                raise KeyError(f"Missing expected forecast column in {p}: {c}")
        df = df[['Time', da_col, id_col]].copy()
        frames.append(df)
    wf = pd.concat(frames, ignore_index=True)
    wf = wf.dropna(subset=['Time']).sort_values('Time')
    wf = wf[~wf['Time'].duplicated(keep='first')]
    for col in wf.columns:
        if col != 'Time':
            wf[col] = pd.to_numeric(wf[col], errors='coerce')
    wf.set_index('Time', inplace=True)
    wf = wf.resample('15min').ffill()
    wf.rename(columns={
        da_col: 'wind_da',
        id_col: 'wind_id',
    }, inplace=True)
    return wf


def load_production(data_dir: Path, years: List[int]) -> pd.DataFrame:
    frames = []
    for y in years:
        p = data_dir / f'Production_{y}_NO1_None_MW.csv'
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p, delimiter=';')
        tcol = 'Delivery Start (CET)' if 'Delivery Start (CET)' in df.columns else 'Delivery Start'
        df.rename(columns={tcol: 'Time'}, inplace=True)
        df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
        # Include total production to compute wind share
        keep_cols = ['Time', 'NO1 Wind Onshore (MW)', 'NO1 Total (MW)']
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing expected columns in production file {p}: {missing}")
        df = df[keep_cols].copy()
        frames.append(df)
    prod = pd.concat(frames, ignore_index=True)
    prod = prod.dropna(subset=['Time']).sort_values('Time')
    prod = prod[~prod['Time'].duplicated(keep='first')]
    prod.set_index('Time', inplace=True)
    prod = prod.resample('15min').ffill()
    prod.rename(columns={'NO1 Wind Onshore (MW)': 'wind_actual', 'NO1 Total (MW)': 'prod_total'}, inplace=True)
    return prod


def load_balance_market(data_dir: Path, years: List[int]) -> pd.DataFrame:
    frames = []
    for y in years:
        p = data_dir / f'BalanceMarket_{y}_NO1_EUR_None_MW.csv'
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p, delimiter=';')
        tcol = 'Delivery Start (CET)' if 'Delivery Start (CET)' in df.columns else 'Delivery Start'
        df.rename(columns={tcol: 'Time'}, inplace=True)
        df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
        df = df[['Time', 'NO1 Activated Up Volume (MW)', 'NO1 Activated Down Volume (MW)']].copy()
        frames.append(df)
    bm = pd.concat(frames, ignore_index=True)
    bm = bm.dropna(subset=['Time']).sort_values('Time')
    bm = bm[~bm['Time'].duplicated(keep='first')]
    bm.set_index('Time', inplace=True)
    for col in ['NO1 Activated Up Volume (MW)', 'NO1 Activated Down Volume (MW)']:
        bm[col] = pd.to_numeric(bm[col], errors='coerce').fillna(0.0)
    return bm


def build_activation_label(bm: pd.DataFrame, min_up: float = 0.0, min_down: float = 0.0) -> pd.Series:
    up = bm['NO1 Activated Up Volume (MW)']
    down = bm['NO1 Activated Down Volume (MW)']
    up_active = up > float(min_up)
    down_active = down > float(min_down)
    label = pd.Series('none', index=bm.index)
    label[up_active & ~down_active] = 'up'
    label[down_active & ~up_active] = 'down'
    both = up_active & down_active
    if both.any():
        label.loc[both] = np.where(up[both] >= down[both], 'up', 'down')
    return label


# ---------------------------- Feature Construction ----------------------------

def derive_wind_metrics(wf: pd.DataFrame, prod: pd.DataFrame) -> pd.DataFrame:
    df = wf.merge(prod, left_index=True, right_index=True, how='outer')
    df.sort_index(inplace=True)
    # Forecast revisions and errors
    df['wind_revision'] = (df['wind_id'] - df['wind_da']).abs()
    df['wind_error_da'] = df['wind_actual'] - df['wind_da']
    df['wind_error_id'] = df['wind_actual'] - df['wind_id']
    # Wind share based on actuals and total production
    denom_total = df['prod_total'].replace(0, np.nan)
    df['wind_share'] = (df['wind_actual'] / denom_total).clip(lower=0, upper=1)
    # Relative errors (clip extreme ratios)
    denom = df['wind_actual'].replace(0, np.nan)
    df['abs_error_da_pct'] = (df['wind_error_da'].abs() / denom).clip(upper=5.0)
    df['abs_error_id_pct'] = (df['wind_error_id'].abs() / denom).clip(upper=5.0)
    # Ramps
    df['wind_actual_ramp_1h'] = df['wind_actual'].diff(4)
    df['wind_revision_ramp_1h'] = df['wind_revision'].diff(4)
    return df


# ---------------------------- Summaries ----------------------------

def summarize_wind(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['wind_da', 'wind_id', 'wind_actual', 'wind_revision', 'wind_error_da', 'wind_error_id', 'abs_error_da_pct', 'abs_error_id_pct', 'wind_share']
    rows = []
    for c in cols:
        s = df[c]
        rows.append({
            'Metric': c,
            'Mean': s.mean(),
            'Std': s.std(),
            'Min': s.min(),
            'P10': s.quantile(0.10),
            'P50': s.quantile(0.50),
            'P90': s.quantile(0.90),
            'Max': s.max(),
            'Count': s.count(),
        })
    return pd.DataFrame(rows)


def summarize_by_activation(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['wind_da', 'wind_id', 'wind_actual', 'wind_revision', 'wind_error_da', 'wind_error_id', 'wind_share', 'wind_actual_ramp_1h', 'wind_revision_ramp_1h']
    res = []
    for act, g in df.groupby('Activation'):
        row = {'Activation': act, 'Count': len(g)}
        for c in cols:
            row[f'{c} Mean'] = g[c].mean()
        res.append(row)
    return pd.DataFrame(res).sort_values('Count', ascending=False)


def activation_probability_by_metric(df: pd.DataFrame, col: str, bins: int = 30) -> pd.DataFrame:
    s = df[col]
    cats = pd.cut(s, bins=bins)
    probs = df.groupby(cats)['Activation'].value_counts(normalize=True).rename('Probability').reset_index()
    probs[col + ' Mid'] = probs[col].apply(lambda iv: (iv.left + iv.right) / 2 if hasattr(iv, 'left') else np.nan)
    return probs


# ---------------------------- Plotting ----------------------------

def plot_distributions(df: pd.DataFrame, out_dir: Path | None):
    sns.set_theme(context='talk', style='whitegrid', font_scale=0.9)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    specs = [
        ('wind_actual', 'Wind Actual (MW)'),
        ('wind_da', 'Wind Day-Ahead (MW)'),
        ('wind_id', 'Wind Intraday (MW)'),
        ('wind_revision', '|Intraday - Day-Ahead| (MW)'),
        ('wind_error_da', 'Actual - Day-Ahead (MW)'),
        ('wind_error_id', 'Actual - Intraday (MW)'),
    ]
    for ax, (col, title) in zip(axes.ravel(), specs):
        s = df[col].dropna()
        sns.histplot(s, bins=40, kde=True, stat='density', ax=ax, color='#4C72B0', edgecolor='white', alpha=0.85)
        p10, p50, p90 = s.quantile([0.1, 0.5, 0.9])
        for val, lab, colr in [(p10, 'P10', '#dd8452'), (p50, 'P50', '#55a868'), (p90, 'P90', '#c44e52')]:
            ax.axvline(val, color=colr, linestyle='--', lw=1.2)
        ax.set_title(title)
        ax.set_ylabel('Density')
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / 'wind_distributions.png'
        fig.savefig(p, dpi=180)
        print(f'Saved: {p}')
    plt.show()


def plot_activation_vs_metric(probs: pd.DataFrame, metric: str, out_dir: Path | None):
    sns.set_theme(context='talk', style='whitegrid', font_scale=0.9)
    fig, ax = plt.subplots(figsize=(8, 5))
    mid_col = metric + ' Mid'
    for act in ['up', 'down', 'none']:
        sub = probs[probs['Activation'] == act]
        if not len(sub):
            continue
        ax.plot(sub[mid_col], sub['Probability'], label=act, lw=2)
    ax.set_xlabel(metric)
    ax.set_ylabel('Activation Probability')
    ax.set_title(f'Activation probability vs {metric}')
    ax.legend(title='Activation')
    ax.grid(alpha=0.3)
    if out_dir:
        p = out_dir / f'activation_probability_vs_{metric.replace(" ", "_")}.png'
        fig.savefig(p, dpi=180)
        print(f'Saved: {p}')
    plt.show()


def plot_hour_month_heatmap(df: pd.DataFrame, col: str, out_dir: Path | None):
    sns.set_theme(context='talk', style='whitegrid', font_scale=0.9)
    tmp = df.reset_index()
    tmp['Hour'] = tmp['Time'].dt.hour
    tmp['Month'] = tmp['Time'].dt.month
    pivot = tmp.pivot_table(index='Hour', columns='Month', values=col, aggfunc='mean')
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, cmap='viridis', ax=ax)
    ax.set_title(f'Average {col} by Hour and Month')
    if out_dir:
        p = out_dir / f'{col}_hour_month_heatmap.png'
        fig.savefig(p, dpi=180)
        print(f'Saved: {p}')
    plt.show()


def plot_scatter_activation(df: pd.DataFrame, out_dir: Path | None, sample: int = 8000):
    sns.set_theme(context='talk', style='whitegrid', font_scale=0.9)
    tmp = df.dropna(subset=['wind_revision', 'wind_actual', 'Activation']).copy()
    if len(tmp) > sample:
        tmp = tmp.sample(sample, random_state=42)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.scatterplot(data=tmp, x='wind_revision', y='wind_actual', hue='Activation',
                    palette={'up': '#d62728', 'down': '#1f77b4', 'none': '#7f7f7f'}, alpha=0.6, ax=ax)
    ax.set_title('Activation vs Wind Revision and Wind Actual')
    ax.set_xlabel('Revision |ID - DA| (MW)')
    ax.set_ylabel('Wind Actual (MW)')
    ax.grid(alpha=0.3)
    if out_dir:
        p = out_dir / 'activation_scatter_revision_vs_actual.png'
        fig.savefig(p, dpi=180)
        print(f'Saved: {p}')
    plt.show()


# ---------------------------- Main ----------------------------

def default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / 'data' / 'raw'


def main() -> None:
    parser = argparse.ArgumentParser(description='Wind-focused summary: forecasts vs actuals, errors, revisions, and activation correlation.')
    parser.add_argument('--data-dir', type=Path, default=default_data_dir(), help='Directory containing raw CSVs')
    parser.add_argument('--years', type=int, nargs='*', default=[2024, 2025], help='Years to include for production and activations')
    parser.add_argument('--min-up', type=float, default=0.0, help='Minimum Up activation (MW) threshold')
    parser.add_argument('--min-down', type=float, default=0.0, help='Minimum Down activation (MW) threshold')
    parser.add_argument('--out-dir', type=Path, default=None, help='Optional output directory for figures & summaries')
    args = parser.parse_args()

    print('Loading wind forecasts...')
    wf = load_wind_forecasts(args.data_dir)
    print('Loading wind actual production...')
    prod = load_production(args.data_dir, args.years)
    print('Merging and deriving metrics...')
    df = derive_wind_metrics(wf, prod)

    print('Loading activations and building labels...')
    bm = load_balance_market(args.data_dir, args.years)
    # Align activation label to df index
    act = build_activation_label(bm, args.min_up, args.min_down)
    df = df.merge(act.rename('Activation'), how='left', left_index=True, right_index=True)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Time'}, inplace=True)

    # Summaries
    core = summarize_wind(df.set_index('Time'))
    cond = summarize_by_activation(df)
    probs_rev = activation_probability_by_metric(df, 'wind_revision')
    probs_actual = activation_probability_by_metric(df, 'wind_actual')
    # New: probabilities vs wind_share and vs day-ahead minus actual
    probs_share = activation_probability_by_metric(df, 'wind_share')
    probs_da_err = activation_probability_by_metric(df, 'wind_error_da')

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        core_csv = args.out_dir / 'wind_summary_stats.csv'
        cond_csv = args.out_dir / 'wind_activation_conditional_summary.csv'
        prob_rev_csv = args.out_dir / 'activation_probability_vs_wind_revision.csv'
        prob_actual_csv = args.out_dir / 'activation_probability_vs_wind_actual.csv'
        core.to_csv(core_csv, index=False)
        cond.to_csv(cond_csv, index=False)
        probs_rev.to_csv(prob_rev_csv, index=False)
        probs_actual.to_csv(prob_actual_csv, index=False)
        # Save new outputs
        prob_share_csv = args.out_dir / 'activation_probability_vs_wind_share.csv'
        prob_da_err_csv = args.out_dir / 'activation_probability_vs_wind_error_da.csv'
        probs_share.to_csv(prob_share_csv, index=False)
        probs_da_err.to_csv(prob_da_err_csv, index=False)
        print(f'Saved: {core_csv}\nSaved: {cond_csv}\nSaved: {prob_rev_csv}\nSaved: {prob_actual_csv}\nSaved: {prob_share_csv}\nSaved: {prob_da_err_csv}')

    # Plots
    plot_distributions(df.set_index('Time'), args.out_dir)
    plot_activation_vs_metric(probs_rev, 'wind_revision', args.out_dir)
    plot_activation_vs_metric(probs_actual, 'wind_actual', args.out_dir)
    # New plots
    plot_activation_vs_metric(probs_share, 'wind_share', args.out_dir)
    plot_activation_vs_metric(probs_da_err, 'wind_error_da', args.out_dir)
    plot_hour_month_heatmap(df.set_index('Time'), 'wind_actual', args.out_dir)
    plot_scatter_activation(df, args.out_dir)


if __name__ == '__main__':
    main()
