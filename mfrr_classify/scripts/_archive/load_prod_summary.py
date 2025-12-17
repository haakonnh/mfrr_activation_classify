from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------- Data Loading ----------------------------

def load_consumption_production(data_dir: Path, years: list[int]) -> pd.DataFrame:
    cons_frames = []
    prod_frames = []
    for y in years:
        cons_path = data_dir / f'Consumption_{y}_NO1_None_MW.csv'
        prod_path = data_dir / f'Production_{y}_NO1_None_MW.csv'
        if not cons_path.exists():
            raise FileNotFoundError(cons_path)
        if not prod_path.exists():
            raise FileNotFoundError(prod_path)
        c = pd.read_csv(cons_path, delimiter=';')
        p = pd.read_csv(prod_path, delimiter=';')
        # Normalize time column
        for df in (c, p):
            if 'Delivery Start (CET)' in df.columns:
                df.rename(columns={'Delivery Start (CET)': 'Time'}, inplace=True)
            elif 'Delivery Start' in df.columns:
                df.rename(columns={'Delivery Start': 'Time'}, inplace=True)
            df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
        cons_frames.append(c[['Time', 'NO1 Volume (MW)']])
        prod_frames.append(p[['Time', 'NO1 Wind Onshore (MW)', 'NO1 Total (MW)']])
    cons = pd.concat(cons_frames, ignore_index=True).dropna(subset=['Time']).sort_values('Time')
    prod = pd.concat(prod_frames, ignore_index=True).dropna(subset=['Time']).sort_values('Time')
    # Drop duplicates keep first
    cons = cons[~cons['Time'].duplicated(keep='first')]
    prod = prod[~prod['Time'].duplicated(keep='first')]
    df = cons.merge(prod, on='Time', how='inner')
    return df


def load_balance_market(data_dir: Path, years: list[int]) -> pd.DataFrame:
    frames = []
    for y in years:
        path = data_dir / f'BalanceMarket_{y}_NO1_EUR_None_MW.csv'
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, delimiter=';')
        if 'Delivery Start (CET)' in df.columns:
            df = df.rename(columns={'Delivery Start (CET)': 'Time'})
        elif 'Delivery Start' in df.columns:
            df = df.rename(columns={'Delivery Start': 'Time'})
        df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
        frames.append(df[['Time', 'NO1 Activated Up Volume (MW)', 'NO1 Activated Down Volume (MW)']])
    out = pd.concat(frames, ignore_index=True).dropna(subset=['Time']).sort_values('Time')
    out = out[~out['Time'].duplicated(keep='first')]
    for col in ['NO1 Activated Up Volume (MW)', 'NO1 Activated Down Volume (MW)']:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0.0)
    return out


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

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Reserve Margin'] = df['NO1 Total (MW)'] - df['NO1 Volume (MW)']  # production - load
    df['Residual Load'] = df['NO1 Volume (MW)'] - df['NO1 Wind Onshore (MW)']
    denom = df['NO1 Total (MW)'].replace(0, np.nan)
    df['Wind Share'] = (df['NO1 Wind Onshore (MW)'] / denom).clip(lower=0, upper=1)
    # 1h ramps (difference vs previous hour) assuming hourly cadence
    df['Load Ramp'] = df['NO1 Volume (MW)'].diff()
    df['Production Ramp'] = df['NO1 Total (MW)'].diff()
    df['Reserve Margin Ramp'] = df['Reserve Margin'].diff()
    df['Residual Load Ramp'] = df['Residual Load'].diff()
    return df


# ---------------------------- Summaries ----------------------------

def summarize_core(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['NO1 Volume (MW)', 'NO1 Total (MW)', 'NO1 Wind Onshore (MW)', 'Reserve Margin', 'Residual Load', 'Wind Share']
    stats = []
    for c in cols:
        s = df[c]
        stats.append({
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
    return pd.DataFrame(stats)


def summarize_by_activation(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ['NO1 Volume (MW)', 'NO1 Total (MW)', 'NO1 Wind Onshore (MW)', 'Reserve Margin', 'Residual Load', 'Wind Share', 'Load Ramp', 'Reserve Margin Ramp']
    rows = []
    for act, g in df.groupby('Activation'):
        row = {'Activation': act, 'Count': len(g)}
        for c in group_cols:
            row[f'{c} Mean'] = g[c].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values('Count', ascending=False)


def activation_probability_by_margin(df: pd.DataFrame, bins: int = 30) -> pd.DataFrame:
    margin = df['Reserve Margin']
    cats = pd.cut(margin, bins=bins)
    probs = df.groupby(cats)['Activation'].value_counts(normalize=True).rename('Probability').reset_index()
    # Rename interval to midpoint value for simpler plotting
    probs['MarginMid'] = probs['Reserve Margin'].apply(lambda iv: (iv.left + iv.right) / 2 if hasattr(iv, 'left') else np.nan)
    return probs


# ---------------------------- Plotting ----------------------------

def plot_distributions(df: pd.DataFrame, out_dir: Path | None):
    sns.set_theme(context='talk', style='whitegrid', font_scale=0.9)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    targets = [
        ('Reserve Margin', 'Reserve Margin'),
        ('Residual Load', 'Residual Load'),
        ('Wind Share', 'Wind Share'),
    ]
    for ax, (col, title) in zip(axes, targets):
        s = df[col].dropna()
        sns.histplot(s, bins=40, kde=True, stat='density', ax=ax, color='#4C72B0', edgecolor='white', alpha=0.85)
        p10, p50, p90 = s.quantile([0.1, 0.5, 0.9])
        for val, lab, colr in [(p10, 'P10', '#dd8452'), (p50, 'P50', '#55a868'), (p90, 'P90', '#c44e52')]:
            ax.axvline(val, color=colr, linestyle='--', lw=1.4)
            ax.text(val, ax.get_ylim()[1] * 0.85, lab, rotation=90, ha='right', va='top', fontsize=9, color=colr)
        ax.set_title(title)
        ax.set_ylabel('Density')
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / 'core_distributions.png'
        fig.savefig(p, dpi=180)
        print(f'Saved: {p}')
    plt.show()


def plot_wind_share_vs_margin(df: pd.DataFrame, out_dir: Path | None, sample: int = 8000):
    sns.set_theme(context='talk', style='whitegrid', font_scale=0.9)
    if len(df) > sample:
        df_plot = df.sample(sample, random_state=42)
    else:
        df_plot = df
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df_plot, x='Wind Share', y='Reserve Margin', hue='Activation', palette={'up': '#d62728', 'down': '#1f77b4', 'none': '#7f7f7f'}, alpha=0.6, ax=ax)
    ax.set_title('Activation direction vs Wind Share & Reserve Margin')
    ax.set_xlabel('Wind Share')
    ax.set_ylabel('Reserve Margin (MW)')
    ax.grid(alpha=0.3)
    if out_dir:
        p = out_dir / 'wind_share_vs_reserve_margin.png'
        fig.savefig(p, dpi=180)
        print(f'Saved: {p}')
    plt.show()


def plot_margin_activation_probability(probs: pd.DataFrame, out_dir: Path | None):
    sns.set_theme(context='talk', style='whitegrid', font_scale=0.9)
    fig, ax = plt.subplots(figsize=(8, 5))
    for act in ['up', 'down', 'none']:
        sub = probs[probs['Activation'] == act]
        if not len(sub):
            continue
        ax.plot(sub['MarginMid'], sub['Probability'], label=act, lw=2)
    ax.set_xlabel('Reserve Margin (MW) midpoint of bin')
    ax.set_ylabel('Activation Probability')
    ax.set_title('Activation probability vs Reserve Margin')
    ax.legend(title='Activation')
    ax.grid(alpha=0.3)
    if out_dir:
        p = out_dir / 'activation_probability_vs_reserve_margin.png'
        fig.savefig(p, dpi=180)
        print(f'Saved: {p}')
    plt.show()


def plot_hourly_heatmap(df: pd.DataFrame, out_dir: Path | None):
    sns.set_theme(context='talk', style='whitegrid', font_scale=0.9)
    df['Hour'] = df['Time'].dt.hour
    df['Month'] = df['Time'].dt.month
    pivot = df.pivot_table(index='Hour', columns='Month', values='Reserve Margin', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, cmap='coolwarm', ax=ax)
    ax.set_title('Average Reserve Margin by Hour and Month')
    if out_dir:
        p = out_dir / 'reserve_margin_hour_month_heatmap.png'
        fig.savefig(p, dpi=180)
        print(f'Saved: {p}')
    plt.show()


# ---------------------------- Main ----------------------------

def default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / 'data' / 'raw'


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize and visualize load & production forecast derived features and activation correlation.')
    parser.add_argument('--data-dir', type=Path, default=default_data_dir(), help='Directory containing raw CSVs')
    parser.add_argument('--years', type=int, nargs='*', default=[2024, 2025], help='Years to include (default: 2024 2025)')
    parser.add_argument('--min-up', type=float, default=0.0, help='Minimum Up activation (MW) threshold')
    parser.add_argument('--min-down', type=float, default=0.0, help='Minimum Down activation (MW) threshold')
    parser.add_argument('--out-dir', type=Path, default=None, help='Optional output directory for figures & summaries')
    args = parser.parse_args()

    print('Loading consumption & production...')
    cp = load_consumption_production(args.data_dir, args.years)
    print('Rows (cons+prod merged):', len(cp))
    print('Loading activations...')
    bm = load_balance_market(args.data_dir, args.years)
    print('Rows (balance market):', len(bm))

    # Merge & derive features
    df = cp.merge(bm, on='Time', how='left')
    df.sort_values('Time', inplace=True)
    df = derive_features(df)
    df['Activation'] = build_activation_label(df[['NO1 Activated Up Volume (MW)', 'NO1 Activated Down Volume (MW)']].fillna(0.0), args.min_up, args.min_down)

    core_summary = summarize_core(df)
    act_summary = summarize_by_activation(df)
    probs = activation_probability_by_margin(df)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        core_csv = args.out_dir / 'core_load_prod_summary.csv'
        act_csv = args.out_dir / 'activation_conditional_summary.csv'
        probs_csv = args.out_dir / 'activation_probability_vs_margin.csv'
        core_summary.to_csv(core_csv, index=False)
        act_summary.to_csv(act_csv, index=False)
        probs.to_csv(probs_csv, index=False)
        print(f'Saved: {core_csv}\nSaved: {act_csv}\nSaved: {probs_csv}')

    print('\nCore Summary:')
    print(core_summary.to_string(index=False))
    print('\nConditional Means by Activation:')
    print(act_summary.to_string(index=False))

    # Plots
    plot_distributions(df, args.out_dir)
    plot_wind_share_vs_margin(df, args.out_dir)
    plot_margin_activation_probability(probs, args.out_dir)
    plot_hourly_heatmap(df, args.out_dir)


if __name__ == '__main__':
    main()
