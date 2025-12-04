from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_balance_market(paths: list[Path]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        df = pd.read_csv(p, delimiter=';')
        # Normalize time column name and parse
        if 'Delivery Start (CET)' in df.columns:
            df = df.rename(columns={'Delivery Start (CET)': 'Time'})
        elif 'Delivery Start' in df.columns:
            df = df.rename(columns={'Delivery Start': 'Time'})
        df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out = out.dropna(subset=['Time']).sort_values('Time')
    out = out[~out['Time'].duplicated(keep='first')]
    # Ensure numeric activated volumes
    for col in ['NO1 Activated Up Volume (MW)', 'NO1 Activated Down Volume (MW)']:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0.0)
        else:
            out[col] = 0.0
    return out[['Time', 'NO1 Activated Up Volume (MW)', 'NO1 Activated Down Volume (MW)']]


def compute_regclass(df: pd.DataFrame, min_up: float = 0.0, min_down: float = 0.0) -> pd.Series:
    """
    Build a ternary class from activated volumes.

    Rules (strict thresholds to allow a real 'none' class):
    - up if (up > min_up) and (down <= min_down)
    - down if (down > min_down) and (up <= min_up)
    - none if (up <= min_up) and (down <= min_down)
    - if both (up > min_up) and (down > min_down): pick the larger side (ties -> 'up')
    """
    up = df['NO1 Activated Up Volume (MW)']
    down = df['NO1 Activated Down Volume (MW)']

    # Strictly greater than threshold to avoid classifying zeros as active
    up_active = up > float(min_up)
    down_active = down > float(min_down)

    label = pd.Series('none', index=df.index)
    # Exclusive cases
    label[up_active & ~down_active] = 'up'
    label[down_active & ~up_active] = 'down'

    # Both sides active: choose the larger volume (ties -> 'up')
    both = up_active & down_active
    if both.any():
        label.loc[both] = np.where(up[both] >= down[both], 'up', 'down')
    return label


def summarize_distribution(labels: pd.Series) -> pd.DataFrame:
    counts = labels.value_counts()
    total = int(len(labels))
    pct = (counts / total * 100)
    summary = (
        pd.DataFrame({'Class': counts.index, 'Count': counts.values, 'Percentage': pct.values})
        .sort_values('Count', ascending=False)
        .reset_index(drop=True)
    )
    return summary


def plot_distribution(summary: pd.DataFrame, out_dir: Path | None = None, title_prefix: str = 'mFRR Activation Direction') -> None:
    # Keep consistent ordering when possible
    order = [c for c in ['none', 'down', 'up'] if c in summary['Class'].tolist()]
    colors = {'up': '#d62728', 'down': '#1f77b4', 'none': '#7f7f7f'}
    plot_colors = [colors.get(c, '#2ca02c') for c in order]

    # Counts and percentages side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    counts_map = dict(zip(summary['Class'], summary['Count']))
    pct_map = dict(zip(summary['Class'], summary['Percentage']))

    ax1.bar(order, [counts_map[c] for c in order], color=plot_colors, alpha=0.85, edgecolor='black')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Activation Direction')
    ax1.set_title(f'{title_prefix} — Counts')
    total = int(sum(counts_map.values()))
    for i, cls in enumerate(order):
        ax1.text(i, counts_map[cls] + max(1, total * 0.01), f"{counts_map[cls]:,}", ha='center', va='bottom', fontsize=10)

    ax2.bar(order, [pct_map[c] for c in order], color=plot_colors, alpha=0.85, edgecolor='black')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_xlabel('Activation Direction')
    ax2.set_title(f'{title_prefix} — Percentages')
    ax2.set_ylim(0, 100)
    for i, cls in enumerate(order):
        ax2.text(i, pct_map[cls] + 2, f"{pct_map[cls]:.1f}%", ha='center', va='bottom', fontsize=10)

    fig.tight_layout()

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'class_distribution_bars.png'
        fig.savefig(out_path, dpi=160)
        print(f"Saved: {out_path}")
    plt.show()

    # Pie Chart
    fig2, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        [counts_map[c] for c in order],
        labels=order,
        colors=plot_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12, 'weight': 'bold'},
    )
    for autotext in autotexts:
        autotext.set_color('white')
    ax.set_title(f'{title_prefix} — Pie (Imbalance)')
    fig2.tight_layout()

    if out_dir:
        out_path2 = out_dir / 'class_distribution_pie.png'
        fig2.savefig(out_path2, dpi=160)
        print(f"Saved: {out_path2}")
    plt.show()


def default_data_dir() -> Path:
    # Up one from scripts to package root, then data/raw
    return Path(__file__).resolve().parent.parent / 'data' / 'raw'


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot distribution of up/down/none activations from raw BalanceMarket CSVs (2024+2025).')
    parser.add_argument('--data-dir', type=Path, default=default_data_dir(), help='Directory containing BalanceMarket_YYYY_NO1_EUR_None_MW.csv files')
    parser.add_argument('--years', type=int, nargs='*', default=[2024, 2025], help='Years to include (default: 2024 2025)')
    parser.add_argument('--min-up', type=float, default=0.0, help='Minimum Up activation (MW) to consider non-zero (default: 0.0)')
    parser.add_argument('--min-down', type=float, default=0.0, help='Minimum Down activation (MW) to consider non-zero (default: 0.0)')
    parser.add_argument('--out-dir', type=Path, default=None, help='Optional directory to save plots and summary CSV')

    args = parser.parse_args()

    files = [args.data_dir / f'BalanceMarket_{y}_NO1_EUR_None_MW.csv' for y in args.years]
    print('Reading files:')
    for f in files:
        print(f' - {f}')

    bm = load_balance_market(files)
    labels = compute_regclass(bm, min_up=args.min_up, min_down=args.min_down)

    summary = summarize_distribution(labels)
    max_count = int(summary['Count'].max())
    min_count = int(summary['Count'].min())
    ratio = (max_count / max(1, min_count)) if min_count else float('inf')

    print('\nClass Distribution Summary:')
    print(summary.to_string(index=False))
    print(f"\nImbalance ratio (majority/minority): {ratio:.2f}:1")

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        csv_out = args.out_dir / 'class_distribution_summary.csv'
        summary.to_csv(csv_out, index=False)
        print(f'Saved: {csv_out}')

    plot_distribution(summary, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
