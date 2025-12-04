from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Columns mapping and NTCs (kept consistent with src/data/preprocess.py)
RENAME_MAP = {
    'NO1 NO1->NO2 Export (MW)': 'NO1-NO2',
    'NO1 NO2->NO1 Import (MW)': 'NO2-NO1',
    'NO1 NO1->NO3 Export (MW)': 'NO1-NO3',
    'NO1 NO3->NO1 Import (MW)': 'NO3-NO1',
    'NO1 NO1->NO5 Export (MW)': 'NO1-NO5',
    'NO1 NO5->NO1 Import (MW)': 'NO5-NO1',
    'NO1 NO1->SE3 Export (MW)': 'NO1-SE3',
    'NO1 SE3->NO1 Import (MW)': 'SE3-NO1',
}
FLOW_NTC = {
    'NO1-NO2': 2200, 'NO1-NO3': 500, 'NO1-NO5': 600, 'NO1-SE3': 2145,
    'NO2-NO1': 3500, 'NO3-NO1': 500, 'NO5-NO1': 3900, 'SE3-NO1': 2095,
}
CONNECTED_ZONES = ['NO2', 'NO3', 'NO5', 'SE3']


def load_flows(paths: list[Path]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        df = pd.read_csv(p, delimiter=';')
        # Normalize time
        if 'Delivery Start (CET)' in df.columns:
            df = df.rename(columns={'Delivery Start (CET)': 'Time'})
        elif 'Delivery Start' in df.columns:
            df = df.rename(columns={'Delivery Start': 'Time'})
        df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
        dfs.append(df)
    flow = pd.concat(dfs, ignore_index=True)
    flow = flow.dropna(subset=['Time']).sort_values('Time')
    flow = flow[~flow['Time'].duplicated(keep='first')]

    present = [c for c in RENAME_MAP.keys() if c in flow.columns]
    flow = flow[['Time'] + present].rename(columns=RENAME_MAP)

    # Ensure numeric
    for c in flow.columns:
        if c == 'Time':
            continue
        flow[c] = pd.to_numeric(flow[c], errors='coerce')

    # 15-min resample to regularize
    flow = flow.set_index('Time').sort_index()
    flow = flow.resample('15min').ffill()

    return flow


def summarize_corridor(flow: pd.DataFrame, zone: str) -> dict:
    exp_col = f'NO1-{zone}'
    imp_col = f'{zone}-NO1'

    if exp_col not in flow.columns or imp_col not in flow.columns:
        return {
            'corridor': f'NO1-{zone}',
            'present': False,
        }

    exp = flow[exp_col].astype(float)
    imp = flow[imp_col].astype(float)

    net = exp - imp  # >0 export from NO1, <0 import to NO1

    # Utilization ratios
    exp_ntc = FLOW_NTC.get(exp_col, np.nan)
    imp_ntc = FLOW_NTC.get(imp_col, np.nan)
    exp_ratio = exp / exp_ntc if exp_ntc and not np.isnan(exp_ntc) else pd.Series(np.nan, index=exp.index)
    imp_ratio = imp / imp_ntc if imp_ntc and not np.isnan(imp_ntc) else pd.Series(np.nan, index=imp.index)

    # Shares
    n = len(net)
    export_share = float((net > 0).mean()) if n else np.nan
    import_share = float((net < 0).mean()) if n else np.nan
    zero_share = float((net == 0).mean()) if n else np.nan

    # Stats helper
    def q(s: pd.Series, p: float) -> float:
        return float(s.quantile(p)) if len(s.dropna()) else np.nan

    summary = {
        'corridor': f'NO1-{zone}',
        'present': True,
        'n_points': int(n),
        'export_mw_mean': float(exp.mean()),
        'import_mw_mean': float(imp.mean()),
        'net_mw_mean': float(net.mean()),
        'net_mw_median': float(net.median()),
        'net_mw_abs_p95': q(net.abs(), 0.95),
        'export_share': export_share,
        'import_share': import_share,
        'zero_share': zero_share,
        'util_export_mean': float(exp_ratio.mean()),
        'util_import_mean': float(imp_ratio.mean()),
        'util_export_p95': q(exp_ratio, 0.95),
        'util_import_p95': q(imp_ratio, 0.95),
        'util_export_gt90_pct': float((exp_ratio > 0.9).mean()),
        'util_import_gt90_pct': float((imp_ratio > 0.9).mean()),
    }
    return summary


def plot_shares(summary_df: pd.DataFrame, out_dir: Path | None) -> None:
    # Bar chart of export/import/zero share per corridor
    df = summary_df[summary_df['present']].copy()
    corr = df['corridor'].tolist()
    x = np.arange(len(corr))
    width = 0.28

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, df['export_share'], width, label='Export share', color='#4C72B0')
    ax.bar(x, df['import_share'], width, label='Import share', color='#55A868')
    ax.bar(x + width, df['zero_share'], width, label='Zero share', color='#C44E52')
    ax.set_xticks(x)
    ax.set_xticklabels(corr, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Fraction of intervals')
    ax.set_title('Cross-zonal flow direction shares (NO1 perspective)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / 'flow_direction_shares.png'
        fig.savefig(p, dpi=160)
        print(f'Saved: {p}')
    plt.show()


def plot_utilization(summary_df: pd.DataFrame, out_dir: Path | None) -> None:
    # Boxplot of utilization ratios by corridor and direction
    df = summary_df[summary_df['present']].copy()
    labels = df['corridor'].tolist()
    exp_p95 = df['util_export_p95']
    imp_p95 = df['util_import_p95']

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, exp_p95, width, label='Export p95 utilization', color='#4C72B0')
    ax.bar(x + width/2, imp_p95, width, label='Import p95 utilization', color='#55A868')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Utilization (ratio)')
    ax.set_title('p95 directional utilization vs NTC')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / 'utilization_p95.png'
        fig.savefig(p, dpi=160)
        print(f'Saved: {p}')
    plt.show()

def plot_utilization_distribution(flow: pd.DataFrame, out_dir: Path | None) -> None:
        """Distribution plots of utilization (flow/NTC) per corridor and direction.

        Produces a grid of KDE + histogram for export and import utilization, clipped to [0, 1.2].
        """
        sns.set_theme(context='talk', style='whitegrid', font_scale=0.9)
        for zone in CONNECTED_ZONES:
            exp_col = f'NO1-{zone}'
            imp_col = f'{zone}-NO1'
            if exp_col not in flow.columns or imp_col not in flow.columns:
                continue
            exp_ntc = FLOW_NTC.get(exp_col, np.nan)
            imp_ntc = FLOW_NTC.get(imp_col, np.nan)
            exp_ratio = (flow[exp_col] / exp_ntc) if exp_ntc and not np.isnan(exp_ntc) else pd.Series(np.nan, index=flow.index)
            imp_ratio = (flow[imp_col] / imp_ntc) if imp_ntc and not np.isnan(imp_ntc) else pd.Series(np.nan, index=flow.index)

            df_plot = pd.DataFrame({
                'Export utilization': exp_ratio,
                'Import utilization': imp_ratio,
            }).dropna()

            # Clip to reasonable display range
            df_plot = df_plot.clip(lower=0, upper=1.2)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
            for ax, col, color in zip(axes, df_plot.columns, ['#4C72B0', '#55A868']):
                sns.histplot(df_plot[col], bins=40, kde=True, stat='density', ax=ax, color=color, edgecolor='white', alpha=0.85)
                ax.set_title(f'{col} — {zone}')
                ax.set_xlabel('Utilization (flow / NTC)')
                ax.set_ylabel('Density')
                ax.set_xlim(0, 1.2)
                ax.grid(alpha=0.3)

            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                p_png = out_dir / f'utilization_distribution_{zone}.png'
                p_svg = out_dir / f'utilization_distribution_{zone}.svg'
                fig.savefig(p_png, dpi=200)
                fig.savefig(p_svg)
                print(f'Saved: {p_png}\nSaved: {p_svg}')
                plt.show()

def plot_utilization_overlaid(flow: pd.DataFrame, out_dir: Path | None) -> None:
    """Single export and single import figure overlaying utilization distributions for all corridors.

    Creates two figures:
      - Export utilization (NO1->zone) layered for NO2, NO3, NO5, SE3
      - Import utilization (zone->NO1) layered for NO2, NO3, NO5, SE3
    """
    sns.set_theme(context='talk', style='whitegrid', font_scale=0.95)
    palette = {
        'NO2': '#4C72B0',
        'NO3': '#55A868',
        'NO5': '#C44E52',
        'SE3': '#8172B2',
    }

    def build_ratios(direction: str) -> dict[str, pd.Series]:
        ratios: dict[str, pd.Series] = {}
        for zone in CONNECTED_ZONES:
            col = f'NO1-{zone}' if direction == 'export' else f'{zone}-NO1'
            if col not in flow.columns:
                continue
            ntc = FLOW_NTC.get(col, np.nan)
            s = (flow[col] / ntc) if ntc and not np.isnan(ntc) else pd.Series(np.nan, index=flow.index)
            ratios[zone] = s.clip(lower=0, upper=1.2).dropna()
        return ratios

    # Export figure
    export_ratios = build_ratios('export')
    fig_e, ax_e = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for zone, s in export_ratios.items():
        sns.kdeplot(s, ax=ax_e, label=f'NO1→{zone}', color=palette.get(zone, None), lw=2)
    ax_e.set_title('Export utilization (flow/NTC) — overlaid')
    ax_e.set_xlabel('Utilization')
    ax_e.set_ylabel('Density')
    ax_e.set_xlim(0, 1.2)
    ax_e.legend(title='Corridor', ncol=2)
    ax_e.grid(alpha=0.3)
    # Cap y-axis to half of max density of NO1→NO5
    try:
        target_label = 'NO1→NO5'
        lines = [l for l in ax_e.lines if l.get_label() == target_label]
        if lines:
            y_max = max(lines[0].get_ydata())
            ax_e.set_ylim(0, y_max * 0.5)
    except Exception:
        pass
    if out_dir:
        p_png = out_dir / 'utilization_export_overlaid.png'
        p_svg = out_dir / 'utilization_export_overlaid.svg'
        fig_e.savefig(p_png, dpi=200)
        fig_e.savefig(p_svg)
        print(f'Saved: {p_png}\nSaved: {p_svg}')
    plt.show()

    # Import figure
    import_ratios = build_ratios('import')
    fig_i, ax_i = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for zone, s in import_ratios.items():
        sns.kdeplot(s, ax=ax_i, label=f'{zone}→NO1', color=palette.get(zone, None), lw=2)
    ax_i.set_title('Import utilization (flow/NTC) — overlaid')
    ax_i.set_xlabel('Utilization')
    ax_i.set_ylabel('Density')
    ax_i.set_xlim(0, 1.2)
    ax_i.legend(title='Corridor', ncol=2)
    ax_i.grid(alpha=0.3)
    # Cap y-axis to half of max density of NO2→NO1
    try:
        target_label = 'NO2→NO1'
        lines = [l for l in ax_i.lines if l.get_label() == target_label]
        if lines:
            y_max = max(lines[0].get_ydata())
            ax_i.set_ylim(0, y_max * 0.5)
    except Exception:
        pass
    if out_dir:
        p_png = out_dir / 'utilization_import_overlaid.png'
        p_svg = out_dir / 'utilization_import_overlaid.svg'
        fig_i.savefig(p_png, dpi=200)
        fig_i.savefig(p_svg)
        print(f'Saved: {p_png}\nSaved: {p_svg}')
    plt.show()


def default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / 'data' / 'raw'


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize cross-zonal flows for NO1 from Exchange_20XX raw CSVs (2024+2025).')
    parser.add_argument('--data-dir', type=Path, default=default_data_dir(), help='Directory containing Exchange_YYYY_NO1_None_MW.csv files')
    parser.add_argument('--years', type=int, nargs='*', default=[2024, 2025], help='Years to include (default: 2024 2025)')
    parser.add_argument('--out-dir', type=Path, default=None, help='Optional directory to save plots and summary CSV')

    args = parser.parse_args()

    files = [args.data_dir / f'Exchange_{y}_NO1_None_MW.csv' for y in args.years]
    print('Reading files:')
    for f in files:
        print(f' - {f}')

    flow = load_flows(files)

    # Corridor summaries
    rows = [summarize_corridor(flow, z) for z in CONNECTED_ZONES]
    summary_df = pd.DataFrame(rows)

    print('\nFlow corridor summary:')
    present_df = summary_df[summary_df['present']].copy()
    if not len(present_df):
        print('No expected corridors present in data.')
    else:
        print(present_df[['corridor','n_points','export_share','import_share','zero_share','net_mw_mean','net_mw_abs_p95','util_export_p95','util_import_p95']].to_string(index=False))

    # Save
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        csv_out = args.out_dir / 'flows_corridor_summary.csv'
        summary_df.to_csv(csv_out, index=False)
        print(f'Saved: {csv_out}')

    # Plots
    plot_shares(summary_df, args.out_dir)
    plot_utilization(summary_df, args.out_dir)
    # Utilization distribution per corridor
    plot_utilization_distribution(flow, args.out_dir)
    # Overlaid export/import utilization for all corridors
    plot_utilization_overlaid(flow, args.out_dir)


if __name__ == '__main__':
    main()
