"""
Preprocessing pipeline for mFRR activation classification (area configurable: NO1/NO2).

This module consolidates the notebook's data loading and feature engineering into
reusable Python code. It builds a time-indexed DataFrame `df` with:
- Cross-zonal flow ratios (resampled to 15min) and combined directions
- Activation target (Activated+4), lag features, and persistency
- Wind forecasts (t+4), consumption/production (t-4), imports/net imports
- Prices (day-ahead and intraday) and intraday hourly statistics (t+4)
- Selected interaction features and regime flags

Important notes about leakage:
- Any column with "+4" is a future-aligned signal relative to the row index.
  Use them only if they are truly known by the decision time, or drop them
  from training features using `build_feature_list(drop_future=True)`.
- The label typically is `Activated+4` (activation one hour ahead).

Usage:
    from preprocess import preprocess_all, build_feature_list
    df = preprocess_all(data_dir='data', include_2024=True, heavy_interactions=False, dropna=True)
    features = build_feature_list(df.columns, label='Activated+4', drop_future=True)

If run as a script, it will build `df` and save it to `preprocessed_df.csv`.
"""

# TODO: Fill NA values for intraday prices, consumption, production, etc.
from __future__ import annotations
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Tuple
from src.data.features import (
    add_temporal_and_regime_features,
    attach_wind_features,
    attach_consumption_production,
    attach_imports_and_residuals,
    attach_price_features,
    attach_intraday_hourly_features,
    add_core_derived_features,
    add_ratioized_accepted_price_features,
)

from src.utils.utils import compute_regclass_series
from src.utils.dataprocessing_utils import resample_to_15min

    
@dataclass
class Config:
    # Default to the repo's raw data directory to match PreprocessConfig
    data_dir: str = os.path.join('..','data', 'raw')
    area: str = 'NO1'
    include_2024: bool = True
    drop_future: bool = True
    heavy_interactions: bool = False
    dropna: bool = True
    horizon: int = 4
    # train/val/test fractions (must sum to <=1; remainder discarded)
    train_frac: float = 0.6
    val_frac: float = 0.2
    test_frac: float = 0.2
    activation_lag_start: int = 4
    single_persistence: bool = False
    # Optional minimum activation volumes to consider as genuine events
    min_up_volume: float | None = None
    min_down_volume: float | None = None



def _read_csv(path: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, **kwargs)

def _ensure_unique_columns(df):
    """Ensure DataFrame has unique column names by suffixing duplicates.
    Keeps first occurrence unchanged; subsequent duplicates get _dup{n} appended.
    """
    cols = list(df.columns)
    seen = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}_dup{seen[c]}")
    if new_cols != cols:
        df = df.copy()
        df.columns = new_cols
    # Defensive: drop any accidental column literally named 'dtype'
    if 'dtype' in df.columns:
        df = df.drop(columns=['dtype'])
    return df



def ensure_datetime_index(df: pd.DataFrame, col: str, fmt: str | None = None, slice_str: Tuple[int,int] | None = None) -> pd.DataFrame:
    if slice_str is not None:
        a, b = slice_str
        df[col] = df[col].astype(str).str[a:b]
    if fmt:
        df[col] = pd.to_datetime(df[col], format=fmt)
    else:
        df[col] = pd.to_datetime(df[col])
    df = df.set_index(col)
    return df


def _persistency_from_activated(activated: pd.Series) -> pd.Series:
    """Legacy helper: consecutive previous True activations shifted by 1."""
    s = activated.astype(float).fillna(0.0).astype(int)
    groups = (s == 0).cumsum()
    consec = s.groupby(groups).cumsum()
    return consec.shift(1).fillna(0).astype(int)

def _persistency_from_bool(series: pd.Series, shift_lag: int = 4) -> pd.Series:
    """Consecutive previous True values, reported at t using values strictly before t.
    The run-length counter is built per block of Trues separated by Falses and shifted by `shift_lag`.
    """
    s = series.fillna(False).astype(bool).astype(int)
    groups = (s == 0).cumsum()
    consec = s.groupby(groups).cumsum()
    return consec.shift(shift_lag).fillna(0).astype(int)


# ---------------- Segmented loaders and feature builders ----------------

def load_and_prepare_flows(data_dir: str, include_2024: bool, area: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Load cross-zonal physical flows from Exchange_20XX files, resample to 15min,
    build flow ratio features, and combine directions into single ratio features.
    Returns:
      - physical_flow_df: time-indexed flows (with both directions kept) for imports calc
      - df_flows: DataFrame with flow ratio features and combined directions (raw flows dropped)
      - connected_zones: list of zones connected to NO1
    """
    ex_2025 = os.path.join(data_dir, 'flows', f'Exchange_2025_{area}_None_MW.csv')
    ex_2024 = os.path.join(data_dir, 'flows', f'Exchange_2024_{area}_None_MW.csv')

    # Read and concatenate with 2024 if requested
    physical_flow_df = _read_csv(ex_2025, delimiter=';')
    if include_2024:
        physical_flow_df_2024 = _read_csv(ex_2024, delimiter=';')
        physical_flow_df = pd.concat([physical_flow_df_2024, physical_flow_df], ignore_index=True)

    # Parse time and set index (use Delivery Start as the interval start)
    physical_flow_df.rename(columns={"Delivery Start (CET)": "Time"}, inplace=True)
    physical_flow_df.drop(columns=[c for c in ['Delivery End (CET)'] if c in physical_flow_df.columns], inplace=True)
    physical_flow_df = ensure_datetime_index(physical_flow_df, 'Time', fmt='%d.%m.%Y %H:%M:%S')
    # Ensure monotonic increasing index prior to any resampling/ffill
    physical_flow_df = physical_flow_df[~physical_flow_df.index.duplicated(keep='first')]
    physical_flow_df.sort_index(inplace=True)

    # Dynamically map directional Export/Import columns to canonical "AREA-ZONE" / "ZONE-AREA" names
    # Example source columns: "NO1 NO1->NO2 Export (MW)", "NO1 NO2->NO1 Import (MW)"
    rename_map: dict[str, str] = {}
    for col in list(physical_flow_df.columns):
        if '->' in col and 'Export (MW)' in col:
            # e.g., "NO1 NO1->NO2 Export (MW)" -> key area->zone
            try:
                left = col.split(' Export')[0]
                # left: "NO1 NO1->NO2"
                a_to_b = left.split()[-1]
                a, b = a_to_b.split('->')
                if a == area:
                    rename_map[col] = f'{a}-{b}'
            except Exception:
                pass
        elif '->' in col and 'Import (MW)' in col:
            # e.g., "NO1 NO2->NO1 Import (MW)" -> key zone->area
            try:
                left = col.split(' Import')[0]
                a_to_b = left.split()[-1]
                a, b = a_to_b.split('->')
                if b == area:
                    rename_map[col] = f'{a}-{b}'
            except Exception:
                pass
    present_cols = [c for c in rename_map.keys() if c in physical_flow_df.columns]
    physical_flow_df = physical_flow_df[present_cols].rename(columns=rename_map)

    # Ensure numeric
    for c in physical_flow_df.columns:
        physical_flow_df[c] = pd.to_numeric(physical_flow_df[c], errors='coerce')
    # Resample to 15-min grid
    physical_flow_df = physical_flow_df.resample('15min').ffill()
    
    # Create ratios vs NTC per direction when known (NO1-only defaults); otherwise keep raw
    df_flows = physical_flow_df.copy()
    ntc_no1 = {'NO1-NO2': 2200, 'NO1-NO3': 500, 'NO1-NO5': 600, 'NO1-SE3': 2145,
               'NO2-NO1': 3500, 'NO3-NO1': 500, 'NO5-NO1': 3900, 'SE3-NO1': 2095}
    flow_ntc = ntc_no1 if area == 'NO1' else {}
    for col in list(df_flows.columns):
        if col in flow_ntc:
            df_flows[col + '_ratio'] = df_flows[col] / flow_ntc[col]

    # Identify connected zones from canonical column names
    connected_zones = sorted({c.split('-')[0] if c.endswith(f'-{area}') else c.split('-')[1]
                              for c in df_flows.columns if '-' in c and (c.startswith(area+'-') or c.endswith('-'+area))
                             if (c.split('-')[0] != c.split('-')[1])})

    # For each neighbor zone, create a single feature column for AREA-ZONE by summing
    # the two directional ratio columns if present, else sum raw MW columns.
    feature_cols_to_keep = []
    def _series_or_zeros(frame: pd.DataFrame, col: str) -> pd.Series:
        return frame[col].fillna(0) if col in frame.columns else pd.Series(0, index=frame.index)

    for zone in connected_zones:
        a_to_z = f'{area}-{zone}'
        z_to_a = f'{zone}-{area}'
        a_to_z_ratio = a_to_z + '_ratio'
        z_to_a_ratio = z_to_a + '_ratio'
        if a_to_z_ratio in df_flows.columns or z_to_a_ratio in df_flows.columns:
            df_flows[a_to_z_ratio] = _series_or_zeros(df_flows, a_to_z_ratio) + _series_or_zeros(df_flows, z_to_a_ratio)
            feature_cols_to_keep.append(a_to_z_ratio)
        else:
            # Fallback to raw MW columns
            df_flows[a_to_z] = _series_or_zeros(df_flows, a_to_z) + _series_or_zeros(df_flows, z_to_a)
            feature_cols_to_keep.append(a_to_z)
        # Drop the opposite direction and raw underlying columns to avoid duplication
        for rm in [a_to_z, z_to_a, z_to_a_ratio]:
            if rm in df_flows.columns and rm not in feature_cols_to_keep:
                df_flows.drop(columns=[rm], inplace=True)

    # Keep only the aggregated directional features
    df_flows = df_flows[feature_cols_to_keep]

    return physical_flow_df, df_flows, connected_zones


def attach_mfrr_features(df: pd.DataFrame, data_dir: str, include_2024: bool, 
                         single_persistence: bool, activation_lag_start: int,
                         cfg: Config) -> pd.DataFrame:
    area = cfg.area
    bm_2025 = os.path.join(data_dir, 'balancing', f'BalanceMarket_2025_{area}_EUR_None_MW.csv')
    bm_2024 = os.path.join(data_dir, 'balancing', f'BalanceMarket_2024_{area}_EUR_None_MW.csv')
    mfrr_df = _read_csv(bm_2025, delimiter=';')
    if include_2024:
        mfrr_df_2024 = _read_csv(bm_2024, delimiter=';')
        mfrr_df = pd.concat([mfrr_df_2024, mfrr_df], ignore_index=True)
    mfrr_df.rename(columns={"Delivery Start (CET)": "Time"}, inplace=True)
    mfrr_df = ensure_datetime_index(mfrr_df, 'Time', fmt='%d.%m.%Y %H:%M:%S')
    mfrr_df = mfrr_df[~mfrr_df.index.duplicated(keep='first')]

    # Up/down activation volumes at t
    up_col = f'{area} Activated Up Volume (MW)'
    down_col = f'{area} Activated Down Volume (MW)'
    df['Activation Volume'] = pd.to_numeric(mfrr_df[up_col], errors='coerce')
    df['Activated Down Volume'] = pd.to_numeric(mfrr_df[down_col], errors='coerce')
    df['Activated'] = df['Activation Volume'].fillna(0).gt(0)
    
    # Engineered accepted volumes and up/down prices (decision-time aligned, t-1)
    # Use ratios/skew rather than raw lag stacks
    accepted_up = pd.to_numeric(mfrr_df[f'{area} Accepted Up Volume (MW)'], errors='coerce').shift(1)
    accepted_down = pd.to_numeric(mfrr_df[f'{area} Accepted Down Volume (MW)'], errors='coerce').shift(1)
    vol_denom = (accepted_up.abs() + accepted_down.abs())
    vol_eps = float(np.nanmedian(vol_denom)) * 1e-3 if np.isfinite(vol_denom).any() else 1.0
    vol_eps = 1.0 if not np.isfinite(vol_eps) or vol_eps <= 0 else vol_eps
    df['Accepted Up Share'] = (accepted_up / (accepted_up + accepted_down + vol_eps)).clip(lower=0.0, upper=1.0)
    df['Accepted Imbalance Ratio'] = (accepted_up - accepted_down) / (accepted_up + accepted_down + vol_eps)

    price_up = pd.to_numeric(mfrr_df[f'{area} Up Price (EUR)'], errors='coerce').shift(1)
    price_down = pd.to_numeric(mfrr_df[f'{area} Down Price (EUR)'], errors='coerce').shift(1)
    price_denom = price_up.abs() + price_down.abs()
    p_eps = float(np.nanmedian(price_denom)) * 1e-3 if np.isfinite(price_denom).any() else 1.0
    p_eps = 1.0 if not np.isfinite(p_eps) or p_eps <= 0 else p_eps
    df['Up-Down Price Skew'] = 2.0 * (price_up - price_down) / (price_denom + p_eps)
    # Keep the shifted raw prices to allow DA comparison later in ratioization helper
    df['PriceUp_t-1'] = price_up
    df['PriceDown_t-1'] = price_down
    
    # Create ternary direction-at-lag features: RegLag-<k> in {-1: down, 0: none, +1: up}
    # Use volume tie-break when both sides are positive
    for lag in range(activation_lag_start, activation_lag_start + 11, 1):
        up_vol_lag = df['Activation Volume'].shift(lag).fillna(0)
        down_vol_lag = df['Activated Down Volume'].shift(lag).fillna(0)
        has_up = up_vol_lag.gt(0)
        has_down = down_vol_lag.gt(0)
        # Start as none
        reglag = pd.Series(0, index=df.index, dtype=int)
        # Only up
        reglag[has_up & ~has_down] = 1
        # Only down
        reglag[has_down & ~has_up] = -1
        # Both > 0 -> tie-break on volume (>= favors up)
        both = has_up & has_down
        if both.any():
            reglag[both] = np.where(up_vol_lag[both] >= down_vol_lag[both], 1, -1)
        df[f'RegLag-{lag}'] = reglag
        # Also provide a categorical version for experiments
        #df[f'RegLagCat-{lag}'] = reglag.map({1: 'up', -1: 'down', 0: 'none'}).astype('category')

    # Retain a single numeric up-volume lag for potential interactions/scale
    #df['Activation Volume-3'] = df['Activation Volume'].shift(3)
    
    # New unified persistency features shifted by 4:
    df['PersistenceUp'] = _persistency_from_bool(df['Activated'], shift_lag=cfg.horizon)
    df['PersistenceDown'] = _persistency_from_bool(df['Activated Down Volume'].gt(0), shift_lag=cfg.horizon)
    # None = no activation (both up and down are zero). Build boolean for none at t-1 and beyond
    none_bool = (~df['Activated']) & (~df['Activated Down Volume'].gt(0))
    df['PersistenceNone'] = _persistency_from_bool(none_bool, shift_lag=cfg.horizon)
    # Backward-compatible aliases if downstream expects old names
    df['Persistency'] = df['PersistenceUp']
    df['PersistencyDown'] = df['PersistenceDown']

    
    # Drop raw volumes after constructing features to avoid accidental leakage as features
    df.drop(columns=['Activation Volume', 'Activated Down Volume'], inplace=True)

    # Remove previous boolean lag columns (assumed to exist if generated upstream)
    for lag in (3, 4, 5, 6, 7, 8, 9):
        for col in (f'Activated-{lag}', f'ActivatedDown-{lag}'):
            df.drop(columns=[col], inplace=True, errors='ignore')
    

    
    
    # Add Target column, RegClass+4 column
    reg = compute_regclass_series(
        mfrr_df,
        up_col=up_col,
        down_col=down_col,
        tie_break='up',
        min_up_volume=cfg.min_up_volume,
        min_down_volume=cfg.min_down_volume,
    )
    # Drop column to avoid leakage
    df.drop(columns=['Activated'], errors='ignore', inplace=True)
    
    df['RegClass+4'] = reg.shift(-4)
    return df

def load_wind_forecasts(data_dir: str, include_2024: bool, area: str) -> pd.DataFrame:
    prod_dir = os.path.join(data_dir, 'production')
    # Only support new GUI_* area-specific files
    gui_2526 = os.path.join(prod_dir, f'GUI_WIND_SOLAR_GENERATION_FORECAST_ONSHORE_202501010000-202601010000_{area}.csv')
    gui_2425 = os.path.join(prod_dir, f'GUI_WIND_SOLAR_GENERATION_FORECAST_ONSHORE_202401010000-202501010000_{area}.csv')

    frames = []
    if os.path.exists(gui_2526):
        frames.append(pd.read_csv(gui_2526))
    if include_2024 and os.path.exists(gui_2425):
        frames.append(pd.read_csv(gui_2425))
    if not frames:
        raise FileNotFoundError('No GUI wind forecast files found for area ' + area)
    wind_forecast_df = pd.concat(frames, ignore_index=True)

    # Parse GUI time span and resample.
    # Column may be 'MTU (UTC)' or 'MTU (CET/CEST)' dep
    #By the way - ending on export.
    if 'MTU (UTC)' in wind_forecast_df.columns:
        wind_forecast_df.rename(columns={'MTU (UTC)': 'TimeSpan'}, inplace=True)
    elif 'MTU (CET/CEST)' in wind_forecast_df.columns:
        wind_forecast_df.rename(columns={'MTU (CET/CEST)': 'TimeSpan'}, inplace=True)
    else:
        raise KeyError('Expected MTU time span column (UTC or CET/CEST) in wind forecast file')

    raw_ts = wind_forecast_df['TimeSpan'].astype(str).str.split(' - ').str[0].str[:16]
    # Try slash format first, then dot format; combine results.
    ts_parsed = pd.to_datetime(raw_ts, format='%d/%m/%Y %H:%M', errors='coerce')
    missing_mask = ts_parsed.isna()
    if missing_mask.any():
        ts_alt = pd.to_datetime(raw_ts[missing_mask], format='%d.%m.%Y %H:%M', errors='coerce')
        ts_parsed.loc[missing_mask] = ts_alt
    if ts_parsed.isna().any():
        # Final fallback: generic parser (slower) for remaining.
        ts_generic = pd.to_datetime(raw_ts[ts_parsed.isna()], errors='coerce')
        ts_parsed.loc[ts_parsed.isna()] = ts_generic
    if ts_parsed.isna().any():
        raise ValueError('Unparsed timestamps remain in wind forecast after multi-format attempt')
    wind_forecast_df['Time'] = ts_parsed
    wind_forecast_df = wind_forecast_df.set_index('Time')
    wind_forecast_df = wind_forecast_df[~wind_forecast_df.index.duplicated(keep='first')].sort_index().resample('15min').ffill()

    # Build area-specific forecast columns consistent with features API
    mask = wind_forecast_df['Area'].astype(str).str.upper().eq(f'BZN|{area}'.upper()) if 'Area' in wind_forecast_df.columns else True
    wf = wind_forecast_df[mask]
    da_col = f'Generation - Wind Onshore [MW] Day Ahead/ BZN|{area}'
    id_col = f'Generation - Wind Onshore [MW] Intraday / BZN|{area}'
    out = pd.DataFrame(index=wind_forecast_df.index.unique())
    out[da_col] = pd.to_numeric(wf['Day-ahead (MW)'], errors='coerce')
    out[id_col] = pd.to_numeric(wf['Intraday (MW)'], errors='coerce')
    out = out.reindex(wind_forecast_df.index).ffill()
    return out

def load_consumption_production(data_dir: str, include_2024: bool, area: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    cons_2025 = os.path.join(data_dir, 'load', f'Consumption_2025_{area}_None_MW.csv')
    cons_2024 = os.path.join(data_dir, 'load', f'Consumption_2024_{area}_None_MW.csv')
    prod_2025_csv = os.path.join(data_dir, 'production', f'Production_2025_{area}_None_MW.csv')
    prod_2024 = os.path.join(data_dir, 'production', f'Production_2024_{area}_None_MW.csv')

    consumption_df = None
    production_df = None
    # Load optional consumption
    if os.path.exists(cons_2025):
        consumption_df = _read_csv(cons_2025, delimiter=';')
        if include_2024 and os.path.exists(cons_2024):
            c24 = _read_csv(cons_2024, delimiter=';')
            consumption_df = pd.concat([c24, consumption_df], ignore_index=True)
    else:
        print(f"Consumption files not found for area {area}; skipping consumption features.")

    # Load optional production
    if os.path.exists(prod_2025_csv):
        production_df = _read_csv(prod_2025_csv, delimiter=';')
        if include_2024 and os.path.exists(prod_2024):
            p24 = _read_csv(prod_2024, delimiter=';')
            production_df = pd.concat([p24, production_df], ignore_index=True)
    else:
        print(f"Production files not found for area {area}; skipping production features.")

    # Ensure DatetimeIndex
    if consumption_df is not None:
        consumption_df.rename(columns={"Delivery Start (CET)": "Time"}, inplace=True)
        if 'Delivery End (CET)' in consumption_df.columns:
            consumption_df.drop('Delivery End (CET)', axis=1, inplace=True)
        consumption_df = ensure_datetime_index(consumption_df, 'Time', fmt='%d.%m.%Y %H:%M:%S')
        consumption_df.sort_index(inplace=True)
        consumption_df.drop_duplicates(inplace=True)
        consumption_df = resample_to_15min(consumption_df)
    if production_df is not None:
        production_df.rename(columns={"Delivery Start (CET)": "Time"}, inplace=True)
        if 'Delivery End (CET)' in production_df.columns:
            production_df.drop('Delivery End (CET)', axis=1, inplace=True)
        production_df = ensure_datetime_index(production_df, 'Time', fmt='%d.%m.%Y %H:%M:%S')
        production_df.sort_index(inplace=True)
        production_df.drop_duplicates(inplace=True)
        production_df = resample_to_15min(production_df)
    return consumption_df, production_df

def load_prices(data_dir: str, include_2024: bool, area: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    da_2025 = os.path.join(data_dir, 'prices', f'AuctionPrice_2025_DayAhead_{area}_EUR_None.csv')
    da_2024 = os.path.join(data_dir, 'prices', f'AuctionPrice_2024_DayAhead_{area}_EUR_None.csv')
    id_2025 = os.path.join(data_dir, 'prices', f'AuctionPrice_2025_SIDC_IntradayAuction1_{area}_EUR_None.csv')
    id_2024 = os.path.join(data_dir, 'prices', f'AuctionPrice_2024_SIDC_IntradayAuction1_{area}_EUR_None.csv')
    id2_2025 = os.path.join(data_dir, 'prices', f'AuctionPrice_2025_SIDC_IntradayAuction2_{area}_EUR_None.csv')
    id2_2024 = os.path.join(data_dir, 'prices', f'AuctionPrice_2024_SIDC_IntradayAuction2_{area}_EUR_None.csv')
    id3_2025 = os.path.join(data_dir, 'prices', f'AuctionPrice_2025_SIDC_IntradayAuction3_{area}_EUR_None.csv')
    id3_2024 = os.path.join(data_dir, 'prices', f'AuctionPrice_2024_SIDC_IntradayAuction3_{area}_EUR_None.csv')
    da_df = _read_csv(da_2025, delimiter=';')
    if include_2024:
        da_24 = _read_csv(da_2024, delimiter=';')
        da_df = pd.concat([da_24, da_df], ignore_index=True)
    id_df = _read_csv(id_2025, delimiter=';')
    # Fallback handling for files with " (1).csv" suffix (observed for NO2 Auction2)
    if not os.path.exists(id2_2025):
        alt = id2_2025[:-4] + ' (1).csv'
        if os.path.exists(alt):
            id2_2025 = alt
    if not os.path.exists(id2_2024):
        alt = id2_2024[:-4] + ' (1).csv'
        if os.path.exists(alt):
            id2_2024 = alt

    id2_df = _read_csv(id2_2025, delimiter=';')
    id3_df = _read_csv(id3_2025, delimiter=';')
    if include_2024:
        id_24 = _read_csv(id_2024, delimiter=';')
        id_df = pd.concat([id_24, id_df], ignore_index=True)
        id2_24 = _read_csv(id2_2024, delimiter=';')
        id2_df = pd.concat([id2_24, id2_df], ignore_index=True).rename(columns={f"{area} Price (EUR)": f"{area} Price 2(EUR)"})
        id3_24 = _read_csv(id3_2024, delimiter=';')
        id3_df = pd.concat([id3_24, id3_df], ignore_index=True).rename(columns={f"{area} Price (EUR)": f"{area} Price 3(EUR)"})
    id_df = pd.concat([id_df, id2_df, id3_df], ignore_index=True)
    for frame in (da_df, id_df):
        frame['Delivery Start (CET)'] = pd.to_datetime(frame['Delivery Start (CET)'], format='%d.%m.%Y %H:%M:%S')
        frame.rename(columns={"Delivery Start (CET)": "Time"}, inplace=True)
        frame.set_index('Time', inplace=True)
        frame.sort_index(inplace=True)
        frame.drop_duplicates(inplace=True)
        # Ensure numeric price and convert empty strings to NaN before filling
        price_col = f'{area} Price (EUR)'
        if price_col in frame.columns:
            frame[price_col] = pd.to_numeric(frame[price_col].replace('', pd.NA), errors='coerce')
            
    # Resample to 15-min and fill gaps forward
    da_df = resample_to_15min(da_df)
    id_df = resample_to_15min(id_df)
    return da_df, id_df

def load_intraday_hourly_stats(data_dir: str, include_2024: bool, area: str) -> pd.DataFrame:
    idh_2025 = os.path.join(data_dir, 'prices', f'IntradayHourlyStatistics_2025_{area}_None.csv')
    idh_2024 = os.path.join(data_dir, 'prices', f'IntradayHourlyStatistics_2024_{area}_None.csv')
    id_hourly_df = _read_csv(idh_2025, delimiter=';')
    if include_2024:
        idh_24 = _read_csv(idh_2024, delimiter=';')
        id_hourly_df = pd.concat([idh_24, id_hourly_df], ignore_index=True)
    id_hourly_df.rename(columns={"Delivery Start (CET)": "Time"}, inplace=True)
    id_hourly_df['Time'] = pd.to_datetime(id_hourly_df['Time'], format='%d.%m.%Y %H:%M:%S')
    stat_cols = [f'{area} High Price (EUR/MWh)', f'{area} Low Price (EUR/MWh)',
                 f'{area} Open Price (EUR/MWh)', f'{area} Close Price (EUR/MWh)',
                 f'{area} Average Price (EUR/MWh)']
    for col in stat_cols:
        id_hourly_df[col] = pd.to_numeric(id_hourly_df[col], errors='coerce')
        id_hourly_df[col] = id_hourly_df[col].ffill().bfill()
        
    id_hourly_df.set_index('Time', inplace=True)
    id_hourly_df = resample_to_15min(id_hourly_df)
    
    # Shift by -4 to align with target interval,
    id_hourly_df = id_hourly_df.shift(-4)
    return id_hourly_df


def load_mfrr_capacity_data(data_dir: str) -> pd.DataFrame:
    """Load directional mFRR capacity (contracted reserves) from NUCS hourly CSV.

    Expects a file like
        data/raw/capacity_market/nucs_mfrr_contracted_hourly_directional.csv

    with columns:
        ts, up_price, down_price, up_quantity, down_quantity

    The data are hourly; we resample to 15-min and forward-fill, then
    shift by -4 steps (1 hour) so that at time t we see capacity for
    the next hour, consistent with other t+4 aligned features.
    """
    cap_path = os.path.join(data_dir, 'capacity_market', 'nucs_mfrr_contracted_hourly_directional.csv')
   
    cap_df = _read_csv(cap_path)

    cap_df = cap_df.rename(columns={'ts': 'Time'})
    cap_df['Time'] = pd.to_datetime(cap_df['Time'], format='%Y-%m-%d %H:%M:%S+00:00')
    cap_df.set_index('Time', inplace=True)
    cap_df.sort_index(inplace=True)
    cap_df = cap_df[~cap_df.index.duplicated(keep='first')]

    # Hourly -> 15-min, forward-fill within gaps
    cap_df = cap_df.resample('15min').ffill()

    # Align to decision time: show capacity one hour ahead at current index
    cap_df = cap_df.shift(-4)

    # Rename to model-facing feature names (assume expected directional columns exist)
    cap_df.rename(columns={
        'up_price': 'mFRR Cap Up Price',
        'down_price': 'mFRR Cap Down Price',
        'up_quantity': 'mFRR Cap Up Quantity',
        'down_quantity': 'mFRR Cap Down Quantity',
    }, inplace=True)

    # Lag features for up/down quantities at t-1 to t-9 and their average
    for lag in range(1, 10):
        cap_df[f'mFRR Cap Up Quantity Lag-{lag}'] = cap_df['mFRR Cap Up Quantity'].shift(lag)
        cap_df[f'mFRR Cap Down Quantity Lag-{lag}'] = cap_df['mFRR Cap Down Quantity'].shift(lag)
        cap_df[f'mFRR Cap Quantity Lag-{lag}'] = (
            cap_df[f'mFRR Cap Up Quantity Lag-{lag}']
            + cap_df[f'mFRR Cap Down Quantity Lag-{lag}']
        ) / 2

    return cap_df

def load_afrr_data(data_dir: str, include_2024: bool, area: str) -> pd.DataFrame:
    afrr = os.path.join(data_dir, 'afrr', 'nucs_hourly_2024_2025_updown.csv')
    afrr_df = _read_csv(afrr)
    afrr_df.rename(columns={"ts": "Time"}, inplace=True)
    # its +00:00 time format
    afrr_df['Time'] = pd.to_datetime(afrr_df['Time'], format='%Y-%m-%d %H:%M:%S+00:00')
    afrr_df.set_index('Time',  inplace=True)
    # Ensure strictly monotonic increasing index before resample
    afrr_df.sort_index(inplace=True)
    afrr_df = afrr_df[~afrr_df.index.duplicated(keep='first')].resample('15min').ffill()
    # Directional aFRR prices/volumes: up/down
    # Clean zeros and interpolate separately for each side (assume columns exist)
    for col in ['up_price', 'down_price', 'up_quantity', 'down_quantity']:
        afrr_df[col] = (
            afrr_df[col]
            .replace(0, np.nan)
            .interpolate(method='time')
            .ffill()
            .bfill()
        )

    # Align to decision time: shift future aFRR state back by horizon (4*15min)
    afrr_df = afrr_df.shift(-4)

    # Drop any legacy smoothing marker if present
    afrr_df.drop(columns=['_smoothed'], inplace=True, errors='ignore')

    # Rename to model-facing feature names (assume expected directional columns exist)
    afrr_df.rename(columns={
        'up_price': 'aFRR Up Price',
        'down_price': 'aFRR Down Price',
        'up_quantity': 'aFRR Up Quantity',
        'down_quantity': 'aFRR Down Quantity',
    }, inplace=True)
    
    
    affr_act = os.path.join(data_dir, 'afrr', f'GUI_BALANCING_OFFERS_AND_RESERVES_202401010000-202501010000_{area}.csv')
    affr_act_df = pd.read_csv(affr_act, delimiter=',')
    if include_2024:
        affr_act_24 = os.path.join(data_dir, 'afrr', f'GUI_BALANCING_OFFERS_AND_RESERVES_202501010000-202601010000_{area}.csv')
        affr_act_24_df = pd.read_csv(affr_act_24, delimiter=',')
        affr_act_df = pd.concat([affr_act_24_df, affr_act_df], ignore_index=True)
    print(f"aFRR activation data rows before processing: {len(affr_act_df)}")
    print(f'aFRR price data rows before processing: {len(afrr_df)}')
    affr_act_df.rename(columns={'ISP (UTC)': 'Time'}, inplace=True)
    affr_act_df['Time'] = pd.to_datetime(affr_act_df['Time'].astype(str).str[:16], format='%d/%m/%Y %H:%M', errors='coerce')
    # Drop rows with invalid timestamps and ensure chronological order before indexing
    affr_act_df = affr_act_df.dropna(subset=['Time']).sort_values('Time')
    affr_act_df.set_index('Time', inplace=True)
    # Ensure monotonic increasing index and remove duplicates before resampling
    affr_act_df = affr_act_df[~affr_act_df.index.duplicated(keep='first')]
    affr_act_df.sort_index(inplace=True)
    affr_act_df = affr_act_df.where(affr_act_df['Reserve Type'] == 'aFRR').dropna(subset=['Reserve Type'])

    # Correct column names use (MWh), not (MW)
    affr_act_df.rename(columns={
        'Regulation Up - Activated [17.1.E] (MWh)': 'aFRR Activated Up',
        'Regulation Down - Activated [17.1.E] (MWh)': 'aFRR Activated Down'
    }, inplace=True)

    # Select only the renamed activation columns
    affr_act_df = affr_act_df[['aFRR Activated Up', 'aFRR Activated Down']]
    affr_act_df['aFRR Activated Up'] = pd.to_numeric(affr_act_df['aFRR Activated Up'], errors='coerce')
    affr_act_df['aFRR Activated Down'] = pd.to_numeric(affr_act_df['aFRR Activated Down'], errors='coerce')
    affr_act_df = resample_to_15min(affr_act_df)

    # Build categorical lag features from t-4 to t-8 (5 features): up / down / none
    # This uses UN-SHIFTED activation volumes (past information only)
    cat_cols = {}
    for k in range(4, 9, 2):  # 4,6,8
        up_lag = affr_act_df['aFRR Activated Up'].shift(k).fillna(0)
        down_lag = affr_act_df['aFRR Activated Down'].shift(k).fillna(0)
        cat = pd.Series('none', index=affr_act_df.index, dtype='string')
        cat[(up_lag > 0) & (down_lag <= 0)] = 'up'
        cat[(down_lag > 0) & (up_lag <= 0)] = 'down'
        both = (up_lag > 0) & (down_lag > 0)
        if both.any():
            cat.loc[both] = np.where(up_lag[both] >= down_lag[both], 'up', 'down')
        cat_cols[f'aFRR_ActCat-{k}'] = cat.astype('category')
    cat_cols['aFRR_Persistency'] = _persistency_from_activated((affr_act_df['aFRR Activated Up'] > 0) |
                                                                (affr_act_df['aFRR Activated Down'] > 0))
    afrr_act_cat_df = pd.DataFrame(cat_cols, index=affr_act_df.index)
    afrr_df = afrr_df.merge(afrr_act_cat_df, how='left', left_index=True, right_index=True)
    
    return afrr_df



def preprocess_all(cfg: Config | None = None,
                   data_dir: str | None = None,
                   include_2024: bool | None = None,
                   heavy_interactions: bool | None = None,
                   dropna: bool | None = None) -> pd.DataFrame:
    """
    Build the full feature DataFrame `df` following the notebook logic.

    Parameters
    - cfg: PreprocessConfig with defaults for paths and flags
    - data_dir: override config.data_dir
    - include_2024: include 2024 files in addition to 2025
    - heavy_interactions: if True, add more interaction features (slower/more columns)
    - dropna: drop rows with NaNs introduced by shifting/alignment

    Returns
    - df: pandas DataFrame indexed by time (15-minute frequency) with engineered features and target
    """
    # Resolve config values
    if data_dir is not None:
        cfg.data_dir = data_dir
    if include_2024 is not None:
        cfg.include_2024 = include_2024
    if heavy_interactions is not None:
        cfg.heavy_interactions = heavy_interactions
    if dropna is not None:
        cfg.dropna = dropna

    ddir = cfg.data_dir
    area = cfg.area
    # Flows and ratios
    physical_flow_df, df, connected_zones = load_and_prepare_flows(ddir, cfg.include_2024, area)
    print(f"Flow index range: {physical_flow_df.index.min()} -> {physical_flow_df.index.max()} (rows={len(physical_flow_df)})")

    # mFRR activations and lags/persistency
    df = attach_mfrr_features(df, ddir, cfg.include_2024,
                              activation_lag_start=cfg.activation_lag_start,
                              single_persistence=cfg.single_persistence, cfg=cfg)
    
    # Temporal and regime features
    df = add_temporal_and_regime_features(df)

    # Wind forecasts (t+4) and derived errors
    # TODO: ADD WIND FORECAST FOR NEW 2025 DATA
    wind_forecast_df = load_wind_forecasts(ddir, cfg.include_2024, area)
    need_cols = [
        f'Generation - Wind Onshore [MW] Day Ahead/ BZN|{area}',
        f'Generation - Wind Onshore [MW] Intraday / BZN|{area}',
    ]
    # Assume required columns exist; fail fast if not
    missing = [c for c in need_cols if c not in wind_forecast_df.columns]
    if missing:
        raise KeyError(f"Missing required wind forecast columns for area {area}: {missing}")
    df = attach_wind_features(df, wind_forecast_df, area)
    
    # Consumption and production (t-4)
    consumption_df, production_df = load_consumption_production(ddir, cfg.include_2024, area)
    if consumption_df is not None and production_df is not None:
        df = attach_consumption_production(df, consumption_df, production_df, area)
    else:
        print('Skipping Consumption/Production attachment due to missing inputs for area', area)
    
    # Intraday hourly statistics (t+4)
    id_hourly_df = load_intraday_hourly_stats(ddir, cfg.include_2024, area)
    df = attach_intraday_hourly_features(df, id_hourly_df, area)
    
    # Prices (t+4)
    da_df, id_df = load_prices(ddir, cfg.include_2024, area)
    df = attach_price_features(df, da_df, id_df, area, include_id=False)
    
    affr_df = load_afrr_data(ddir, cfg.include_2024, area)
    df = df.merge(affr_df, how='left', left_index=True, right_index=True)

    # mFRR capacity market (contracted reserves, directional up/down)
    cap_df = load_mfrr_capacity_data(ddir)
    df = df.merge(cap_df, how='left', left_index=True, right_index=True)

    print(f"Final index range pre-dropna: {df.index.min()} -> {df.index.max()} (rows={len(df)})")
    
    # Core engineered features (volatility, momentum, calendar, scarcity)
    df = add_core_derived_features(df)

    # Add ratio/normalized features for accepted volumes and prices if present
    df = add_ratioized_accepted_price_features(df)
    
    # Imports, net imports, residuals and ramps
    df = attach_imports_and_residuals(df, physical_flow_df, area)
    


    # Interactions
    #df = add_interactions(df, connected_zones, cfg.heavy_interactions, area)
    
    """ print('Wind forecast shape:', wind_forecast_df.shape) """
    if 'consumption_df' in locals() and consumption_df is not None:
        print('Consumption shape:', consumption_df.shape)
    if 'production_df' in locals() and production_df is not None:
        print('Production shape:', production_df.shape)
    print('Day-ahead prices shape:', da_df.shape)
    # print('Intraday prices shape:', id_df.shape)
    print('Intraday hourly stats shape:', id_hourly_df.shape)
    print('aFRR data shape:', affr_df.shape)
    if 'cap_df' in locals():
        print('mFRR capacity data shape:', cap_df.shape)
    print('flows shape:', physical_flow_df.shape)
    

    # Ensure unique columns for downstream frameworks
    df = _ensure_unique_columns(df)

    # Final NA handling
    if cfg.dropna:
        print("Number of NaNs before dropna:", df.isna().sum().sum())
        df.dropna(inplace=True)
    print('Final preprocessed DataFrame shape:', df.shape)
    return df


def build_dataset(cfg: Config, label_name: str = 'RegClass+4') -> Tuple[pd.DataFrame, List[str]]:
    # 1) Build full preprocessed DataFrame
    df = preprocess_all(cfg)
    
    # 2) Drop rows with missing label (from shift) before splitting
    df = df[~df[label_name].isna()].copy()
    
    # 3) Create contiguous time splits
    n = len(df)
    t_end = int(n * cfg.train_frac)
    v_end = t_end + int(n * cfg.val_frac)
    # ensure bounds
    t_end = max(min(t_end, n), 0)
    v_end = max(min(v_end, n), t_end)

    train_df = df.iloc[:t_end].copy()
    val_df = df.iloc[t_end:v_end].copy()
    test_df = df.iloc[v_end:].copy() if cfg.test_frac > 0 else df.iloc[v_end:v_end].copy()
    
    # 4) Get feature list
    features = [c for c in df.columns if c != label_name]
    
    print(f"Dataset splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} (total={len(df)})")
    return df, (train_df, val_df, test_df), features