"""
Feature engineering utilities for mFRR classification.

This module contains transformations applied on already-indexed, resampled
base DataFrames. It avoids raw file IO; callers supply the aligned inputs.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def add_temporal_and_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic encodings and simple regime flags from the DatetimeIndex."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * (df.index.month - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df.index.month - 1) / 12)
    df['Working Day'] = df.index.dayofweek.isin(range(0, 5))
    df['Peak'] = df.index.hour.isin(range(8, 20)) & df['Working Day']
    return df


def attach_wind_features(df: pd.DataFrame, wind_forecast_df: pd.DataFrame, area: str = 'NO1') -> pd.DataFrame:
    """Attach wind day-ahead/intraday forecasts at t+4 and error features.

    Fills missing intraday forecast with day-ahead forecast as a fallback.
    """
    df = df.copy()
    da_col = f'Generation - Wind Onshore [MW] Day Ahead/ BZN|{area}'
    id_col = f'Generation - Wind Onshore [MW] Intraday / BZN|{area}'
    df['wind_da_t+4'] = wind_forecast_df[da_col].shift(-4)
    df['wind_id_t+4'] = wind_forecast_df[id_col].shift(-4)
    df['wind_id_t+4'] = df['wind_id_t+4'].fillna(df['wind_da_t+4'])
    df['wind_error_t+4'] = (df['wind_id_t+4'] - df['wind_da_t+4']).abs()
    df.drop(columns=['wind_da_t+4', 'wind_id_t+4'], inplace=True)
    df['wind_error_t+2'] = df['wind_error_t+4'].shift(2)
    return df


def attach_consumption_production(df: pd.DataFrame, consumption_df: pd.DataFrame, production_df: pd.DataFrame, area: str = 'NO1') -> pd.DataFrame:
    """Attach consumption/production signals aligned to decision time (t-4).

    Align consumption/production to the modeling index first to avoid large NaN runs,
    then shift by +4 so value at time t reflects information available at t-1h.
    """

    df = df.copy()

    # Reindex to df.index to ensure label alignment before shifting
    cons = consumption_df[f'{area} Volume (MW)'].reindex(df.index).ffill().shift(4)
    wind_prod = production_df[f'{area} Wind Onshore (MW)'].reindex(df.index).ffill().shift(4)
    total_prod = production_df[f'{area} Total (MW)'].reindex(df.index).ffill().shift(4)

    df['Consumption'] = cons
    df['Wind Production'] = wind_prod
    df['Total Production'] = total_prod

    # Safe division for Wind Share
    denom = total_prod.replace(0, np.nan)
    df['Wind Share'] = (wind_prod / denom).clip(lower=0, upper=1)

    # Residual balance proxy
    df['Consumption / Production'] = df['Consumption'] / df['Total Production']
    
    #df.drop(columns=['Consumption', 'Wind Production'], inplace=True)

    return df


def attach_imports_and_residuals(df: pd.DataFrame, physical_flow_df: pd.DataFrame, area: str = 'NO1') -> pd.DataFrame:
    """Attach derived imports, residual load, and ramp features."""
    df = df.copy()
    inbound_cols = [c for c in physical_flow_df.columns if c.endswith(f'-{area}')]
    outbound_cols = [c for c in physical_flow_df.columns if c.startswith(f'{area}-')]
    sum_import = physical_flow_df[inbound_cols].sum(axis=1) if inbound_cols else pd.Series(0, index=physical_flow_df.index)
    df['Total Imports'] = sum_import.shift(4)
    sum_export = physical_flow_df[outbound_cols].sum(axis=1) if outbound_cols else pd.Series(0, index=physical_flow_df.index)
    net_import = sum_import - sum_export
    df['Net Import'] = net_import.shift(4)
    if 'Consumption' in df.columns:
        df['Import/Consumption'] = df['Net Import'] / df['Consumption']
    if 'Consumption' in df.columns and 'Wind Production' in df.columns:
        df['Residual Load'] = df['Consumption'] - df['Wind Production']
        df['Residual Load Delta'] = df['Residual Load'].diff()
    else:
        # Provide neutral defaults if consumption/production missing
        if 'Residual Load' not in df.columns:
            df['Residual Load'] = np.nan
        df['Residual Load Delta'] = df['Residual Load'].diff()
    df['Residual Load Delta'] = df['Residual Load'].diff()
    df['Net Import Ramp'] = df['Net Import'].diff()
    return df


def attach_price_features(df: pd.DataFrame, da_df: pd.DataFrame, id_df: pd.DataFrame, area: str = 'NO1', include_id: bool = False) -> pd.DataFrame:
    """Attach day-ahead and intraday prices at t+4 and their spread."""
    df = df.copy()
    price_col = f'{area} Price (EUR)'
    price2_col = f'{area} Price 2(EUR)'
    price3_col = f'{area} Price 3(EUR)'
    df['DA Price'] = da_df[price_col].shift(-4)
    df['ID Price'] = id_df.get(price_col, pd.Series(index=id_df.index, dtype=float)).shift(-4)
    # Some 2024 merges rename auction 2/3; if absent, we create from generic average
    df['ID Price 2'] = id_df.get(price2_col, id_df.get(price_col)).shift(-4)
    df['ID Price 3'] = id_df.get(price3_col, id_df.get(price_col)).shift(-4)
    
    #df['ID Price'] = df['ID Price'].fillna(df['ID Average Price'])
    #df['ID Price 2'] = df['ID Price 2'].fillna(df['ID Average Price'])
    df['ID Price 3'] = df['ID Price 3'].fillna(df['ID Average Price'])
    
    # ID Price 1-3 are filled between 01.01.2024 to 15.06.2024.
    # We make a boolean feature to indicate whether or not a row has real or filled ID prices.
    #df['ID Price Filled'] = id_df['NO1 Price (EUR)'].shift(-4).isna()
    
    # Robust price difference and ratio-like features (avoid blow-ups when DA~0)
    price_da = df['DA Price']
    price_id = df['ID Price 3']

    # 1) Plain difference (stable, sign-aware)
    #df['Price Diff'] = price_id - price_da

    # 2) Symmetric relative difference in [-2, 2] (bounded, scale-free)
    #    2*(x-y)/(abs(x)+abs(y)+eps)
    eps = float(np.nanmedian(price_da.abs())) * 0.01 if np.isfinite(price_da.abs()).any() else 1.0
    if not np.isfinite(eps) or eps <= 0:
        eps = 1.0
    df['Price Symm Rel Diff'] = 2.0 * (price_id - price_da) / (price_id.abs() + price_da.abs() + eps)

    # 3) Safe ratio (clipped) – keep legacy column name for compatibility
    #    Use dynamic epsilon to avoid extreme values when DA ~ 0
    denom = price_da.copy()
    small_mask = denom.abs() < eps
    # preserve sign where possible; if zero, use +eps
    denom = np.where(small_mask, np.sign(denom.where(denom != 0, 1.0)) * eps, denom)
    safe_ratio = price_id / denom
    df['DA/ID Price Ratio'] = np.clip(safe_ratio, -10.0, 10.0)

    drop_cols = ['ID Average Price', 'ID Open Price', 'ID High Price', 'ID Low Price', 'ID Close Price', 'ID Price 2', 'ID Price']
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    return df


def attach_intraday_hourly_features(df: pd.DataFrame, id_hourly_df: pd.DataFrame, area: str = 'NO1') -> pd.DataFrame:
    """Attach intraday hourly stats at t+4 and range features."""
    df = df.copy()
    rename_stats = {
        f'{area} High Price (EUR/MWh)': 'ID High Price',
        f'{area} Low Price (EUR/MWh)': 'ID Low Price',
        f'{area} Open Price (EUR/MWh)': 'ID Open Price',
        f'{area} Close Price (EUR/MWh)': 'ID Close Price',
        f'{area} Average Price (EUR/MWh)': 'ID Average Price',
    }
    for src, dst in rename_stats.items():
        if src in id_hourly_df.columns:
            df[dst] = id_hourly_df[src]
    #df['ID Price Range'] = df['ID High Price'] - df['ID Low Price']
    return df


def add_interactions(df: pd.DataFrame, connected_zones: List[str], heavy_interactions: bool = False, area: str = 'NO1') -> pd.DataFrame:
    """Create interaction features among key drivers and regimes.

    If heavy_interactions is True, add broader pairwise products among selected stats.
    """
    df = df.copy()
    # Guard against duplicate column labels which can break arithmetic alignment
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    if 'Price Diff' in df.columns:
        if 'Wind Share' in df.columns:
            df['Wind Share x Price Diff'] = df['Wind Share'] * df['Price Diff']
        if 'Residual Load' in df.columns:
            df['Residual Load x Price Diff'] = df['Residual Load'] * df['Price Diff']
            df['Residual Load Delta x Price Diff'] = df['Residual Load Delta'] * df['Price Diff']
        if 'Import/Consumption' in df.columns:
            df['Import/Consumption x Price Diff'] = df['Import/Consumption'] * df['Price Diff']
        if 'Persistency' in df.columns:
            df['Persistency x Price Diff'] = df['Persistency'] * df['Price Diff']
        if 'PersistencyDown' in df.columns:
            df['PersistencyDown x Price Diff'] = df['PersistencyDown'] * df['Price Diff']
        if 'ID Open Price' in df.columns:
            df['ID Open Price x Price Diff'] = df['ID Open Price'] * df['Price Diff']
        if 'ID Price Range' in df.columns:
            df['ID Price Range x Price Diff'] = df['ID Price Range'] * df['Price Diff']
        if 'DA Price' in df.columns and 'ID Open Price' in df.columns:
            df['ID Open Price - Day Ahead Price'] = df['ID Open Price'] - df['DA Price']

    for zone in connected_zones:
        ratio_col = f'{area}-{zone}_ratio'
        raw_col = f'{area}-{zone}'
        use_col = ratio_col if ratio_col in df.columns else (raw_col if raw_col in df.columns else None)
        if use_col is None:
            continue
        if use_col in df.columns:
            if 'Price Diff' in df.columns:
                df[f'{use_col} x Price Diff'] = df[use_col] * df['Price Diff']
            if 'Residual Load' in df.columns:
                df[f'{use_col} x Residual Load'] = df[use_col] * df['Residual Load']
            if 'Wind Share' in df.columns:
                df[f'{use_col} x Wind Share'] = df[use_col] * df['Wind Share']
            if 'Persistency' in df.columns:
                df[f'{use_col} x Persistency'] = df[use_col] * df['Persistency']
            if 'PersistencyDown' in df.columns:
                df[f'{use_col} x PersistencyDown'] = df[use_col] * df['PersistencyDown']

    if 'Net Import' in df.columns and 'Price Diff' in df.columns:
        df['Cross Border Stress'] = df['Net Import'] * df['Price Diff']
    
    # Persistency-centric interactions (robust, with clear economic rationale)
    # - Persistency x Residual Load: streak effects tend to amplify under tight system balance
    if 'Persistency' in df.columns and 'Residual Load' in df.columns:
        df['Persistency x Residual Load'] = df['Persistency'] * df['Residual Load']
    if 'PersistencyDown' in df.columns and 'Residual Load' in df.columns:
        df['PersistencyDown x Residual Load'] = df['PersistencyDown'] * df['Residual Load']

    # - Persistency x Import/Consumption: import dependence conditions the impact of recent streaks
    if 'Persistency' in df.columns and 'Import/Consumption' in df.columns:
        df['Persistency x Import/Consumption'] = df['Persistency'] * df['Import/Consumption']
    if 'PersistencyDown' in df.columns and 'Import/Consumption' in df.columns:
        df['PersistencyDown x Import/Consumption'] = df['PersistencyDown'] * df['Import/Consumption']

    # - Persistency x Peak and Persistency x DA Scarcity: regime-strengthened streak effects
    if 'Persistency' in df.columns and 'Peak' in df.columns:
        df['Persistency x Peak'] = (df['Persistency'] * df['Peak'].astype(int))
    if 'PersistencyDown' in df.columns and 'Peak' in df.columns:
        df['PersistencyDown x Peak'] = (df['PersistencyDown'] * df['Peak'].astype(int))
    if 'Persistency' in df.columns and 'DA Scarcity' in df.columns:
        df['Persistency x DA Scarcity'] = (df['Persistency'] * df['DA Scarcity'].astype(int))
    if 'PersistencyDown' in df.columns and 'DA Scarcity' in df.columns:
        df['PersistencyDown x DA Scarcity'] = (df['PersistencyDown'] * df['DA Scarcity'].astype(int))

    # - Persistency x aFRR context: streaks in mFRR often co-occur with elevated aFRR stress
    # If directional aFRR prices are available, build separate up/down interactions;
    # fall back to legacy aggregate price if present.
    if 'Persistency' in df.columns and 'aFRR Up Price' in df.columns:
        df['Persistency x aFRR Up Price'] = df['Persistency'] * df['aFRR Up Price']
    if 'PersistencyDown' in df.columns and 'aFRR Down Price' in df.columns:
        df['PersistencyDown x aFRR Down Price'] = df['PersistencyDown'] * df['aFRR Down Price']
    if 'Persistency' in df.columns and 'aFRR Price' in df.columns:
        df['Persistency x aFRR Price'] = df['Persistency'] * df['aFRR Price']
    if 'PersistencyDown' in df.columns and 'aFRR Price' in df.columns:
        df['PersistencyDown x aFRR Price'] = df['PersistencyDown'] * df['aFRR Price']
    if 'Persistency' in df.columns and 'aFRR Scarcity' in df.columns:
        df['Persistency x aFRR Scarcity'] = (df['Persistency'] * df['aFRR Scarcity'].astype(int))
    if 'PersistencyDown' in df.columns and 'aFRR Scarcity' in df.columns:
        df['PersistencyDown x aFRR Scarcity'] = (df['PersistencyDown'] * df['aFRR Scarcity'].astype(int))
        
    if 'ID Price 3' in df.columns and 'DA Price' in df.columns:
        df['ID Price 3 - Day Ahead Price'] = df['ID Price 3'] - df['DA Price']
    
    # Safe categorical * numeric interactions for aFRR activation category vs mFRR RegLag direction.
    # Map categorical aFRR activation ('up','down','none') to numeric (+1,-1,0) before combining.
    cat_map = {'up': 1, 'down': -1, 'none': 0}
    # Convert any available aFRR_ActCat-* lags (e.g., 4..20) into numeric direction columns.
    lags = []
    for c in df.columns:
        if isinstance(c, str) and c.startswith('aFRR_ActCat-'):
            suffix = c.rsplit('-', 1)[-1]
            if suffix.isdigit():
                lags.append(int(suffix))
    for lag in sorted(set(lags)):
        cat_col = f'aFRR_ActCat-{lag}'
        if cat_col in df.columns:
            num_col = f'aFRR_ActDirNum-{lag}'
            df[num_col] = df[cat_col].astype(str).str.lower().map(cat_map).fillna(0).astype('Int8')
    # Create interaction features for a subset of lags (example: -4 and -6); extend easily if needed.
    if 'RegLag-4' in df.columns and 'aFRR_ActDirNum-4' in df.columns:
        df['aFRR Activation x mFRR Activation -4'] = df['RegLag-4'] * df['aFRR_ActDirNum-4']
    if 'RegLag-6' in df.columns and 'aFRR_ActDirNum-6' in df.columns:
        df['aFRR Activation x mFRR Activation -6'] = df['RegLag-6'] * df['aFRR_ActDirNum-6']

    if heavy_interactions:
        stats_simple = ['ID High Price', 'ID Low Price', 'ID Open Price', 'ID Close Price', 'ID Average Price']
        present = [s for s in stats_simple if s in df.columns]
        for i, s1 in enumerate(present):
            for s2 in present[i + 1: i + 4]:
                df[f'{s1} x {s2}'] = df[s1] * df[s2]
    return df


def add_core_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a compact set of robust, decision-time-safe features.

    Only uses past and present information relative to index time t.
    Assumes df has 15-minute frequency and already contains:
      - DA Price, ID Price 3, Price Diff (ID3 - DA), ID Average Price
      - hour/day regime columns may exist but are not required

    Features added (examples):
      - Rolling volatility (1h) of ID Average Price and Price Diff
      - Momentum (1h) of Price Diff and ID Price 3
      - Calendar ints (HOD, DOW) alongside sinusoidal encodings
      - Scarcity flag: DA Price above rolling 90th percentile over past 1w
    """
    df = df.copy()

    # Helper: safe rolling with min periods
    def _roll(series: pd.Series, window: int, func: str):
        if func == 'std':
            return series.rolling(window=window, min_periods=max(2, window // 2)).std()
        if func == 'mean':
            return series.rolling(window=window, min_periods=max(2, window // 2)).mean()
        if func == 'q90':
            return series.rolling(window=window, min_periods=max(10, window // 4)).quantile(0.9)
        return series

    # Window sizes in 15-min steps
    W_1H = 4
    W_1D = 96
    W_1W = 96 * 7

    # Volatility features (use only past info)
    if 'ID Average Price' in df.columns:
        df['ID Avg Vol_1h'] = _roll(df['ID Average Price'], W_1H, 'std')

    if 'Price Diff' in df.columns:
        df['PriceDiff Vol_1h'] = _roll(df['Price Diff'], W_1H, 'std')
        df['PriceDiff Mom_1h'] = df['Price Diff'] - df['Price Diff'].shift(4)

    if 'ID Price 3' in df.columns:
        df['ID3 Mom_1h'] = df['ID Price 3'] - df['ID Price 3'].shift(4)

    # Calendar integers (complement existing cyclical encodings)
    idx = pd.to_datetime(df.index)
    df['HOD'] = idx.hour.astype(int)
    df['DOW'] = idx.dayofweek.astype(int)

    # Scarcity flag: DA Price above its rolling 90th percentile in last week
    if 'DA Price' in df.columns:
        q90 = _roll(df['DA Price'], W_1W, 'q90')
        df['DA Scarcity'] = (df['DA Price'] > q90).astype('Int8')
        
    # Replace inf with NaN to be handled by downstream dropna
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def add_ratioized_accepted_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio/normalized variants for accepted volumes and prices if present.

    Looks for columns matching acceptedvolup/acceptedvoldown/priceup/pricedown
    in a case- and separator-insensitive manner. Generates:
      - Accepted Up Share: up/(up+down)
      - Accepted Imbalance Ratio: (up-down)/(up+down)
      - PriceUp - DA, PriceDown - DA (if DA Price present)
      - PriceUp/DA Ratio, PriceDown/DA Ratio (safe denom)
      - Up-Down Price Skew: 2*(priceUp-priceDown)/(abs(priceUp)+abs(priceDown)+eps)
    """
    df = df.copy()

    # Build a lookup for flexible matching
    def match_col(name_parts):
        target = ''.join(name_parts).lower()
        for c in df.columns:
            key = c.lower().replace(' ', '').replace('_', '')
            if target in key:
                return c
        return None

    up_vol_col = match_col(['accepted', 'vol', 'up']) or match_col(['activated', 'up', 'volume'])
    down_vol_col = match_col(['accepted', 'vol', 'down']) or match_col(['activated', 'down', 'volume'])
    up_price_col = match_col(['price', 'up']) or match_col(['accepted', 'up', 'price'])
    down_price_col = match_col(['price', 'down']) or match_col(['accepted', 'down', 'price'])

    # Attempt canonical BM column names as a fallback
    if up_vol_col is None:
        for c in df.columns:
            if 'activated up volume' in c.lower() or 'accepted up volume' in c.lower():
                up_vol_col = c
                break
    if down_vol_col is None:
        for c in df.columns:
            if 'activated down volume' in c.lower() or 'accepted down volume' in c.lower():
                down_vol_col = c
                break

    # Volumes: shares and imbalance
    if up_vol_col is not None and down_vol_col is not None:
        u = pd.to_numeric(df[up_vol_col], errors='coerce')
        d = pd.to_numeric(df[down_vol_col], errors='coerce')
        denom = (u + d).abs()
        eps = float(np.nanmedian(denom)) * 1e-3 if np.isfinite(denom).any() else 1.0
        eps = 1.0 if not np.isfinite(eps) or eps <= 0 else eps
        df['Accepted Up Share'] = (u / (u + d + eps)).clip(lower=0.0, upper=1.0)
        df['Accepted Imbalance Ratio'] = (u - d) / (u + d + eps)

    # Prices: compare to DA and price skew
    if up_price_col is not None or down_price_col is not None:
        da = df.get('DA Price', None)
        # Safe denom for DA ratio
        if da is not None:
            da = pd.to_numeric(da, errors='coerce')
            da_eps = float(np.nanmedian(da.abs())) * 0.01 if np.isfinite(da).any() else 1.0
            da_eps = 1.0 if not np.isfinite(da_eps) or da_eps <= 0 else da_eps
            da_denom = da.copy()
            small = da_denom.abs() < da_eps
            da_denom = np.where(small, np.sign(da_denom.where(da_denom != 0, 1.0)) * da_eps, da_denom)
        if up_price_col is not None:
            pu = pd.to_numeric(df[up_price_col], errors='coerce')
            if da is not None:
                df['PriceUp - DA'] = pu - da
                df['PriceUp/DA Ratio'] = np.clip(pu / da_denom, -10.0, 10.0)
        if down_price_col is not None:
            pdn = pd.to_numeric(df[down_price_col], errors='coerce')
            if da is not None:
                df['PriceDown - DA'] = pdn - da
                df['PriceDown/DA Ratio'] = np.clip(pdn / da_denom, -10.0, 10.0)
        # Up-Down skew
        if up_price_col is not None and down_price_col is not None:
            pu = pd.to_numeric(df[up_price_col], errors='coerce')
            pdn = pd.to_numeric(df[down_price_col], errors='coerce')
            denom = pu.abs() + pdn.abs()
            eps = float(np.nanmedian(denom)) * 1e-3 if np.isfinite(denom).any() else 1.0
            eps = 1.0 if not np.isfinite(eps) or eps <= 0 else eps
            df['Up-Down Price Skew'] = 2.0 * (pu - pdn) / (denom + eps)

    # Clean infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df
