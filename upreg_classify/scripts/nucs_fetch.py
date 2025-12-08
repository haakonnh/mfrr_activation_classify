import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import xml.etree.ElementTree as ET

# -----------------------------
# Utilities
# -----------------------------

ISO_YMDHM = "%Y%m%d%H%M"
_DUR_RE = re.compile(r"^P(T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?)$", re.I)


def parse_iso_duration(text: str) -> timedelta:
    """Parse ISO-8601 duration like PT60M/PT1H. Fallback to seconds if plain int.
    Returns timedelta(hours=1) when missing/invalid.
    """
    if not text:
        return timedelta(hours=1)
    m = _DUR_RE.match(str(text).strip())
    if not m:
        try:
            return timedelta(seconds=int(text))
        except Exception:
            return timedelta(hours=1)
    h = int(m.group("h") or 0)
    m_ = int(m.group("m") or 0)
    s = int(m.group("s") or 0)
    return timedelta(hours=h) + timedelta(minutes=m_) + timedelta(seconds=s)


# -----------------------------
# Period chunking
# -----------------------------

def build_periods(period_start: str, period_end: str, chunk_days: int = 50) -> List[Dict[str, str]]:
    """Build [periodStart, periodEnd] chunks in YYYYMMDDHHMM spanning [start, end]."""
    start_dt = datetime.strptime(period_start, ISO_YMDHM)
    end_dt = datetime.strptime(period_end, ISO_YMDHM)
    periods: List[Dict[str, str]] = []
    cur = start_dt
    while cur < end_dt:
        next_end = min(cur + timedelta(days=int(chunk_days)), end_dt)
        periods.append({
            "periodStart": cur.strftime(ISO_YMDHM),
            "periodEnd": next_end.strftime(ISO_YMDHM),
        })
        cur = next_end
    return periods


# -----------------------------
# NUCS fetch + parse
# -----------------------------

@dataclass
class NucsConfig:
    base_url: str
    token: str
    document_type: str = "A81"
    process_type: str = "A51"
    market_agreement_type: str = "A01"
    control_area: str = "10YNO-1--------2"
    # For maximum transfer capacity documents (A26) we need in/out domain codes
    in_domain: Optional[str] = None
    out_domain: Optional[str] = None


def build_params(template: Dict[str, Any], window: Dict[str, str]) -> Dict[str, Any]:
    p = dict(template)
    p.update(window)
    return p


def fetch_nucs_xml(base_url: str, token: str, params: Dict[str, Any], timeout: int = 60) -> bytes:
    if not base_url or "<" in base_url or ">" in base_url:
        raise ValueError("base_url looks like a placeholder. Provide a real host.")
    if not token or "<" in token or ">" in token:
        raise ValueError("securityToken looks like a placeholder; pass the raw token.")

    url = base_url.rstrip("/") + "/api"
    q = {"securityToken": token}
    q.update(params or {})
    r = requests.get(url, params=q, timeout=timeout)
    print("NUCS request URL:", r.url)
    r.raise_for_status()
    return r.content


# namespace-agnostic local tag
ltag = lambda e: e.tag.split('}')[-1] if isinstance(e.tag, str) else ''  # noqa: E731


def parse_nucs_points(xml_bytes: bytes) -> List[Dict[str, Any]]:
    """Parse NUCS XML (namespace-agnostic) into point-level records with timestamps.
    Returns list of dicts: {ts, procurement_price, quantity, ...meta}
    """
    root = ET.fromstring(xml_bytes)
    points: List[Dict[str, Any]] = []

    series_list = [el for el in root.iter() if ltag(el) == 'TimeSeries']
    for ts in series_list:
        # period and time interval
        period = next((el for el in ts.iter() if ltag(el) == 'Period'), None)
        if period is None:
            continue
        ti = next((el for el in period.iter() if ltag(el) == 'timeInterval'), None)
        start_dt: Optional[pd.Timestamp] = None
        if ti is not None:
            s_el = next((ch for ch in ti if ltag(ch) == 'start'), None)
            if s_el is not None and s_el.text:
                try:
                    start_dt = pd.to_datetime(s_el.text)
                except Exception:
                    start_dt = None
            if start_dt is None and ti.text and '/' in ti.text:
                a, _ = ti.text.split('/', 1)
                try:
                    start_dt = pd.to_datetime(a)
                except Exception:
                    start_dt = None
        if start_dt is None:
            continue

        # resolution
        res_el = next((el for el in period.iter() if ltag(el) == 'resolution' and el.text), None)
        delta = parse_iso_duration(res_el.text if res_el is not None else 'PT60M')

        # optional meta
        meta: Dict[str, Any] = {}
        for tag in ("businessType", "currency_Unit.name", "quantity_Measure_Unit.name", "price_Measure_Unit.name", "flowDirection.direction"):
            el = next((x for x in ts.iter() if ltag(x) == tag and x.text), None)
            if el is not None and el.text:
                meta[tag] = el.text

        for pt in [el for el in period.iter() if ltag(el) == 'Point']:
            pos_el = next((ch for ch in pt if ltag(ch) == 'position' and ch.text), None)
            if pos_el is None:
                continue
            try:
                position = int(pos_el.text)
            except Exception:
                continue

            # price: any descendant with 'price' in the local tag
            price_val: Optional[float] = None
            for el in pt.iter():
                name = ltag(el).lower()
                if 'price' in name and el.text:
                    try:
                        price_val = float(el.text)
                        break
                    except Exception:
                        pass

            qty_val: Optional[float] = None
            q_el = next((el for el in pt.iter() if ltag(el) == 'quantity' and el.text), None)
            if q_el is not None:
                try:
                    qty_val = float(q_el.text)
                except Exception:
                    qty_val = None

            ts_point = pd.to_datetime(start_dt) + (position - 1) * pd.to_timedelta(delta)
            points.append({
                'ts': ts_point,
                'procurement_price': price_val,
                'quantity': qty_val,
                **meta,
            })

    return points


def points_to_hourly(df_points: pd.DataFrame) -> pd.DataFrame:
    """Aggregate NUCS point-level data to hourly up/down volumes and prices.

    The NUCS XML contains two records per timeslot distinguished by
    ``flowDirection.direction``:

    - ``A01``: up
    - ``A02``: down

    This function pivots those into separate hourly columns:

    - ``up_price``, ``up_quantity``
    - ``down_price``, ``down_quantity``
    """
    if df_points.empty:
        return df_points

    df = df_points.copy()
    # Normalize direction labels
    if 'flowDirection.direction' in df.columns:
        df['direction'] = df['flowDirection.direction'].astype(str).str.upper()
    else:
        df['direction'] = None

    # Split into up / down subsets
    up_mask = df['direction'] == 'A01'
    down_mask = df['direction'] == 'A02'

    hourly_up = (
        df[up_mask]
        .dropna(subset=['ts'])
        .set_index('ts')
        .sort_index()
        .resample('1h')
        .agg(
            up_price=('procurement_price', 'mean'),
            up_quantity=('quantity', 'sum'),
        )
    )

    hourly_down = (
        df[down_mask]
        .dropna(subset=['ts'])
        .set_index('ts')
        .sort_index()
        .resample('1h')
        .agg(
            down_price=('procurement_price', 'mean'),
            down_quantity=('quantity', 'sum'),
        )
    )

    # Outer-join to keep any hour that appears in either direction
    hourly = (
        hourly_up.join(hourly_down, how='outer')
        .reset_index()
        .rename(columns={'ts': 'ts'})
        .sort_values('ts')
    )

    return hourly


def plot_outputs(hourly: pd.DataFrame, plots_dir: str, stem: str = 'nucs_hourly') -> Tuple[str, str, str]:
    os.makedirs(plots_dir, exist_ok=True)

    # line: plot up and down prices if available, else fall back
    plt.figure(figsize=(12, 4))
    if {'up_price', 'down_price'}.issubset(hourly.columns):
        sns.lineplot(data=hourly, x='ts', y='up_price', linewidth=0.8, label='up_price')
        sns.lineplot(data=hourly, x='ts', y='down_price', linewidth=0.8, label='down_price')
        plt.ylabel('Price (up/down)')
    elif 'up_price' in hourly.columns:
        sns.lineplot(data=hourly, x='ts', y='up_price', linewidth=0.8, label='up_price')
        plt.ylabel('Price')
    elif 'procurement_price' in hourly.columns:
        sns.lineplot(data=hourly, x='ts', y='procurement_price', linewidth=0.8, label='price')
        plt.ylabel('Price')
    else:
        sns.lineplot(data=hourly, x='ts', y=hourly.columns[1], linewidth=0.8)
        plt.ylabel('Value')
    plt.title('NUCS aFRR hourly prices')
    plt.xlabel('Time')
    plt.tight_layout()
    p1 = os.path.join(plots_dir, f'{stem}.png')
    plt.savefig(p1, dpi=150)

    # histogram of prices: prefer up_price, then procurement_price, else first numeric
    price_series = None
    if 'up_price' in hourly.columns:
        price_series = hourly['up_price']
    elif 'procurement_price' in hourly.columns:
        price_series = hourly['procurement_price']
    else:
        num_cols = hourly.select_dtypes(include='number').columns
        if len(num_cols) > 0:
            price_series = hourly[num_cols[0]]

    p2 = os.path.join(plots_dir, f'{stem}_hist.png')
    if price_series is not None:
        plt.figure(figsize=(9, 4))
        sns.histplot(price_series.dropna(), bins=50)
        plt.title('NUCS aFRR hourly price distribution')
        plt.xlabel('Price')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(p2, dpi=150)

    # box by hour-of-day
    h = hourly.copy()
    h['_hour'] = pd.to_datetime(h['ts']).dt.hour
    price_col = None
    if 'up_price' in h.columns:
        price_col = 'up_price'
    elif 'procurement_price' in h.columns:
        price_col = 'procurement_price'
    else:
        num_cols = [c for c in h.columns if c not in ['ts', '_hour'] and pd.api.types.is_numeric_dtype(h[c])]
        if num_cols:
            price_col = num_cols[0]

    if price_col is not None:
        plt.figure(figsize=(12, 4))
        sns.boxplot(data=h, x='_hour', y=price_col)
        plt.title('NUCS hourly price by hour-of-day')
        plt.xlabel('Hour of day')
        plt.ylabel('Price')
        plt.tight_layout()
        p3 = os.path.join(plots_dir, f'{stem}_by_hour.png')
        plt.savefig(p3, dpi=150)

    return p1, p2, p3


def smooth_outliers(
    hourly_df: pd.DataFrame,
    price_col: str = "up_price",
    max_price: Optional[float] = None,
    iqr_factor: float = 4.0,
) -> pd.DataFrame:
    """Replace extreme price spikes with neighbor-interpolated values.

    Strategy:
    - If max_price is provided, clip outliers above max_price by interpolation.
    - Else compute an upper bound using IQR: Q3 + iqr_factor * IQR (robust).
    - We don't forward-fill long gaps; only individual/burst spikes get smoothed.
    """
    if hourly_df is None or hourly_df.empty or price_col not in hourly_df.columns:
        return hourly_df

    df = hourly_df.copy()
    s = df[price_col].astype(float)
    if max_price is None:
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        upper = q3 + iqr_factor * iqr
    else:
        upper = float(max_price)

    mask = s > upper
    if not mask.any():
        return df

    # interpolate only where masked, using nearby neighbors
    s2 = s.copy()
    s2[mask] = pd.NA
    s2 = s2.interpolate(method='linear', limit_direction='both')
    # If still NaN (e.g., at boundaries), backfill/forward-fill minimally
    s2 = s2.bfill().ffill()
    df[price_col] = s2
    df["_smoothed"] = mask.astype(int)
    return df


def fetch_and_parse(config: NucsConfig, period_start: str, period_end: str, chunk_days: int = 50) -> pd.DataFrame:
    """Fetch NUCS over chunked periods and return point-level DataFrame.

    Graceful handling:
    - If the overall period_start is in the future (later than now), no request is made and
      an empty DataFrame is returned with a note.
    - For individual windows, HTTP 204/400/404 are treated as "no data for this period" and
      safely skipped.
    """

    template = {
        "documentType": config.document_type,
        "processType": config.process_type,
        "type_marketagreement.type": config.market_agreement_type,
        "controlArea_domain": config.control_area,
    }
    # Include in/out domains if provided (for A26 maximum transfer capacity requests)
    if config.in_domain:
        template["In_Domain"] = config.in_domain
    if config.out_domain:
        template["Out_Domain"] = config.out_domain

    # Optional early-out: if overall start is later than current time, don't request
    try:
        overall_start = datetime.strptime(period_start, ISO_YMDHM)
        now_ts = datetime.utcnow()
        if overall_start > now_ts:
            print("Note: start date is later than current time; skipping NUCS requests.")
            return pd.DataFrame(columns=[
                'ts', 'procurement_price', 'quantity',
            ])
    except Exception:
        # If parsing fails, proceed without early-out
        pass

    periods = build_periods(period_start, period_end, chunk_days)
    all_points: List[Dict[str, Any]] = []
    for w in periods:
        params = build_params(template, w)
        try:
            xml_bytes = fetch_nucs_xml(config.base_url, config.token, params)
        except requests.exceptions.HTTPError as e:
            code = getattr(e.response, 'status_code', None)
            if code in (204, 400, 404):
                # Treat as "no data for this time window"
                print(f"OK: no NUCS data for window {w['periodStart']}..{w['periodEnd']} (HTTP {code}); skipping.")
                continue
            # Other HTTP errors: surface them
            raise
        except Exception as e:
            # Non-HTTP error while requesting; log and continue
            print(f"Warning: request failed for window {w['periodStart']}..{w['periodEnd']}: {e}")
            continue

        if not xml_bytes:
            print(f"OK: empty response for window {w['periodStart']}..{w['periodEnd']}; skipping.")
            continue
        try:
            pts = parse_nucs_points(xml_bytes)
        except ET.ParseError:
            print(f"OK: invalid/empty XML for window {w['periodStart']}..{w['periodEnd']}; skipping.")
            continue
        print(f" -> {len(pts)} points from {w['periodStart']} to {w['periodEnd']}")
        all_points.extend(pts)

    df_points = pd.DataFrame(all_points)
    return df_points


def get_nucs_mtc(
    base_url: str,
    token: str,
    period_start: str,
    period_end: str,
    in_domain: str,
    out_domain: str,
    process_type: str = "A47",  # A47 = mFRR, A51 = aFRR (can pass A51 for aFRR capacity flows)
    business_type: str = "A31",  # A31 Offered capacity (or A27 Net transmission capacity)
    chunk_days: int = 50,
    save_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch Maximum Transfer Capacity (MTC) / offered or net transmission capacity via NUCS.

    Parameters
    ----------
    base_url : str
        Base NUCS host (e.g., https://...)
    token : str
        securityToken
    period_start / period_end : str
        YYYYMMDDHHMM boundaries (<=370 days span due to API limit)
    in_domain / out_domain : str
        EIC codes for importing/exporting areas
    process_type : str
        A47 (mFRR) or A51 (aFRR) depending on product (default A47)
    business_type : str
        A31 offered capacity or A27 net transmission capacity
    chunk_days : int
        Chunk size for large ranges
    save_csv : Optional[str]
        Path to save aggregated hourly capacity

    Returns
    -------
    DataFrame with columns:
        ts (hour), mtc_capacity (mean), meta columns (businessType, ...)
    """
    # documentType A26 = Capacity document
    config = NucsConfig(
        base_url=base_url,
        token=token,
        document_type="A26",
        process_type=process_type,
        market_agreement_type="A01",  # keep default; may not be required for A26
        control_area="",  # not used for MTC (we rely on in/out domains)
        in_domain=in_domain,
        out_domain=out_domain,
    )

    # Reuse fetch_and_parse which will inject In_Domain/Out_Domain
    df_points = fetch_and_parse(config, period_start, period_end, chunk_days)
    if df_points.empty:
        out = pd.DataFrame(columns=["ts", "mtc_capacity"])
        if save_csv:
            os.makedirs(os.path.dirname(save_csv), exist_ok=True)
            out.to_csv(save_csv, index=False)
        return out

    # Rename procurement_price/quantity meaning when documentType=A26 -> interpret quantity if present
    # Some A26 docs may use quantity or price fields; we search for 'quantity' first.
    # Build hourly aggregation for capacity values.
    df_points = df_points.rename(columns={"quantity": "capacity_raw"})
    hourly = (
        df_points.dropna(subset=["ts"])
                  .set_index("ts")
                  .sort_index()
                  .resample("1h")
                  .agg(
                      mtc_capacity=("capacity_raw", "mean"),
                  )
                  .reset_index()
    )

    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        hourly.to_csv(save_csv, index=False)
        print("Saved MTC CSV:", save_csv)
    if not hourly.empty:
        last_hour = pd.to_datetime(hourly["ts"]).max().floor("h")
        print("Last MTC hour fetched:", last_hour.strftime("%Y-%m-%d %H:%M"))
    return hourly


def to_price_df(
    hourly_df: pd.DataFrame,
    time_col: str = "time",
    price_col: str = "aFRR_price",
) -> pd.DataFrame:
    """Return a compact two-column DataFrame with named time/price columns.

    - hourly_df: DataFrame from points_to_hourly (expects columns ['ts','procurement_price']).
    - time_col: output column name for timestamps (default 'time').
    - price_col: output column name for price (default 'aFRR_price').
    """
    if hourly_df is None or hourly_df.empty:
        return pd.DataFrame(columns=[time_col, price_col])
    # Prefer up_price if available, otherwise fall back to a generic price column
    src_col = None
    if 'up_price' in hourly_df.columns:
        src_col = 'up_price'
    elif 'procurement_price' in hourly_df.columns:
        src_col = 'procurement_price'
    else:
        # take the second column as a best-effort fallback
        src_col = hourly_df.columns[1]

    out = (
        hourly_df.loc[:, ["ts", src_col]]
        .rename(columns={"ts": time_col, src_col: price_col})
        .dropna(subset=[time_col, price_col])
        .sort_values(time_col)
        .reset_index(drop=True)
    )
    return out


def get_nucs_hourly(
    base_url: str,
    token: str,
    period_start: str,
    period_end: str,
    document_type: str = "A81",
    process_type: str = "A51",
    market_agreement_type: str = "A01",
    control_area: str = "10YNO-1--------2",
    business_type: str | None = None,
    chunk_days: int = 50,
    save_csv: Optional[str] = None,
    plots_dir: Optional[str] = None,
    stem: str = "nucs_hourly",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function: fetch NUCS balancing/capacity data across period chunks,
    return (points_df, hourly_df). Optionally save outputs.

    points_df columns: ts, procurement_price, quantity, plus any meta (businessType,...)
    hourly_df columns: ts (hour), procurement_price (mean), quantity (sum)
    """
    config = NucsConfig(
        base_url=base_url,
        token=token,
        document_type=document_type,
        process_type=process_type,
        market_agreement_type=market_agreement_type,
        control_area=control_area,
    )
    # Inject businessType if requested (DF09 contracted reserves)
    points_df = fetch_and_parse(config, period_start, period_end, chunk_days)
    if business_type:
        # Filter to requested businessType if meta present
        if 'businessType' in points_df.columns:
            points_df = points_df[points_df['businessType'].astype(str).str.upper() == str(business_type).upper()]
    hourly_df = points_to_hourly(points_df)
    # Smooth extreme spikes (e.g., single 2500) using robust threshold
    hourly_df = smooth_outliers(hourly_df, price_col="procurement_price", max_price=None, iqr_factor=4.0)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        hourly_df.to_csv(save_csv, index=False)
        print("Saved CSV:", save_csv)
    if plots_dir:
        plot_outputs(hourly_df, plots_dir, stem=stem)
    if not hourly_df.empty:
        last_hour = pd.to_datetime(hourly_df["ts"]).max()
        try:
            last_hour_floor = pd.to_datetime(last_hour).floor('h')
        except Exception:
            last_hour_floor = last_hour
        print("Last hour fetched:", last_hour_floor.strftime("%Y-%m-%d %H:%M"))
    return points_df, hourly_df


def get_nucs_price_df(
    base_url: str,
    token: str,
    period_start: str,
    period_end: str,
    document_type: str = "A81",
    process_type: str = "A51",
    market_agreement_type: str = "A01",
    control_area: str = "10YNO-1--------2",
    chunk_days: int = 50,
    product_label: str = "aFRR",  # used to name the price column
    time_col: str = "time",
    save_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch NUCS and return a compact DF with columns [time_col, f"{product_label}_price"]."""
    _, hourly_df = get_nucs_hourly(
        base_url=base_url,
        token=token,
        period_start=period_start,
        period_end=period_end,
        document_type=document_type,
        process_type=process_type,
        market_agreement_type=market_agreement_type,
        control_area=control_area,
        chunk_days=chunk_days,
    )
    price_col = f"{product_label}_price" if product_label else "price"
    price_df = to_price_df(hourly_df, time_col=time_col, price_col=price_col)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        price_df.to_csv(save_csv, index=False)
        print("Saved CSV:", save_csv)
    return price_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch NUCS hourly (DF09/DFxx) and save CSV/plots")
    parser.add_argument("--base-url", nargs="?", const="", required=False)
    parser.add_argument("--token", nargs="?", const="", required=False)
    parser.add_argument("--start", required=True, help="YYYYMMDDHHMM")
    parser.add_argument("--end", required=True, help="YYYYMMDDHHMM")
    parser.add_argument("--chunk-days", type=int, default=50)
    parser.add_argument("--control-area", type=str, default="10YNO-1--------2")
    parser.add_argument("--document-type", type=str, default="A81")
    parser.add_argument("--process-type", type=str, default="A51", help="A51=aFRR, A47=mFRR")
    parser.add_argument("--market-agreement-type", type=str, default="A01")
    parser.add_argument("--business-type", type=str, default=None, help="Optional businessType filter: A96=aFRR, A97=mFRR")
    parser.add_argument("--out-csv", type=str, default=str(os.path.join("upreg_classify","reports","dataframes","nucs_hourly.csv")))
    parser.add_argument("--plots-dir", type=str, default=str(os.path.join("upreg_classify","reports","plots")))
    parser.add_argument("--stem", type=str, default="nucs_hourly")
    args = parser.parse_args()

    # Resolve Windows-style %ENV% placeholders or empty values from environment
    def _resolve_env(val: Optional[str], env_key: str) -> Optional[str]:
        if not val:
            return os.environ.get(env_key)
        v = str(val)
        # If user passed PowerShell-incompatible %VAR% text, replace from env
        if v.startswith("%") and v.endswith("%"):
            return os.environ.get(env_key, v)
        # If literal placeholder like %NUCS_BASE_URL% appears inside, replace
        if f"%{env_key}%" in v:
            return os.environ.get(env_key, v)
        return v

    # Minimal .env loader: try repo root and upreg_classify/.env
    def _load_env_file(path: str) -> None:
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip()
                            if k and v and k not in os.environ:
                                os.environ[k] = v
        except Exception:
            pass

    cwd = os.getcwd()
    _load_env_file(os.path.join(cwd, ".env"))
    _load_env_file(os.path.join(cwd, "upreg_classify", ".env"))

    base_url = _resolve_env(args.base_url, "NUCS_BASE_URL")
    # Accept either NUCS_TOKEN or NUCS_API_KEY as token source
    token = _resolve_env(args.token, "NUCS_TOKEN")
    if not token:
        token = os.environ.get("NUCS_API_KEY")
    if not base_url or not token:
        print("Error: Missing NUCS credentials. Provide --base-url/--token or set NUCS_BASE_URL/NUCS_TOKEN.")
        print("Tip: You can set NUCS_TOKEN or NUCS_API_KEY. In PowerShell use $env:NUCS_BASE_URL / $env:NUCS_TOKEN; in cmd.exe use %NUCS_BASE_URL%.")
        sys.exit(1)

    pts, hourly = get_nucs_hourly(
        base_url=base_url,
        token=token,
        period_start=args.start,
        period_end=args.end,
        document_type=args.document_type,
        process_type=args.process_type,
        market_agreement_type=args.market_agreement_type,
        control_area=args.control_area,
        business_type=args.business_type,
        chunk_days=args.chunk_days,
        save_csv=args.out_csv,
        plots_dir=args.plots_dir,
        stem=args.stem,
    )
    print(f"Points rows: {len(pts)} | Hourly rows: {len(hourly)}")
    print("Saved:", args.out_csv)
