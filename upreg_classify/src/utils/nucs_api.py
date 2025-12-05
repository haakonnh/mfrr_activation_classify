from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

# -----------------------------
# Time helpers
# -----------------------------

ISO_YMDHM = "%Y%m%d%H%M"
_DUR_RE = re.compile(r"^P(T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?)$", re.I)


def parse_iso_duration(text: str) -> timedelta:
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
class NucsClient:
    base_url: str
    token: str

    def _api_url(self) -> str:
        return self.base_url.rstrip("/") + "/api"

    def fetch_xml(self, params: Dict[str, Any], timeout: int = 90) -> bytes:
        if not self.base_url or "<" in self.base_url or ">" in self.base_url:
            raise ValueError("base_url looks like a placeholder. Provide a real host.")
        if not self.token or "<" in self.token or ">" in self.token:
            raise ValueError("securityToken looks like a placeholder; pass the raw token.")
        q = {"securityToken": self.token}
        q.update(params or {})
        r = requests.get(self._api_url(), params=q, timeout=timeout)
        # Surface useful context in logs
        print("NUCS URL:", r.url)
        r.raise_for_status()
        return r.content


# namespace-agnostic local tag
ltag = lambda e: e.tag.split('}')[-1] if isinstance(e.tag, str) else ''  # noqa: E731


def parse_nucs_points(xml_bytes: bytes) -> List[Dict[str, Any]]:
    """Parse NUCS XML (namespace-agnostic) into point-level rows.

    Returns list of dicts: {ts, procurement_price, quantity, ...meta}
    """
    import xml.etree.ElementTree as ET

    if not xml_bytes:
        return []
    root = ET.fromstring(xml_bytes)
    points: List[Dict[str, Any]] = []

    series_list = [el for el in root.iter() if ltag(el) == 'TimeSeries']
    for ts in series_list:
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

        res_el = next((el for el in period.iter() if ltag(el) == 'resolution' and el.text), None)
        delta = parse_iso_duration(res_el.text if res_el is not None else 'PT60M')

        # meta fields (optional, present in many docs like DF09/DF11)
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

            # price: any descendant tag with 'price' in its localname
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
    if df_points is None or df_points.empty:
        return pd.DataFrame(columns=['ts', 'procurement_price', 'quantity'])
    hourly = (
        df_points.dropna(subset=['ts'])
                 .set_index('ts')
                 .sort_index()
                 .resample('1h')
                 .agg(
                     procurement_price=('procurement_price', 'mean'),
                     quantity=('quantity', 'sum'),
                 )
                 .reset_index()
    )
    return hourly


# -----------------------------
# Contracted reserves (DF09/A81)
# -----------------------------

@dataclass
class DF09Query:
    control_area: str = "10YNO-1--------2"  # NO1
    market_agreement_type: str = "A01"      # Daily (keep default)
    business_type: str = "A96"              # mFRR contracted reserves (A96 for aFRR)


def fetch_df09_points(
    client: NucsClient,
    period_start: str,
    period_end: str,
    cfg: DF09Query | None = None,
    chunk_days: int = 50,
) -> pd.DataFrame:
    """Fetch DF09 (A81) contracted reserves points across chunked windows.

    Returns a DataFrame with columns ['ts','procurement_price','quantity', ...meta].
    """
    periods = build_periods(period_start, period_end, chunk_days)
    rows: List[Dict[str, Any]] = []
    for win in periods:
        print(win)
        params = {
            "documentType": "A81",  # DF09
            "type_marketagreement.type": cfg.market_agreement_type,
            "controlArea_domain": cfg.control_area,
            "processType": "A47",
            **win,
        }
        try:
            xml_bytes = client.fetch_xml(params)
        except requests.exceptions.HTTPError as e:
            code = getattr(e.response, 'status_code', None)
            if code in (204, 404, 400):
                text = getattr(e.response, 'text', '')
                msg = f"HTTP {code} for {win['periodStart']}..{win['periodEnd']} — treating as no data; continuing."
                if text:
                    # Log a trimmed body for debugging, but do not fail the run
                    print(text[:500])
                print(msg)
                continue
            raise
        except Exception as e:
            print(f"Warning: request failed for window {win['periodStart']}..{win['periodEnd']}: {e}")
            continue
        try:
            pts = parse_nucs_points(xml_bytes)
        except Exception as e:
            print(f"Warning: parse error for window {win['periodStart']}..{win['periodEnd']}: {e}")
            continue
        print(f" -> {len(pts)} points from {win['periodStart']} to {win['periodEnd']}")
        rows.extend(pts)
    return pd.DataFrame(rows)


def get_contracted_reserves_hourly(
    client: NucsClient,
    period_start: str,
    period_end: str,
    cfg: DF09Query | None = None,
    chunk_days: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: fetch DF09 and aggregate to hourly.

    Returns (points_df, hourly_df) where hourly_df has columns ['ts','procurement_price','quantity'].
    """
    points = fetch_df09_points(client, period_start, period_end, cfg=cfg, chunk_days=chunk_days)
    hourly = points_to_hourly(points)
    return points, hourly


def get_mfrr_contracted_hourly(
    base_url: str,
    token: str,
    period_start: str,
    period_end: str,
    control_area: str = "10YNO-1--------2",
    market_agreement_type: str = "A01",
    business_type: str = "A97",
    chunk_days: int = 50,
    save_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch DF09 (A81) contracted reserves and aggregate hourly.

    Query params used: documentType=A81, type_marketagreement.type, businessType,
    controlArea_domain, and period windows. No processType is included.

    If save_csv is provided, saves hourly to the given path.
    """
    client = NucsClient(base_url=base_url, token=token)
    cfg = DF09Query(
        control_area=control_area,
        market_agreement_type=market_agreement_type,
        business_type=business_type,
    )
    points, hourly = get_contracted_reserves_hourly(client, period_start,
                                                    period_end, cfg=cfg, chunk_days=chunk_days)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        hourly.to_csv(save_csv, index=False)
        print("Saved CSV:", save_csv)
    if not hourly.empty:
        last_hour = pd.to_datetime(hourly['ts']).max()
        print("Last hour fetched:", pd.to_datetime(last_hour).floor('h'))
    return points, hourly
