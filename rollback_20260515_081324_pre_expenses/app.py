"""
Venu Cast — Full-Stack Forecasting System
=========================================
Loads 5 template files via web upload, parses all sheets/tabs/venues,
generates daily forecasts (May 2026 - June 2027) using Prophet/SARIMA/Holt-Winters
with cluster interactions, weather, holidays, and 15-month multipliers.

Output: Daily forecast per venue × 14 months, including:
- Servings sales (with multiplier applied)
- Retail sales
- POS discounts
- ATP forecast
- Transaction forecast (sales / ATP)
"""
import os, io, json, math, logging, traceback, warnings, uuid
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

try:
    import holidays as _holidays_pkg
    HOLIDAYS_OK = True
except ImportError:
    HOLIDAYS_OK = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
CORS(app)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
SNAPSHOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snapshots')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

FORECAST_START = datetime(2026, 5, 1)
FORECAST_END = datetime(2027, 6, 30)
FORECAST_DAYS = (FORECAST_END - FORECAST_START).days + 1

SESSION_STATE = {
    'sales_servings': None,
    'sales_retail': None,
    'sales_discounts': None,
    'atp_history': None,
    'location_bi': None,
    'venue_details': None,
    'multipliers': None,
    'atp_growth_template': None,
    'last_results': None,
    # Scenarios: { scenario_name: { venue_overrides: { venue: { target_growth_pct, comments: [c1..c5] } }, created_at } }
    # Up to 3 scenarios kept (oldest deleted on overflow).
    'scenarios': {},
    'active_scenario': None,
}


# ── UTILS ──────────────────────────────────────────────────────────────────────
def safe(v):
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except Exception:
        return None

def iso(d):
    return d.strftime("%Y-%m-%d")

def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


# ── STATE → REPRESENTATIVE CITY (for weather) ──────────────────────────────────
STATE_COORDS = {
    'VIC': (-37.8136, 144.9631, 'Melbourne'),
    'NSW': (-33.8688, 151.2093, 'Sydney'),
    'QLD': (-27.4698, 153.0251, 'Brisbane'),
    'SA':  (-34.9285, 138.6007, 'Adelaide'),
    'WA':  (-31.9505, 115.8605, 'Perth'),
    'TAS': (-42.8821, 147.3272, 'Hobart'),
    'ACT': (-35.2809, 149.1300, 'Canberra'),
    'NT':  (-12.4634, 130.8456, 'Darwin'),
}

WEATHER_CACHE = {}     # state -> {date_str: temp_max}
HOLIDAY_CACHE = {}     # state -> {date_str: name}


# ── HOLIDAYS BY STATE (historical + future) ────────────────────────────────────
def load_holidays_for_state(state, year_range):
    """Get all public holidays for a state for the given year range."""
    if not HOLIDAYS_OK or not state:
        return {}
    cache_key = f"{state}_{year_range[0]}_{year_range[-1]}"
    if cache_key in HOLIDAY_CACHE:
        return HOLIDAY_CACHE[cache_key]
    try:
        au_hols = _holidays_pkg.Australia(years=list(year_range), subdiv=state)
        result = {d.strftime("%Y-%m-%d"): str(name) for d, name in au_hols.items()}
        HOLIDAY_CACHE[cache_key] = result
        log.info(f"  Loaded {len(result)} holidays for state {state}")
        return result
    except Exception as e:
        log.warning(f"  Holidays load failed for {state}: {e}")
        return {}


# ── WEATHER (Open-Meteo: temp + rain + wind + cyclone-like extreme events) ─────
# Extreme-event thresholds (cyclone-like / severe storm)
EXTREME_GUSTS_KMH = 90.0   # Cyclone Cat 1 gusts start ~90 km/h
EXTREME_PRECIP_MM = 50.0   # Heavy rain / flood day

def load_weather_for_state(state, start_date, end_date):
    """Get daily weather for a state across date range.
    Variables: temp_max (°C), precip (mm), wind_max (km/h), gusts_max (km/h), extreme (0/1).
    Historical: Open-Meteo archive. Future (beyond today): seasonal climatology.
    Extreme-event flag captures cyclones, severe storms, heavy rain days.
    Returns dict {date_str: {temp_max, precip, wind_max, gusts_max, extreme}}.
    """
    if not state or state not in STATE_COORDS:
        return {}

    cache_key = f"{state}_{start_date}_{end_date}"
    if cache_key in WEATHER_CACHE:
        return WEATHER_CACHE[cache_key]

    lat, lon, city = STATE_COORDS[state]
    today = datetime.now().date()
    hist_end = min(end_date.date() if isinstance(end_date, datetime) else end_date,
                   today - timedelta(days=2))
    hist_start = start_date.date() if isinstance(start_date, datetime) else start_date

    result = {}

    # Historical archive
    if hist_start <= hist_end:
        try:
            r = requests.get('https://archive-api.open-meteo.com/v1/archive', params={
                'latitude': lat,
                'longitude': lon,
                'start_date': hist_start.strftime("%Y-%m-%d"),
                'end_date': hist_end.strftime("%Y-%m-%d"),
                'daily': 'temperature_2m_max,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max',
                'timezone': 'auto',
            }, timeout=120)
            if r.ok:
                d = r.json().get('daily', {})
                times = d.get('time', [])
                temps = d.get('temperature_2m_max', [])
                precips = d.get('precipitation_sum', [])
                winds = d.get('wind_speed_10m_max', [])
                gusts = d.get('wind_gusts_10m_max', [])
                n_extreme = 0
                for i, dt_str in enumerate(times):
                    t = temps[i] if i < len(temps) and temps[i] is not None else None
                    p = precips[i] if i < len(precips) and precips[i] is not None else 0.0
                    w = winds[i] if i < len(winds) and winds[i] is not None else None
                    g = gusts[i] if i < len(gusts) and gusts[i] is not None else None
                    extreme = 1 if (g is not None and g >= EXTREME_GUSTS_KMH) or (p is not None and p >= EXTREME_PRECIP_MM) else 0
                    if extreme:
                        n_extreme += 1
                    result[dt_str] = {
                        'temp_max': float(t) if t is not None else None,
                        'precip': float(p),
                        'wind_max': float(w) if w is not None else None,
                        'gusts_max': float(g) if g is not None else None,
                        'extreme': extreme,
                    }
                log.info(f"  Weather archive for {state} ({city}): {len(result)} days, {n_extreme} extreme-weather days")
        except Exception as e:
            log.warning(f"  Weather archive {state}: {e}")

    # Climatology for future dates (per-(month,day) averages from historical)
    end_d = end_date.date() if isinstance(end_date, datetime) else end_date
    if end_d > hist_end:
        clim = {}  # (month, day) -> {var: [values]}
        for date_str, w in result.items():
            d = datetime.strptime(date_str, "%Y-%m-%d")
            key = (d.month, d.day)
            if key not in clim:
                clim[key] = {'temp_max': [], 'precip': [], 'wind_max': [], 'gusts_max': [], 'extreme_freq': []}
            for var in ('temp_max', 'precip', 'wind_max', 'gusts_max'):
                v = w.get(var)
                if v is not None:
                    clim[key][var].append(v)
            clim[key]['extreme_freq'].append(w.get('extreme', 0))

        cur = max(hist_end + timedelta(days=1), hist_start)
        while cur <= end_d:
            key = (cur.month, cur.day)
            if key not in clim and key == (2, 29):
                key = (2, 28)
            if key in clim:
                avg = {var: (sum(vals) / len(vals) if vals else None) for var, vals in clim[key].items() if var != 'extreme_freq'}
                # For future climatology, use probabilistic extreme (rounded freq)
                extreme_freq = mean(clim[key]['extreme_freq']) if clim[key]['extreme_freq'] else 0
                extreme_future = 1 if extreme_freq >= 0.5 else 0
                result[cur.strftime("%Y-%m-%d")] = {
                    'temp_max': avg.get('temp_max'),
                    'precip': avg.get('precip') if avg.get('precip') is not None else 0.0,
                    'wind_max': avg.get('wind_max'),
                    'gusts_max': avg.get('gusts_max'),
                    'extreme': extreme_future,
                }
            cur += timedelta(days=1)

    WEATHER_CACHE[cache_key] = result
    return result


def get_venue_state(venue):
    """Resolve a venue's state from loaded templates (location_bi or venue_details or multipliers)."""
    loc_bi = SESSION_STATE.get('location_bi') or {}
    if venue in loc_bi:
        s = (loc_bi[venue].get('state') or '').strip().upper()
        if s and s != 'NAN':
            return s
    details = SESSION_STATE.get('venue_details') or {}
    if venue in details:
        s = (details[venue].get('state') or '').strip().upper()
        if s and s != 'NAN':
            return s
    mults = SESSION_STATE.get('multipliers') or {}
    if venue in mults:
        s = (mults[venue].get('state') or '').strip().upper()
        if s and s != 'NAN':
            return s
    return None


# ── FILE PARSERS ───────────────────────────────────────────────────────────────
def parse_sales_history(file_storage):
    """Parse 01_Sales_History_Template.xlsx with 3 tabs (Servings, Retail, POS Discounts).
    Wide format: rows = venues, columns = dates.
    Returns dict of {tab_name: {venue_name: {date_str: value}}}.
    """
    data = io.BytesIO(file_storage.read())
    file_storage.seek(0)
    xls = pd.ExcelFile(data)

    result = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(data, sheet_name=sheet)
        date_cols = []
        for col in df.columns[1:]:
            try:
                if isinstance(col, str):
                    dt = pd.to_datetime(col)
                elif isinstance(col, datetime):
                    dt = col
                else:
                    dt = pd.to_datetime(col)
                date_cols.append((col, dt.strftime("%Y-%m-%d")))
            except Exception:
                continue

        venues_dict = {}
        for _, row in df.iterrows():
            venue = row.iloc[0]
            if pd.isna(venue) or not isinstance(venue, str):
                continue
            venue = str(venue).strip()
            series = {}
            for col, date_str in date_cols:
                try:
                    v = row[col]
                    if pd.isna(v):
                        continue
                    series[date_str] = float(v)
                except Exception:
                    continue
            if series:
                venues_dict[venue] = series
        result[sheet] = venues_dict
        log.info(f"  Parsed sheet '{sheet}': {len(venues_dict)} venues, {len(date_cols)} dates")

    return result


def parse_atp_history(file_storage):
    """Parse Average ticket history.xlsx (long format).
    Columns: Date, Account, Net Amount, TrackingCategoryOption1, Table.Venue Name Budget, Transactions, ATV
    Returns dict {venue: {date_str: atv}}.
    """
    data = io.BytesIO(file_storage.read())
    file_storage.seek(0)
    df = pd.read_excel(data)
    df.columns = [str(c).strip() for c in df.columns]

    date_col = None
    venue_col = None
    atv_col = None
    for c in df.columns:
        if c.lower() == 'date':
            date_col = c
        elif 'venue' in c.lower() and 'budget' in c.lower():
            venue_col = c
        elif c.lower() == 'atv':
            atv_col = c

    if not (date_col and venue_col and atv_col):
        log.warning(f"ATP columns missing: date={date_col}, venue={venue_col}, atv={atv_col}")
        return {}

    result = {}
    for _, row in df.iterrows():
        venue = row[venue_col]
        if pd.isna(venue):
            continue
        venue = str(venue).strip()
        try:
            d = pd.to_datetime(row[date_col]).strftime("%Y-%m-%d")
            atv = float(row[atv_col])
            if atv > 0 and not math.isnan(atv):
                result.setdefault(venue, {})[d] = atv
        except Exception:
            continue

    log.info(f"  Parsed ATP history: {len(result)} venues")
    return result


def parse_location_bi(file_storage):
    """Parse D03_Location_BI.xlsx for cluster information.
    Returns dict {venue: {cluster, cluster_type, region, state, open_status}}.
    """
    data = io.BytesIO(file_storage.read())
    file_storage.seek(0)
    df = pd.read_excel(data)
    df.columns = [str(c).strip() for c in df.columns]

    result = {}
    for _, row in df.iterrows():
        venue = row.get('RC_Loc_Name', None)
        if pd.isna(venue) or not isinstance(venue, str):
            continue
        venue = venue.strip()
        if venue == 'Default location':
            continue
        result[venue] = {
            'cluster': str(row.get('Cat_Cluster', '') or '').strip(),
            'cluster_type': str(row.get('Cat_Cluster_Type', '') or '').strip(),
            'cannibalisation': str(row.get('Cat_Cannibalisation', '') or '').strip(),
            'region': str(row.get('Cat_Region', '') or '').strip(),
            'state': str(row.get('Address_State', '') or '').strip(),
            'date_open': str(row.get('Date_Open', '') or '').strip(),
            'open_status': str(row.get('Open_Closed', '') or '').strip(),
            'reporting': str(row.get('Reporting', '') or '').strip(),
        }

    log.info(f"  Parsed Location BI: {len(result)} venues")
    return result


def parse_venue_details(file_storage):
    """Parse Venue Details.xlsx with venue-level drivers.
    Columns include: venue_name, state, opening_date, is_active, New Monthly sales, Model off,
    Month 1-6 Ramp, Impact venue 1-4 + Mth 1-6 columns.
    Returns dict {venue: {state, opening_date, ramp_up, impacts, base_monthly_sales}}.
    """
    data = io.BytesIO(file_storage.read())
    file_storage.seek(0)
    df = pd.read_excel(data)
    df.columns = [str(c).strip() for c in df.columns]

    result = {}
    for _, row in df.iterrows():
        venue = row.get('venue_name', None)
        if pd.isna(venue) or not isinstance(venue, str):
            continue
        venue = venue.strip()

        ramp = []
        for i in range(1, 7):
            v = row.get(f'Month {i} Ramp', None)
            try:
                ramp.append(float(v) if v is not None and not pd.isna(v) else None)
            except Exception:
                ramp.append(None)

        impacts = []
        for impact_idx in range(1, 5):
            impact_col = f'Impact venue {impact_idx}' if impact_idx == 1 else f'Impact venue {impact_idx}'
            if impact_col not in df.columns:
                continue
            impacted_venue = row.get(impact_col, None)
            if pd.isna(impacted_venue) or not isinstance(impacted_venue, str):
                continue
            mth_pcts = []
            for m in range(1, 7):
                if impact_idx == 1:
                    col = f'Mth {m}'
                else:
                    col = f'Mth {m}.{impact_idx - 1}'
                v = row.get(col, None)
                try:
                    mth_pcts.append(float(v) if v is not None and not pd.isna(v) else None)
                except Exception:
                    mth_pcts.append(None)
            impacts.append({'impacted_venue': impacted_venue.strip(), 'monthly_impact_pcts': mth_pcts})

        try:
            new_monthly = float(row.get('New Monthly sales', 0) or 0)
        except Exception:
            new_monthly = 0.0
        try:
            model_off_v = row.get('Model off', None)
            model_off = bool(model_off_v) if model_off_v is not None and not pd.isna(model_off_v) else False
        except Exception:
            model_off = False

        opening = row.get('opening_date', None)
        opening_str = ''
        try:
            if opening is not None and not pd.isna(opening):
                opening_str = pd.to_datetime(opening).strftime("%Y-%m-%d")
        except Exception:
            opening_str = str(opening) if opening is not None else ''

        is_active = row.get('is_active', '')
        is_active_str = str(is_active).strip().upper() if not pd.isna(is_active) else ''

        result[venue] = {
            'state': str(row.get('state', '') or '').strip(),
            'opening_date': opening_str,
            'is_active': is_active_str,
            'new_monthly_sales': new_monthly,
            'model_off': model_off,
            'ramp_up_pcts': ramp,
            'impacts': impacts,
        }

    log.info(f"  Parsed Venue Details: {len(result)} venues")
    return result


def parse_multipliers(file_storage):
    """Parse venue_city_mapping_updated_1.csv with 15-month multipliers.
    Returns dict {venue: [m1, m2, ..., m15]}.
    """
    data = io.BytesIO(file_storage.read())
    file_storage.seek(0)
    df = pd.read_csv(data)
    df.columns = [str(c).strip() for c in df.columns]

    result = {}
    for _, row in df.iterrows():
        venue = row.get('venue', None)
        if pd.isna(venue) or not isinstance(venue, str):
            continue
        venue = venue.strip()
        mults = []
        for i in range(1, 16):
            v = row.get(f'Mth {i} Multiplier', 1.0)
            try:
                mults.append(float(v) if v is not None and not pd.isna(v) else 1.0)
            except Exception:
                mults.append(1.0)
        result[venue] = {
            'city': str(row.get('city', '') or '').strip(),
            'state': str(row.get('state', '') or '').strip(),
            'multipliers': mults,
        }

    log.info(f"  Parsed multipliers: {len(result)} venues")
    return result


# ── HISTORY TRIMMING ───────────────────────────────────────────────────────────
def trim_history(dates, values):
    """Drop leading zero days so the model trains only on the actual trading period.
    Critical for venues that opened mid-history: the sales sheet has 5 years of
    leading zeros which would otherwise mislead Prophet into yearly seasonality
    and runaway trend extrapolation.
    """
    first_nz = next((i for i, v in enumerate(values) if v > 0), None)
    if first_nz is None:
        return dates, values
    return dates[first_nz:], values[first_nz:]


# ── VENUE HISTORY ANALYSIS (from prior Venu Cast model) ────────────────────────
def analyse_venue_history(dates, values):
    """Per prior Venu Cast model: classify venue and pick model complexity safely.
    Returns dict with trading-day flags so Prophet is bypassed for very new venues.
    """
    n_days = len(dates)
    non_zero = [v for v in values if v > 0]
    n_trading = len(non_zero)

    if not non_zero:
        return dict(n_days=n_days, n_trading=0, patchiness=1.0,
                    stable_mean=0, recent_mean=0, recent_max=0,
                    dow_means={}, is_patchy=True, use_flat_forecast=True,
                    bypass_prophet=True, has_any_history=False)

    patchiness = 1 - (n_trading / n_days) if n_days > 0 else 0
    sorted_nz = sorted(non_zero)
    mid = len(sorted_nz) // 2
    stable_mean = (sorted_nz[mid - 1] + sorted_nz[mid]) / 2 if len(sorted_nz) % 2 == 0 else sorted_nz[mid]

    # Last 56 days -> last 28 non-zero
    recent_window = [v for v in values[-56:] if v > 0][-28:]
    recent_mean = mean(recent_window) if recent_window else stable_mean
    recent_max = max(recent_window) if recent_window else stable_mean

    # Day-of-week averages from non-zero days
    dow_sums = [[] for _ in range(7)]
    for d_str, v in zip(dates, values):
        if v > 0:
            try:
                dow = datetime.strptime(d_str, "%Y-%m-%d").weekday()
                dow_sums[dow].append(v)
            except Exception:
                continue
    dow_means = {i: (mean(lst) if lst else recent_mean) for i, lst in enumerate(dow_sums)}

    is_patchy = patchiness > 0.30 or n_trading < 21
    bypass_prophet = n_trading < 21          # <-- Kawana Waters fix
    use_flat_forecast = n_trading < 28 or (is_patchy and n_trading < 56)

    return dict(n_days=n_days, n_trading=n_trading, patchiness=patchiness,
                stable_mean=stable_mean, recent_mean=recent_mean, recent_max=recent_max,
                dow_means=dow_means, is_patchy=is_patchy,
                use_flat_forecast=use_flat_forecast, bypass_prophet=bypass_prophet,
                has_any_history=True)


def dow_flat_forecast(analysis, h, last_date_str):
    """Day-of-week-flat forecast for very new venues (< 21 trading days).
    Projects each weekday's historical average forward — no trend extrapolation.
    """
    dow_means = analysis["dow_means"] or {}
    recent = analysis["recent_mean"]
    recent_max = analysis["recent_max"]
    ceiling = max(recent_max * 1.5, recent * 2.0, 1.0)

    last_d = datetime.strptime(last_date_str, "%Y-%m-%d")
    fc_out, lo_out, hi_out = [], [], []
    for i in range(h):
        d = last_d + timedelta(days=i + 1)
        base = dow_means.get(d.weekday(), recent)
        base = min(base, ceiling)
        fc_out.append(max(0, base))
        lo_out.append(max(0, base * 0.80))
        hi_out.append(min(ceiling, base * 1.20))

    n = len(analysis["dow_means"])
    flat = recent
    fitted = [flat] * n if n else []
    rmse = 0.0
    return fitted, fc_out, lo_out, hi_out, rmse, "DOW-Flat"


def sanitise_forecast(fc, lo, hi, analysis):
    """Apply hard ceilings so forecasts can't balloon beyond a venue's trading range.
    Bands tighten during the ramp window — a venue with only ~6 months of history
    shouldn't be allowed to forecast 4× its current stable mean.
      < 28 trading days   → recent_max × 1.5
      28-89   trading days → max(recent_max × 2, stable × 3)
      90-179  trading days → max(recent_max × 1.8, stable × 2.5)     [ramp guard]
      180-364 trading days → max(recent_max × 1.6, stable × 2.5)     [ramp guard]
      365+    trading days → stable × 4                              [mature]
    CI scaled proportionally when a day is clamped.
    """
    stable = analysis["stable_mean"]
    recent = analysis["recent_mean"]
    recent_max = analysis["recent_max"]
    n_trade = analysis["n_trading"]
    h = len(fc)

    if stable <= 0 and recent <= 0:
        return [0.0] * h, [0.0] * h, [0.0] * h, 0

    if n_trade < 28:
        ceiling = max(recent_max * 1.5, recent * 1.5, 1.0)
    elif n_trade < 90:
        ceiling = max(recent_max * 2.0, stable * 3.0)
    elif n_trade < 180:
        ceiling = max(recent_max * 1.8, stable * 2.5)   # still ramping
    elif n_trade < 365:
        ceiling = max(recent_max * 1.6, stable * 2.5)   # still sub-year
    else:
        ceiling = stable * 4.0                            # mature, allow seasonal peaks

    # Recent-mean floor: a venue with established trading patterns shouldn't
    # forecast below 60% of its recent average for any single day.
    # Only applied for venues with 60+ trading days where recent_mean is reliable.
    floor = 0.0
    if n_trade >= 60 and recent > 0:
        floor = recent * 0.60

    fc_out, lo_out, hi_out = [], [], []
    n_clamped = 0
    n_floored = 0
    for f_orig, l_orig, h_orig in zip(fc, lo, hi):
        f_new = max(0, f_orig)
        # Apply ceiling
        if f_new > ceiling:
            f_new = ceiling
            n_clamped += 1
        # Apply floor (after ceiling — never lift above ceiling)
        if floor > 0 and f_new < floor and floor <= ceiling:
            f_new = floor
            n_floored += 1
        scale = (f_new / f_orig) if f_orig > 0 else 1.0
        fc_out.append(f_new)
        lo_out.append(max(0, l_orig * scale))
        hi_out.append(min(ceiling, h_orig * scale))
    return fc_out, lo_out, hi_out, n_clamped + n_floored


# ── HOLT-WINTERS (always available) ────────────────────────────────────────────
def hw_init(series, M):
    L0 = sum(series[:M]) / M
    T0 = (sum(series[M:2*M]) / M - L0) / M if len(series) >= 2 * M else 0
    S = [series[i] - L0 for i in range(M)]
    return L0, T0, S

def hw_forecast(series, M, alpha, beta, gamma, h):
    if len(series) < M * 2:
        flat = mean(series)
        fc = [flat] * h
        return [flat] * len(series), fc, [flat * 0.8] * h, [flat * 1.2] * h, 0

    L, T, S = hw_init(series, M)
    fitted = []
    for i, y in enumerate(series):
        s_idx = i % M
        if i == 0:
            f = L + T + S[s_idx]
            fitted.append(f)
            continue
        L_new = alpha * (y - S[s_idx]) + (1 - alpha) * (L + T)
        T_new = beta * (L_new - L) + (1 - beta) * T
        S[s_idx] = gamma * (y - L_new) + (1 - gamma) * S[s_idx]
        L, T = L_new, T_new
        fitted.append(L + T + S[s_idx])

    residuals = [s - f for s, f in zip(series, fitted)]
    sigma = math.sqrt(sum(r ** 2 for r in residuals) / max(1, len(residuals)))

    fc = []
    for k in range(1, h + 1):
        f = L + k * T + S[(len(series) + k - 1) % M]
        fc.append(max(0, f))
    z = 1.645  # 90% CI
    lo = [max(0, f - z * sigma) for f in fc]
    hi = [max(0, f + z * sigma) for f in fc]
    rmse = math.sqrt(sum(r ** 2 for r in residuals) / max(1, len(residuals)))
    return [max(0, f) for f in fitted], fc, lo, hi, rmse


def optim_hw(series, M):
    best = None
    for a in [0.1, 0.2, 0.3, 0.5]:
        for b in [0.01, 0.05, 0.1]:
            for g in [0.1, 0.2, 0.3]:
                try:
                    _, _, _, _, rmse = hw_forecast(series, M, a, b, g, 1)
                    if best is None or rmse < best[3]:
                        best = (a, b, g, rmse)
                except Exception:
                    continue
    if best is None:
        return 0.2, 0.05, 0.1
    return best[0], best[1], best[2]


# ── PROPHET ────────────────────────────────────────────────────────────────────
def run_prophet(dates, values, h, holiday_dates=None, holiday_names=None, weather_map=None):
    """Run Prophet forecast with holidays + weather regressor.
    holiday_dates: list of YYYY-MM-DD strings (includes historical + future)
    holiday_names: optional list of same length, mapping each date to its holiday name
    weather_map: dict {YYYY-MM-DD: temp_max} covering historical AND forecast horizon
    Returns (fitted, forecast, lower, upper, rmse, label).
    """
    from prophet import Prophet
    df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": [max(0, v) for v in values]})
    n_days = len(dates)
    n_trading = sum(1 for v in values if v > 0)  # routing by trading days, not calendar days

    hols_df = None
    if holiday_dates:
        try:
            names = holiday_names if holiday_names and len(holiday_names) == len(holiday_dates) else ["public_holiday"] * len(holiday_dates)
            hols_df = pd.DataFrame({
                "holiday": names,
                "ds": pd.to_datetime(holiday_dates),
                "lower_window": 0,
                "upper_window": 1,
            })
        except Exception:
            hols_df = None

    # Adaptive params by TRADING days (matches Venu Cast PDF spec, not calendar days).
    # Tighter changepoint_prior across the board — prevents old step-changes (e.g. a
    # 2-year-old plateau drop) from being extrapolated forward as a continuing trend.
    if n_trading < 30:
        params = dict(yearly_seasonality=False, weekly_seasonality=True,
                      changepoint_prior_scale=0.001, seasonality_prior_scale=1, n_changepoints=2)
    elif n_trading < 60:
        params = dict(yearly_seasonality=False, weekly_seasonality=True,
                      changepoint_prior_scale=0.005, seasonality_prior_scale=2, n_changepoints=3)
    elif n_trading < 180:
        params = dict(yearly_seasonality=False, weekly_seasonality=True,
                      changepoint_prior_scale=0.01, seasonality_prior_scale=5, n_changepoints=5)
    elif n_trading < 365:
        params = dict(yearly_seasonality=False, weekly_seasonality=True,
                      changepoint_prior_scale=0.02, seasonality_prior_scale=8, n_changepoints=8)
    else:
        # Full Prophet: yearly seasonality enabled, but trend is TIGHT (0.015 not 0.05)
        # and changepoints concentrated in the recent 80% of history so old step-downs
        # don't dominate. This prevents the Gouger-St-style downward extrapolation.
        params = dict(yearly_seasonality=True, weekly_seasonality=True,
                      changepoint_prior_scale=0.015, seasonality_prior_scale=10, n_changepoints=15,
                      changepoint_range=0.85)

    prophet_kwargs = dict(
        yearly_seasonality=params["yearly_seasonality"],
        weekly_seasonality=params["weekly_seasonality"],
        daily_seasonality=False,
        holidays=hols_df,
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        holidays_prior_scale=10,
        seasonality_mode="multiplicative",
        interval_width=0.90,
        n_changepoints=params["n_changepoints"],
    )
    if "changepoint_range" in params:
        prophet_kwargs["changepoint_range"] = params["changepoint_range"]
    m = Prophet(**prophet_kwargs)

    # Weather: weather_map now has rich structure: {date: {temp_max, precip, wind_max, gusts_max, extreme}}
    use_wx = bool(weather_map) and len(weather_map) > 0
    wx_vars = ['temp_max', 'precip', 'gusts_max', 'extreme']  # 4 regressors
    if use_wx:
        for var in wx_vars:
            # standardize=False for binary "extreme" flag, True for continuous vars
            m.add_regressor(var, standardize=(var != 'extreme'))

        def wx_lookup(d, var, default):
            w = weather_map.get(d)
            if not w:
                return default
            v = w.get(var)
            return v if v is not None else default

        df_dates = df["ds"].dt.strftime("%Y-%m-%d")
        for var in wx_vars:
            default = 0.0 if var in ('precip', 'extreme') else None
            df[var] = df_dates.map(lambda d, v=var, dft=default: wx_lookup(d, v, dft))
            df[var] = pd.to_numeric(df[var], errors="coerce")
            med = df[var].median()
            fill_val = float(med) if not pd.isna(med) else 0.0
            df[var] = df[var].fillna(fill_val)

    m.fit(df)
    future = m.make_future_dataframe(periods=h)

    if use_wx:
        future_dates = future["ds"].dt.strftime("%Y-%m-%d")
        for var in wx_vars:
            default = 0.0 if var in ('precip', 'extreme') else None
            future[var] = future_dates.map(lambda d, v=var, dft=default: wx_lookup(d, v, dft))
            future[var] = pd.to_numeric(future[var], errors="coerce")
            med = df[var].median()
            fill_val = float(med) if not pd.isna(med) else 0.0
            future[var] = future[var].fillna(fill_val)

    fc_df = m.predict(future)

    n = len(dates)
    fi = [max(0, v) for v in fc_df["yhat"].iloc[:n].tolist()]
    fc = [max(0, v) for v in fc_df["yhat"].iloc[n:].tolist()]
    lo = [max(0, v) for v in fc_df["yhat_lower"].iloc[n:].tolist()]
    hi = [max(0, v) for v in fc_df["yhat_upper"].iloc[n:].tolist()]
    resid = [v - f for v, f in zip(values, fi)]
    rmse = math.sqrt(sum(r ** 2 for r in resid) / len(resid)) if resid else 0
    return fi, fc, lo, hi, rmse, "Prophet+Wx+Hol" if use_wx else "Prophet+Hol"


# ── SARIMA ─────────────────────────────────────────────────────────────────────
def run_sarima(dates, values, h):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    series = pd.Series([max(0, v) for v in values], index=pd.to_datetime(dates))
    for order, seas in [((1, 1, 1), (1, 1, 1, 7)), ((1, 1, 1), (0, 1, 1, 7)), ((0, 1, 1), (0, 1, 1, 7))]:
        try:
            mod = SARIMAX(series, order=order, seasonal_order=seas,
                          enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False, maxiter=200)
            fi = [max(0, v) for v in res.fittedvalues.tolist()]
            fcr = res.get_forecast(steps=h)
            fc = [max(0, v) for v in fcr.predicted_mean.tolist()]
            ci = fcr.conf_int(alpha=0.10)
            lo = [max(0, v) for v in ci.iloc[:, 0].tolist()]
            hi = [max(0, v) for v in ci.iloc[:, 1].tolist()]
            rmse = math.sqrt(sum(r ** 2 for r in res.resid.tolist()) / len(res.resid))
            return fi, fc, lo, hi, rmse, f"SARIMA{order}x{seas}"
        except Exception:
            continue
    raise RuntimeError("All SARIMA configurations failed")


def forecast_engine(dates, values, h, holiday_dates=None, holiday_names=None, weather_map=None, analysis=None):
    """Adaptive: DOW-Flat for very new venues, Prophet→SARIMA→HW for mature.
    Hard ceiling applied after model output (sanitise_forecast).
    """
    if analysis is None:
        analysis = analyse_venue_history(dates, values)

    # Sub-21-trading-days venues: bypass Prophet entirely (the Kawana Waters fix)
    if analysis['bypass_prophet']:
        fi, fc, lo, hi, rmse, label = dow_flat_forecast(analysis, h, dates[-1])
        fc, lo, hi, n_clamped = sanitise_forecast(fc, lo, hi, analysis)
        return fi, fc, lo, hi, rmse, f"{label} (new<21d)"

    fi, fc, lo, hi, rmse, model_used = None, None, None, None, 0, None
    try:
        fi, fc, lo, hi, rmse, model_used = run_prophet(
            dates, values, h, holiday_dates=holiday_dates,
            holiday_names=holiday_names, weather_map=weather_map)
    except Exception as e:
        log.info(f"  Prophet failed: {str(e)[:80]} — trying SARIMA")

    if model_used is None:
        try:
            fi, fc, lo, hi, rmse, model_used = run_sarima(dates, values, h)
        except Exception as e:
            log.info(f"  SARIMA failed: {str(e)[:80]} — using Holt-Winters")

    if model_used is None:
        M = 7
        a, b, g = optim_hw(values, M)
        fi, fc, lo, hi, rmse = hw_forecast(values, M, a, b, g, h)
        model_used = "Holt-Winters"

    # Apply hard ceiling — prevents Kawana Waters-style runaway
    fc, lo, hi, n_clamped = sanitise_forecast(fc, lo, hi, analysis)
    if n_clamped:
        model_used = f"{model_used} (clamped {n_clamped}d)"
    return fi, fc, lo, hi, rmse, model_used


# ── MULTIPLIER & ATP LOGIC ─────────────────────────────────────────────────────
def get_multiplier(venue, date_obj, start_anchor=FORECAST_START):
    """Get the 15-month multiplier for venue/date.
    Cycle anchored to start of forecast period; repeats every 15 months."""
    mults_obj = (SESSION_STATE.get('multipliers') or {}).get(venue)
    if not mults_obj:
        return 1.0
    mults = mults_obj.get('multipliers', [1.0] * 15)
    months_from_start = (date_obj.year - start_anchor.year) * 12 + (date_obj.month - start_anchor.month)
    idx = months_from_start % 15
    if 0 <= idx < len(mults):
        return mults[idx]
    return 1.0


def get_atp_growth(venue, date_obj):
    """Get monthly ATP growth percent (0.0 = nil) for venue/date.
    Template is keyed by venue + month-index-from-start (1..15)."""
    growth_obj = SESSION_STATE.get('atp_growth_template') or {}
    venue_growth = growth_obj.get(venue, growth_obj.get('_default', {}))
    months_from_start = (date_obj.year - FORECAST_START.year) * 12 + (date_obj.month - FORECAST_START.month)
    return float(venue_growth.get(str(months_from_start + 1), 0.0))


def forecast_atp_series(venue, atp_history, forecast_dates):
    """Forecast daily ATP for venue across forecast period.
    Uses last known ATP * cumulative monthly growth from template."""
    if not atp_history:
        return [25.0] * len(forecast_dates)

    sorted_items = sorted(atp_history.items())
    last_dates = sorted_items[-30:]
    last_atp = mean([v for _, v in last_dates]) if last_dates else 25.0

    out = []
    current_atp = last_atp
    last_month_key = None
    for date_str in forecast_dates:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        month_key = (d.year, d.month)
        if month_key != last_month_key:
            growth_pct = get_atp_growth(venue, d)
            current_atp = current_atp * (1.0 + growth_pct / 100.0)
            last_month_key = month_key
        out.append(max(0.01, current_atp))
    return out


def cluster_dow_template(venue):
    """Build a Mon-Sun DOW share template averaged across mature cluster peers.
    Returns dict {0..6: daily_share_of_weekly_total} summing to 1.0.
    Used when a brand-new venue has no history of its own.
    """
    loc_bi = SESSION_STATE.get('location_bi') or {}
    servings = SESSION_STATE.get('sales_servings') or {}
    me = loc_bi.get(venue)
    if not me:
        return None
    my_cluster = me.get('cluster')
    peers = [v for v, info in loc_bi.items()
             if v != venue and info.get('cluster') == my_cluster and v in servings]
    if not peers:
        # Fall back to ALL venues
        peers = list(servings.keys())[:10]

    dow_totals = [0.0] * 7
    dow_counts = [0] * 7
    for p in peers:
        series = servings.get(p, {})
        # Use last 90 days of each peer
        sorted_items = sorted(series.items())[-90:]
        for d_str, v in sorted_items:
            if v and v > 0:
                try:
                    dow = datetime.strptime(d_str, "%Y-%m-%d").weekday()
                    dow_totals[dow] += v
                    dow_counts[dow] += 1
                except Exception:
                    continue
    dow_avg = [(dow_totals[i] / dow_counts[i]) if dow_counts[i] > 0 else 0 for i in range(7)]
    weekly_total = sum(dow_avg)
    if weekly_total <= 0:
        return {i: 1 / 7 for i in range(7)}
    return {i: dow_avg[i] / weekly_total for i in range(7)}


def forecast_future_venue_from_template(venue, vdetails, multipliers_obj, holiday_map, weather_map):
    """For venues with no history that open in the future.
    Uses: new_monthly_sales, Month 1-6 ramp %, cluster DOW pattern, 15-month multipliers.
    Returns same shape as forecast_engine.
    """
    new_monthly = (vdetails or {}).get('new_monthly_sales', 0) or 0
    ramp_pcts = (vdetails or {}).get('ramp_up_pcts', []) or []
    opening_date_str = (vdetails or {}).get('opening_date', '') or ''

    try:
        opening_date = datetime.strptime(opening_date_str, "%Y-%m-%d") if opening_date_str else None
    except Exception:
        opening_date = None

    dow_share = cluster_dow_template(venue) or {i: 1 / 7 for i in range(7)}

    # Build daily forecasts from FORECAST_START onward
    daily = {}  # date_str -> base $ (before mults)
    cur = FORECAST_START
    while cur <= FORECAST_END:
        if opening_date and cur < opening_date:
            daily[iso(cur)] = 0.0
        else:
            # months since opening (1-indexed for ramp)
            if opening_date:
                months_since_open = (cur.year - opening_date.year) * 12 + (cur.month - opening_date.month) + 1
            else:
                months_since_open = 99  # treat as fully ramped if no opening date
            # Ramp factor: 1..6 then 100% thereafter (placeholder of 100% if blanks)
            if 1 <= months_since_open <= 6 and months_since_open - 1 < len(ramp_pcts):
                rp = ramp_pcts[months_since_open - 1]
                ramp_factor = (rp / 100.0) if (rp is not None and rp > 1.5) else (rp if rp is not None else 1.0)
            else:
                ramp_factor = 1.0
            # Convert monthly $ to daily $ (using DOW share)
            # Approx: weekly_total = new_monthly / 4.345, daily = weekly_total * dow_share[weekday]
            weekly_total = new_monthly / 4.345
            dow = cur.weekday()
            daily_base = weekly_total * dow_share.get(dow, 1 / 7) * ramp_factor
            daily[iso(cur)] = max(0.0, daily_base)
        cur = cur + timedelta(days=1)

    forecast_dates = sorted(daily.keys())
    fc = [daily[d] for d in forecast_dates]
    # No CI for template-based forecast; use ±20% band
    lo = [v * 0.80 for v in fc]
    hi = [v * 1.20 for v in fc]
    fi = []
    rmse = 0.0
    return fi, fc, lo, hi, rmse, "FutureVenue-Template", forecast_dates


def get_cluster_peers(venue):
    """Return list of peer venues in the same cluster (excluding self)."""
    loc_bi = SESSION_STATE.get('location_bi') or {}
    me = loc_bi.get(venue)
    if not me or not me.get('cluster'):
        return []
    my_cluster = me['cluster']
    peers = []
    for v, info in loc_bi.items():
        if v != venue and info.get('cluster') == my_cluster:
            peers.append(v)
    return peers


# ── PER-VENUE FORECAST ─────────────────────────────────────────────────────────
def forecast_venue(venue, sales_series, atp_history, extra_holiday_dates=None):
    """Generate full forecast for a single venue.
    Routing logic (matches prior Venu Cast model):
      • No history + future opening → template-based ramp forecast
      • < 21 trading days → DOW-flat (Prophet bypass)
      • Otherwise → Prophet/SARIMA/HW + hard ceiling sanitisation
    All branches: state-based weather + holidays + 15-month multipliers + ATP forecast.
    """
    # Resolve venue details / state up front
    venue_details = (SESSION_STATE.get('venue_details') or {}).get(venue)
    multipliers_obj = (SESSION_STATE.get('multipliers') or {}).get(venue)
    state = get_venue_state(venue)
    state_holidays_map = {}
    weather_map = {}
    if state:
        year_range = range(2020, FORECAST_END.year + 1)
        state_holidays_map = load_holidays_for_state(state, year_range)
        weather_map = load_weather_for_state(state, datetime(2020, 1, 1), FORECAST_END)

    # ── Path A: FUTURE VENUE — no history of its own ──
    if not sales_series:
        if venue_details and venue_details.get('new_monthly_sales', 0) > 0:
            log.info(f"  [Future venue] {venue}: opens {venue_details.get('opening_date')} — using ramp template")
            fi, fc, lo, hi, rmse, model_used, dates_window = forecast_future_venue_from_template(
                venue, venue_details, multipliers_obj, state_holidays_map, weather_map)
            # Forecast already in May 2026 - Jun 2027 window; no shifting needed
            fc_window, lo_window, hi_window = fc, lo, hi
            multipliers = []
            fc_with_mult = []
            lo_with_mult = []
            hi_with_mult = []
            for date_str, f, l, hv in zip(dates_window, fc_window, lo_window, hi_window):
                d = datetime.strptime(date_str, "%Y-%m-%d")
                m = get_multiplier(venue, d)
                multipliers.append(m)
                fc_with_mult.append(f * m)
                lo_with_mult.append(l * m)
                hi_with_mult.append(hv * m)
            atp_fc = forecast_atp_series(venue, atp_history, dates_window)
            transactions = [s / a if a > 0 else 0 for s, a in zip(fc_with_mult, atp_fc)]

            def _w(d, key):
                w = weather_map.get(d)
                return None if not w else w.get(key)
            fc_holidays = [state_holidays_map.get(d, '') for d in dates_window]
            return {
                'venue': venue,
                'state': state or '',
                'model': model_used,
                'rmse': round(rmse, 2),
                'forecast_dates': dates_window,
                'sales_forecast': [round(v, 2) for v in fc_with_mult],
                'sales_lower_90': [round(v, 2) for v in lo_with_mult],
                'sales_upper_90': [round(v, 2) for v in hi_with_mult],
                'multipliers_applied': [round(v, 4) for v in multipliers],
                'atp_forecast': [round(v, 4) for v in atp_fc],
                'transaction_forecast': [round(v, 2) for v in transactions],
                'weather_temp_max': [round(_w(d, 'temp_max'), 1) if _w(d, 'temp_max') is not None else None for d in dates_window],
                'weather_precip_mm': [round(_w(d, 'precip'), 2) if _w(d, 'precip') is not None else None for d in dates_window],
                'weather_gusts_kmh': [round(_w(d, 'gusts_max'), 1) if _w(d, 'gusts_max') is not None else None for d in dates_window],
                'weather_extreme': [int(_w(d, 'extreme') or 0) for d in dates_window],
                'holiday_names': fc_holidays,
                'holidays_loaded_count': len(state_holidays_map),
                'weather_days_loaded': len(weather_map),
            }
        return None

    sorted_items = sorted(sales_series.items())
    dates = [d for d, v in sorted_items]
    values = [v for d, v in sorted_items]

    # Drop leading zeros — crucial for venues that opened mid-history
    dates, values = trim_history(dates, values)
    if not dates:
        return None

    last_hist_date = datetime.strptime(dates[-1], "%Y-%m-%d")
    first_hist_date = datetime.strptime(dates[0], "%Y-%m-%d")
    days_until_forecast_start = (FORECAST_START - last_hist_date).days

    if days_until_forecast_start > 0:
        h = days_until_forecast_start + FORECAST_DAYS - 1
    else:
        h = FORECAST_DAYS

    h = max(1, min(900, h))

    # Combine state holidays + any user-supplied holiday dates
    all_holiday_dates = list(state_holidays_map.keys())
    all_holiday_names = [state_holidays_map[d] for d in all_holiday_dates]
    if extra_holiday_dates:
        for d in extra_holiday_dates:
            if d not in state_holidays_map:
                all_holiday_dates.append(d)
                all_holiday_names.append("user_holiday")

    # ── Path B: Analyse history, then route to engine ──
    analysis = analyse_venue_history(dates, values)
    if analysis['n_trading'] < 21:
        log.info(f"  [New venue <21d] {venue}: {analysis['n_trading']} trading days — DOW-flat bypass")
    elif analysis['is_patchy']:
        log.info(f"  [Patchy] {venue}: {analysis['n_trading']} trading days, patchiness {analysis['patchiness']:.0%}")

    try:
        fi, fc, lo, hi, rmse, model_used = forecast_engine(
            dates, values, h,
            holiday_dates=all_holiday_dates,
            holiday_names=all_holiday_names,
            weather_map=weather_map,
            analysis=analysis,
        )
    except Exception as e:
        log.warning(f"  All models failed for {venue}: {e} — using flat mean")
        flat = mean([v for v in values if v > 0] or [0])
        fi = [flat] * len(values)
        fc = [flat] * h
        lo = [flat * 0.8] * h
        hi = [flat * 1.2] * h
        rmse = 0
        model_used = "FlatMean"

    forecast_dates = []
    d = last_hist_date
    for _ in range(h):
        d = d + timedelta(days=1)
        forecast_dates.append(iso(d))

    if days_until_forecast_start > 0:
        start_idx = days_until_forecast_start
    else:
        start_idx = 0
    end_idx = min(len(fc), start_idx + FORECAST_DAYS)

    fc_window = fc[start_idx:end_idx]
    lo_window = lo[start_idx:end_idx]
    hi_window = hi[start_idx:end_idx]
    dates_window = forecast_dates[start_idx:end_idx]

    while len(fc_window) < FORECAST_DAYS:
        d = datetime.strptime(dates_window[-1] if dates_window else iso(FORECAST_START), "%Y-%m-%d") + timedelta(days=1)
        fc_window.append(fc_window[-1] if fc_window else 0)
        lo_window.append(lo_window[-1] if lo_window else 0)
        hi_window.append(hi_window[-1] if hi_window else 0)
        dates_window.append(iso(d))

    multipliers = []
    fc_with_mult = []
    lo_with_mult = []
    hi_with_mult = []
    for date_str, f, l, hv in zip(dates_window, fc_window, lo_window, hi_window):
        d = datetime.strptime(date_str, "%Y-%m-%d")
        m = get_multiplier(venue, d)
        multipliers.append(m)
        fc_with_mult.append(f * m)
        lo_with_mult.append(l * m)
        hi_with_mult.append(hv * m)

    atp_fc = forecast_atp_series(venue, atp_history, dates_window)
    transactions = [s / a if a > 0 else 0 for s, a in zip(fc_with_mult, atp_fc)]

    # Forecast-window weather + holiday metadata for transparency
    def _w(d, key):
        w = weather_map.get(d)
        return None if not w else w.get(key)

    fc_holidays = [state_holidays_map.get(d, '') for d in dates_window]

    return {
        'venue': venue,
        'state': state or '',
        'model': model_used,
        'rmse': round(rmse, 2),
        'forecast_dates': dates_window,
        'sales_forecast': [round(v, 2) for v in fc_with_mult],
        'sales_lower_90': [round(v, 2) for v in lo_with_mult],
        'sales_upper_90': [round(v, 2) for v in hi_with_mult],
        'multipliers_applied': [round(v, 4) for v in multipliers],
        'atp_forecast': [round(v, 4) for v in atp_fc],
        'transaction_forecast': [round(v, 2) for v in transactions],
        'weather_temp_max': [round(_w(d, 'temp_max'), 1) if _w(d, 'temp_max') is not None else None for d in dates_window],
        'weather_precip_mm': [round(_w(d, 'precip'), 2) if _w(d, 'precip') is not None else None for d in dates_window],
        'weather_gusts_kmh': [round(_w(d, 'gusts_max'), 1) if _w(d, 'gusts_max') is not None else None for d in dates_window],
        'weather_extreme': [int(_w(d, 'extreme') or 0) for d in dates_window],
        'holiday_names': fc_holidays,
        'holidays_loaded_count': len(state_holidays_map),
        'weather_days_loaded': len(weather_map),
    }


# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    resp = app.make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools_probe():
    return "", 204


@app.route("/health", methods=["GET"])
def health():
    libs = {}
    for lib in ["prophet", "statsmodels", "numpy", "pandas", "flask"]:
        try:
            __import__(lib)
            libs[lib] = "ok"
        except ImportError:
            libs[lib] = "missing"
    return jsonify({"status": "ok", "libs": libs, "session_loaded": {
        k: (v is not None) for k, v in SESSION_STATE.items()
    }})


@app.route("/upload", methods=["POST"])
def upload():
    """Receive uploaded template files; parse and store in SESSION_STATE."""
    summary = {}
    errors = []

    try:
        if 'sales_history' in request.files:
            f = request.files['sales_history']
            if f.filename:
                parsed = parse_sales_history(f)
                if 'Servings' in parsed:
                    SESSION_STATE['sales_servings'] = parsed['Servings']
                if 'Retail' in parsed:
                    SESSION_STATE['sales_retail'] = parsed['Retail']
                if 'POS Discounts' in parsed:
                    SESSION_STATE['sales_discounts'] = parsed['POS Discounts']
                summary['sales_history'] = {
                    'tabs': list(parsed.keys()),
                    'venues_per_tab': {k: len(v) for k, v in parsed.items()},
                }
    except Exception as e:
        errors.append(f"sales_history: {e}")
        log.exception("sales_history parse error")

    try:
        if 'atp_history' in request.files:
            f = request.files['atp_history']
            if f.filename:
                SESSION_STATE['atp_history'] = parse_atp_history(f)
                summary['atp_history'] = {'venues': len(SESSION_STATE['atp_history'])}
    except Exception as e:
        errors.append(f"atp_history: {e}")
        log.exception("atp_history parse error")

    try:
        if 'location_bi' in request.files:
            f = request.files['location_bi']
            if f.filename:
                SESSION_STATE['location_bi'] = parse_location_bi(f)
                summary['location_bi'] = {'venues': len(SESSION_STATE['location_bi'])}
    except Exception as e:
        errors.append(f"location_bi: {e}")
        log.exception("location_bi parse error")

    try:
        if 'venue_details' in request.files:
            f = request.files['venue_details']
            if f.filename:
                SESSION_STATE['venue_details'] = parse_venue_details(f)
                summary['venue_details'] = {'venues': len(SESSION_STATE['venue_details'])}
    except Exception as e:
        errors.append(f"venue_details: {e}")
        log.exception("venue_details parse error")

    try:
        if 'multipliers' in request.files:
            f = request.files['multipliers']
            if f.filename:
                SESSION_STATE['multipliers'] = parse_multipliers(f)
                summary['multipliers'] = {'venues': len(SESSION_STATE['multipliers'])}
    except Exception as e:
        errors.append(f"multipliers: {e}")
        log.exception("multipliers parse error")

    if SESSION_STATE.get('atp_growth_template') is None:
        all_venues = set()
        if SESSION_STATE.get('multipliers'):
            all_venues.update(SESSION_STATE['multipliers'].keys())
        if SESSION_STATE.get('venue_details'):
            all_venues.update(SESSION_STATE['venue_details'].keys())
        template = {v: {str(m): 0.0 for m in range(1, 16)} for v in all_venues}
        template['_default'] = {str(m): 0.0 for m in range(1, 16)}
        SESSION_STATE['atp_growth_template'] = template
        summary['atp_growth_template'] = {
            'venues': len(template) - 1,
            'months': 15,
            'note': 'Placeholder: 0% growth per month per venue (edit via /atp-growth)',
        }

    return jsonify({
        'status': 'ok' if not errors else 'partial',
        'summary': summary,
        'errors': errors,
        'session_loaded': {k: (v is not None) for k, v in SESSION_STATE.items()},
    })


@app.route("/atp-growth", methods=["GET", "POST"])
def atp_growth():
    """GET: return current growth template. POST: update growth percents."""
    if request.method == "GET":
        return jsonify({
            'template': SESSION_STATE.get('atp_growth_template') or {},
            'note': 'Each value is a monthly % growth (0 = nil). Months are 1..15 from May 2026.',
        })

    body = request.get_json(force=True) or {}
    incoming = body.get('template', {})
    current = SESSION_STATE.get('atp_growth_template') or {}
    for venue, months in incoming.items():
        if venue not in current:
            current[venue] = {str(m): 0.0 for m in range(1, 16)}
        for m, pct in months.items():
            try:
                current[venue][str(m)] = float(pct)
            except Exception:
                pass
    SESSION_STATE['atp_growth_template'] = current
    return jsonify({'status': 'ok', 'template': current})


@app.route("/venues", methods=["GET"])
def venues():
    """Return list of all venues across all loaded templates."""
    all_v = set()
    for key in ['sales_servings', 'atp_history', 'location_bi', 'venue_details', 'multipliers']:
        d = SESSION_STATE.get(key)
        if d:
            all_v.update(d.keys())
    venue_list = sorted(all_v)
    enriched = []
    for v in venue_list:
        info = {'venue': v}
        if SESSION_STATE.get('location_bi') and v in SESSION_STATE['location_bi']:
            info.update(SESSION_STATE['location_bi'][v])
        if SESSION_STATE.get('multipliers') and v in SESSION_STATE['multipliers']:
            info['multipliers'] = SESSION_STATE['multipliers'][v]['multipliers']
        if SESSION_STATE.get('venue_details') and v in SESSION_STATE['venue_details']:
            info['details'] = SESSION_STATE['venue_details'][v]
        enriched.append(info)
    return jsonify({'count': len(enriched), 'venues': enriched})


@app.route("/generate", methods=["POST"])
def generate():
    """Run forecasts across all venues for May 2026 - June 2027."""
    body = request.get_json(silent=True) or {}
    target_venues = body.get('venues', None)
    holiday_dates = body.get('holiday_dates', [])

    servings = SESSION_STATE.get('sales_servings') or {}
    retail = SESSION_STATE.get('sales_retail') or {}
    discounts = SESSION_STATE.get('sales_discounts') or {}
    atp_hist = SESSION_STATE.get('atp_history') or {}

    if not servings:
        return jsonify({'error': 'No sales history loaded. Upload 01_Sales_History_Template.xlsx first.'}), 400

    # Future venues: in venue_details with new_monthly_sales > 0, no sales history
    venue_details = SESSION_STATE.get('venue_details') or {}
    future_venues = [
        v for v, d in venue_details.items()
        if v not in servings and (d.get('new_monthly_sales', 0) or 0) > 0
    ]

    if target_venues:
        venue_list = [v for v in target_venues if v in servings or v in future_venues]
    else:
        venue_list = sorted(set(servings.keys()) | set(future_venues))
    log.info(f"  Existing venues: {len(servings)}, future venues with templates: {len(future_venues)}")

    log.info(f"Generating forecasts for {len(venue_list)} venues, {FORECAST_DAYS} days")

    # Pre-warm caches: load weather + holidays for all distinct states first
    distinct_states = set()
    for v in venue_list:
        s = get_venue_state(v)
        if s:
            distinct_states.add(s)
    log.info(f"  Pre-loading weather + holidays for states: {sorted(distinct_states)}")
    year_range = range(2020, FORECAST_END.year + 1)
    for s in distinct_states:
        load_holidays_for_state(s, year_range)
        load_weather_for_state(s, datetime(2020, 5, 1), FORECAST_END)

    results = []
    state_models_used = {}
    for i, venue in enumerate(venue_list, 1):
        log.info(f"[{i}/{len(venue_list)}] {venue}")
        try:
            servings_result = forecast_venue(venue, servings.get(venue, {}), atp_hist.get(venue, {}), holiday_dates)
            if not servings_result:
                continue

            retail_result = None
            if venue in retail:
                retail_result = forecast_venue(venue, retail[venue], {}, holiday_dates)

            discounts_result = None
            if venue in discounts:
                discounts_result = forecast_venue(venue, discounts[venue], {}, holiday_dates)

            combined = {
                'venue': venue,
                'state': servings_result.get('state', ''),
                'forecast_dates': servings_result['forecast_dates'],
                'servings': {
                    'model': servings_result['model'],
                    'rmse': servings_result['rmse'],
                    'sales': servings_result['sales_forecast'],
                    'lower_90': servings_result['sales_lower_90'],
                    'upper_90': servings_result['sales_upper_90'],
                    'multipliers': servings_result['multipliers_applied'],
                    'atp_forecast': servings_result['atp_forecast'],
                    'transactions': servings_result['transaction_forecast'],
                },
                'weather_temp_max': servings_result.get('weather_temp_max', []),
                'weather_precip_mm': servings_result.get('weather_precip_mm', []),
                'weather_gusts_kmh': servings_result.get('weather_gusts_kmh', []),
                'weather_extreme': servings_result.get('weather_extreme', []),
                'holiday_names': servings_result.get('holiday_names', []),
                'data_sources': {
                    'state_holidays_loaded': servings_result.get('holidays_loaded_count', 0),
                    'weather_days_loaded': servings_result.get('weather_days_loaded', 0),
                },
            }
            if retail_result:
                combined['retail'] = {
                    'model': retail_result['model'],
                    'sales': retail_result['sales_forecast'],
                }
            if discounts_result:
                # Negate POS discounts (discounts reduce revenue)
                combined['discounts'] = {
                    'model': discounts_result['model'],
                    'sales': [round(-abs(v), 2) for v in discounts_result['sales_forecast']],
                }
            if SESSION_STATE.get('location_bi') and venue in SESSION_STATE['location_bi']:
                combined['cluster_info'] = SESSION_STATE['location_bi'][venue]
                combined['cluster_peers'] = get_cluster_peers(venue)

            results.append(combined)
        except Exception as e:
            log.exception(f"Failed for {venue}")
            results.append({'venue': venue, 'error': str(e)})

    SESSION_STATE['last_results'] = results

    summary = {
        'total_venues': len(results),
        'successful': sum(1 for r in results if 'error' not in r),
        'failed': sum(1 for r in results if 'error' in r),
        'forecast_period': f"{iso(FORECAST_START)} to {iso(FORECAST_END)}",
        'forecast_days': FORECAST_DAYS,
        'models_used': {},
        'states_processed': sorted(distinct_states),
        'weather_cache_keys': len(WEATHER_CACHE),
        'holiday_cache_keys': len(HOLIDAY_CACHE),
    }
    for r in results:
        if 'servings' in r:
            mdl = r['servings']['model']
            summary['models_used'][mdl] = summary['models_used'].get(mdl, 0) + 1

    return jsonify({'status': 'ok', 'summary': summary, 'results': results})


def _fy_label(date_obj):
    """Australian financial year: Jul Y -> Jun Y+1 = FY (Y+1).
    e.g. 2025-07-01 .. 2026-06-30 = FY 2026."""
    return f"FY {date_obj.year + 1}" if date_obj.month >= 7 else f"FY {date_obj.year}"


# Actuals export start: Jul 2025 (start of FY 2026); end at FORECAST_START - 1
ACTUALS_START = datetime(2025, 7, 1)


@app.route("/download-csv", methods=["GET"])
def download_csv():
    """Export results as CSV with both actuals (Jul 2025 - Apr 2026) and
    forecasts (May 2026 - Jun 2027) so FY 2026 vs FY 2027 can be compared.
    POS discounts are negative. Weather includes temp, rain, gusts and extreme-event flag.
    """
    results = SESSION_STATE.get('last_results')
    if not results:
        return jsonify({'error': 'No results. Run /generate first.'}), 400

    servings = SESSION_STATE.get('sales_servings') or {}
    retail = SESSION_STATE.get('sales_retail') or {}
    discounts = SESSION_STATE.get('sales_discounts') or {}
    atp_hist = SESSION_STATE.get('atp_history') or {}

    rows = []

    for r in results:
        if 'error' in r or 'servings' not in r:
            continue
        venue = r['venue']
        state = r.get('state', '')
        cluster = r.get('cluster_info', {}).get('cluster', '')
        cluster_type = r.get('cluster_info', {}).get('cluster_type', '')
        model_label = r['servings']['model']

        # Get state weather + holidays from cache (used for both actuals & forecasts)
        weather_map = {}
        holiday_map = {}
        if state:
            for key, m in WEATHER_CACHE.items():
                if key.startswith(state + "_"):
                    weather_map.update(m)
            for key, m in HOLIDAY_CACHE.items():
                if key.startswith(state + "_"):
                    holiday_map.update(m)

        def wx_row(date_str):
            w = weather_map.get(date_str) or {}
            return {
                'weather_temp_max': round(w['temp_max'], 1) if w.get('temp_max') is not None else None,
                'weather_precip_mm': round(w['precip'], 2) if w.get('precip') is not None else None,
                'weather_gusts_kmh': round(w['gusts_max'], 1) if w.get('gusts_max') is not None else None,
                'weather_extreme_event': int(w.get('extreme') or 0),
                'public_holiday': holiday_map.get(date_str, ''),
            }

        # ── 1) ACTUALS: Jul 2025 → day before FORECAST_START ──
        venue_serv_actual = servings.get(venue, {})
        venue_retail_actual = retail.get(venue, {})
        venue_disc_actual = discounts.get(venue, {})
        venue_atp_actual = atp_hist.get(venue, {})

        cur = ACTUALS_START
        last_actual_day = FORECAST_START - timedelta(days=1)
        while cur <= last_actual_day:
            ds = iso(cur)
            serv_v = venue_serv_actual.get(ds)
            retail_v = venue_retail_actual.get(ds)
            disc_v = venue_disc_actual.get(ds)
            atp_v = venue_atp_actual.get(ds)
            # Only emit row if at least one actual value exists
            if serv_v is not None or retail_v is not None or disc_v is not None:
                transactions_actual = None
                if serv_v is not None and atp_v is not None and atp_v > 0:
                    transactions_actual = round(serv_v / atp_v, 2)
                wxr = wx_row(ds)
                rows.append({
                    'venue': venue,
                    'state': state,
                    'date': ds,
                    'fy': _fy_label(cur),
                    'type': 'actual',
                    'servings_sales': round(serv_v, 2) if serv_v is not None else None,
                    'servings_lower_90': None,
                    'servings_upper_90': None,
                    'multiplier': None,
                    'atp': round(atp_v, 4) if atp_v is not None else None,
                    'transactions': transactions_actual,
                    'retail_sales': round(retail_v, 2) if retail_v is not None else None,
                    'pos_discounts': round(-abs(disc_v), 2) if disc_v is not None else None,
                    **wxr,
                    'cluster': cluster,
                    'cluster_type': cluster_type,
                    'model': '',
                })
            cur = cur + timedelta(days=1)

        # ── 2) FORECASTS: May 2026 → Jun 2027 ──
        wx_t = r.get('weather_temp_max', [])
        wx_p = r.get('weather_precip_mm', [])
        wx_g = r.get('weather_gusts_kmh', [])
        wx_e = r.get('weather_extreme', [])
        hols = r.get('holiday_names', [])
        retail_fc = r.get('retail', {}).get('sales', [])
        disc_fc = r.get('discounts', {}).get('sales', [])

        for i, date_str in enumerate(r['forecast_dates']):
            d_obj = datetime.strptime(date_str, "%Y-%m-%d")
            rows.append({
                'venue': venue,
                'state': state,
                'date': date_str,
                'fy': _fy_label(d_obj),
                'type': 'forecast',
                'servings_sales': r['servings']['sales'][i],
                'servings_lower_90': r['servings']['lower_90'][i],
                'servings_upper_90': r['servings']['upper_90'][i],
                'multiplier': r['servings']['multipliers'][i],
                'atp': r['servings']['atp_forecast'][i],
                'transactions': r['servings']['transactions'][i],
                'retail_sales': retail_fc[i] if i < len(retail_fc) else None,
                'pos_discounts': disc_fc[i] if i < len(disc_fc) else None,  # already negated upstream
                'weather_temp_max': wx_t[i] if i < len(wx_t) else None,
                'weather_precip_mm': wx_p[i] if i < len(wx_p) else None,
                'weather_gusts_kmh': wx_g[i] if i < len(wx_g) else None,
                'weather_extreme_event': wx_e[i] if i < len(wx_e) else 0,
                'public_holiday': hols[i] if i < len(hols) else '',
                'cluster': cluster,
                'cluster_type': cluster_type,
                'model': model_label,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(['venue', 'date']).reset_index(drop=True)
    out_path = os.path.join(RESULTS_DIR, f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(out_path, index=False)
    return send_file(out_path, as_attachment=True, download_name=os.path.basename(out_path))


def _venue_movement_comment(venue, cohort, opening_date_str, fy26, fy27, fy27_base,
                            growth_pct, base_growth_pct, multiplier_lift_pct,
                            model_label, extreme_fy26, extreme_fy27):
    """Generate a short plain-English comment explaining the movement from FY26 to FY27."""
    parts = []

    # Opening / cohort context
    if cohort == 'FY27 New Opening':
        parts.append(f"New venue opening {opening_date_str} — forecast from template.")
        return ' '.join(parts)
    if cohort == 'FY26 New Opening':
        parts.append(f"Opened {opening_date_str} during FY26 — partial-year FY26 base inflates apparent growth.")

    # Growth band
    if growth_pct is None:
        parts.append("No FY26 base for comparison.")
    else:
        if growth_pct > 25:
            band = "Strong growth"
        elif growth_pct > 10:
            band = "Positive growth"
        elif growth_pct > 2:
            band = "Modest growth"
        elif growth_pct > -2:
            band = "Flat"
        elif growth_pct > -10:
            band = "Mild decline"
        else:
            band = "Significant decline"
        parts.append(f"{band} ({growth_pct:+.1f}%).")

        # Attribute the growth: base vs multiplier
        if base_growth_pct is not None and multiplier_lift_pct is not None:
            if abs(multiplier_lift_pct) >= 0.5:
                parts.append(f"Prophet trend {base_growth_pct:+.1f}%, multipliers add {multiplier_lift_pct:+.1f}pp.")
            else:
                parts.append(f"Prophet trend {base_growth_pct:+.1f}% (multipliers ~flat).")

    # Model / clamp note
    if 'clamped' in (model_label or ''):
        import re
        m = re.search(r'clamped (\d+)d', model_label)
        if m:
            n = int(m.group(1))
            if n > 30:
                parts.append(f"⚠️ {n} days clamped by ceiling/floor — Prophet's trend was constrained.")
    if 'DOW-Flat' in (model_label or ''):
        parts.append("Limited history → DOW-flat forecast (no trend extrapolation).")

    # Weather event note
    if extreme_fy26 > 2 or extreme_fy27 > 2:
        parts.append(f"⚠️ Extreme weather: {extreme_fy26} day(s) FY26, {extreme_fy27} day(s) FY27.")

    return ' '.join(parts)


def _classify_cohort(venue):
    """Return cohort label based on opening_date in venue_details.
    Cohorts:
      'FY26 Existing'      → opened on or before 2025-06-30 (full FY26 trading expected)
      'FY26 New Opening'   → opened during FY26 (2025-07-01 → 2026-06-30)
      'FY27 New Opening'   → opens during FY27 (2026-07-01 → 2027-06-30)
      'Other'              → unknown / no opening date
    """
    details = (SESSION_STATE.get('venue_details') or {}).get(venue) or {}
    opening = details.get('opening_date', '')
    if not opening:
        return 'Other'
    try:
        d = datetime.strptime(opening[:10], "%Y-%m-%d")
    except Exception:
        return 'Other'
    if d <= datetime(2025, 6, 30):
        return 'FY26 Existing'
    if d <= datetime(2026, 6, 30):
        return 'FY26 New Opening'
    if d <= datetime(2027, 6, 30):
        return 'FY27 New Opening'
    return 'Other'


def _venue_filter_matches(venue, state=None, cluster=None, cohort=None, venue_filter=None):
    """Return True if venue matches the supplied filters."""
    if venue_filter and venue_filter != 'all' and venue != venue_filter:
        return False
    venue_state = get_venue_state(venue) or ''
    if state and state != 'all' and venue_state != state:
        return False
    loc_info = (SESSION_STATE.get('location_bi') or {}).get(venue) or {}
    if cluster and cluster != 'all':
        if (loc_info.get('cluster') or '') != cluster:
            return False
    if cohort and cohort != 'all':
        if _classify_cohort(venue) != cohort:
            return False
    return True


@app.route("/sales-detail", methods=["GET"])
def sales_detail():
    """Aggregated FY26 vs FY27 sales detail with KPIs + monthly chart series.
    Filters (query params): state, cluster, cohort, venue.
    FY26 = Jul 2025 - Jun 2026 (actuals + May/Jun 2026 completion forecast)
    FY27 = Jul 2026 - Jun 2027 (full forecast)
    LFL  = same-store: only venues with FY26 trading history.
    """
    state = request.args.get('state', 'all')
    cluster = request.args.get('cluster', 'all')
    cohort = request.args.get('cohort', 'all')
    venue_q = request.args.get('venue', 'all')
    sales_type = request.args.get('sales_type', 'servings')   # servings | retail | pos_discounts | all
    if sales_type not in ('servings', 'retail', 'pos_discounts', 'all'):
        sales_type = 'servings'
    apply_guards = request.args.get('guardrails', 'on').lower() in ('on', '1', 'true', 'yes')

    results = SESSION_STATE.get('last_results') or []
    if not results:
        return jsonify({'error': 'No forecast results. Run /generate first.'}), 400

    servings_actual = SESSION_STATE.get('sales_servings') or {}
    retail_actual = SESSION_STATE.get('sales_retail') or {}
    discounts_actual = SESSION_STATE.get('sales_discounts') or {}

    def actuals_for(venue):
        """Return {date_str: $value} for the selected sales_type, applying the
        POS-discount sign convention (negative)."""
        if sales_type == 'servings':
            return servings_actual.get(venue, {})
        if sales_type == 'retail':
            return retail_actual.get(venue, {})
        if sales_type == 'pos_discounts':
            return {d: -abs(v) for d, v in discounts_actual.get(venue, {}).items()}
        # 'all' = servings + retail - |discounts| (net revenue)
        s = servings_actual.get(venue, {})
        r_ = retail_actual.get(venue, {})
        d = discounts_actual.get(venue, {})
        all_dates = set(s.keys()) | set(r_.keys()) | set(d.keys())
        return {dt: s.get(dt, 0) + r_.get(dt, 0) - abs(d.get(dt, 0)) for dt in all_dates}

    def forecast_for(r):
        """Return list of $values aligned to r['forecast_dates'].
        Resolution priority: active scenario override → baseline guardrail → raw Prophet.
        """
        dates = r['forecast_dates']
        n = len(dates)
        serv, _override, _note = _resolve_venue_sales(r, apply_guardrails=apply_guards)
        ret = r.get('retail', {}).get('sales', [0.0] * n)
        disc = r.get('discounts', {}).get('sales', [0.0] * n)   # already negated upstream
        # Pad to consistent length
        while len(ret) < n:
            ret.append(0.0)
        while len(disc) < n:
            disc.append(0.0)
        if sales_type == 'servings':
            return serv
        if sales_type == 'retail':
            return ret
        if sales_type == 'pos_discounts':
            return disc
        # 'all' = net (discounts already negative)
        return [serv[i] + ret[i] + disc[i] for i in range(n)]

    def forecast_base_for(r):
        """Return un-multiplied (Prophet base) forecast — guardrailed sales / multiplier per day.
        Uses the same scenario/guardrail resolution as forecast_for so the "Prophet base growth"
        attribution panel is consistent with the dashboard.
        """
        dates = r['forecast_dates']
        n = len(dates)
        serv_resolved, _o, _n = _resolve_venue_sales(r, apply_guardrails=apply_guards)
        mults = r['servings']['multipliers']
        base_serv = [serv_resolved[i] / mults[i] if i < len(mults) and mults[i] > 0 else serv_resolved[i] for i in range(n)]
        ret = r.get('retail', {}).get('sales', [0.0] * n)
        disc = r.get('discounts', {}).get('sales', [0.0] * n)
        while len(ret) < n: ret.append(0.0)
        while len(disc) < n: disc.append(0.0)
        if sales_type == 'servings':
            return base_serv
        if sales_type == 'retail':
            return ret
        if sales_type == 'pos_discounts':
            return disc
        return [base_serv[i] + ret[i] + disc[i] for i in range(n)]

    FY26_START = datetime(2025, 7, 1)
    FY26_END = datetime(2026, 6, 30)
    FY27_START = datetime(2026, 7, 1)
    FY27_END = datetime(2027, 6, 30)
    LAST_ACTUAL = datetime(2026, 4, 30)   # Sales-history sheet ends here

    fy_labels = [f"{m:02d}" for m in [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]]
    month_labels = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

    monthly_fy26 = {m: 0.0 for m in range(1, 13)}
    monthly_fy27 = {m: 0.0 for m in range(1, 13)}
    monthly_lfl_fy26 = {m: 0.0 for m in range(1, 13)}   # LFL FY26 = same-store FY26 (actuals + completion fc)
    monthly_lfl_fy27 = {m: 0.0 for m in range(1, 13)}   # LFL FY27 = same-store FY27 forecast
    monthly_fy27_base = {m: 0.0 for m in range(1, 13)}  # FY27 BEFORE multiplier (Prophet base)
    monthly_lfl_fy27_base = {m: 0.0 for m in range(1, 13)}

    fy26_total = 0.0
    fy27_total = 0.0
    fy26_completion = 0.0   # May+Jun 2026 forecast
    fy26_completion_base = 0.0
    lfl_fy26_total = 0.0
    lfl_fy27_total = 0.0
    fy27_base_total = 0.0
    lfl_fy27_base_total = 0.0

    venues_processed = 0
    venues_fy26_existing = 0

    available_filters = {
        'states': set(),
        'clusters': set(),
        'cohorts': set(),
        'venues': set(),
    }

    # Track which states are inside the filter so we can scope weather/holiday events
    states_in_filter = set()

    for r in results:
        if 'error' in r or 'servings' not in r:
            continue
        venue = r['venue']

        # Collect all filter-option candidates from the FULL result set
        v_state = get_venue_state(venue) or ''
        loc_info = (SESSION_STATE.get('location_bi') or {}).get(venue) or {}
        v_cluster = loc_info.get('cluster') or ''
        v_cohort = _classify_cohort(venue)
        if v_state:
            available_filters['states'].add(v_state)
        if v_cluster:
            available_filters['clusters'].add(v_cluster)
        if v_cohort:
            available_filters['cohorts'].add(v_cohort)
        available_filters['venues'].add(venue)

        # Apply filters
        if not _venue_filter_matches(venue, state, cluster, cohort, venue_q):
            continue
        venues_processed += 1
        if v_state:
            states_in_filter.add(v_state)
        is_fy26_existing = v_cohort == 'FY26 Existing'
        if is_fy26_existing:
            venues_fy26_existing += 1

        # FY26 actuals: Jul 2025 -> Apr 2026 from the chosen sales_type
        venue_actuals = actuals_for(venue)
        venue_fy26 = 0.0
        for d_str, sales_val in venue_actuals.items():
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d")
            except Exception:
                continue
            if FY26_START <= d <= LAST_ACTUAL and sales_val is not None:
                monthly_fy26[d.month] += sales_val
                fy26_total += sales_val
                venue_fy26 += sales_val
                if is_fy26_existing:
                    monthly_lfl_fy26[d.month] += sales_val
                    lfl_fy26_total += sales_val

        # Completion forecast (May/Jun 2026) + FY27 forecast — read from results
        fc_values = forecast_for(r)
        fc_base_values = forecast_base_for(r)
        for d_str, sales_val, base_val in zip(r['forecast_dates'], fc_values, fc_base_values):
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d")
            except Exception:
                continue
            if FY26_START <= d <= FY26_END and d > LAST_ACTUAL:
                # Completion forecast portion of FY26 (May/Jun 2026)
                monthly_fy26[d.month] += sales_val
                fy26_total += sales_val
                fy26_completion += sales_val
                fy26_completion_base += base_val
                venue_fy26 += sales_val
                if is_fy26_existing:
                    monthly_lfl_fy26[d.month] += sales_val
                    lfl_fy26_total += sales_val
            elif FY27_START <= d <= FY27_END:
                monthly_fy27[d.month] += sales_val
                monthly_fy27_base[d.month] += base_val
                fy27_total += sales_val
                fy27_base_total += base_val
                if is_fy26_existing:
                    monthly_lfl_fy27[d.month] += sales_val
                    monthly_lfl_fy27_base[d.month] += base_val
                    lfl_fy27_total += sales_val
                    lfl_fy27_base_total += base_val

    # ── EVENT AGGREGATION: public holidays + extreme weather days per month/FY ──
    # Scoped to states that survived the venue filter. Counts unique (date,state) tuples
    # so a single QLD cyclone day doesn't get multiplied by the number of QLD venues.
    holidays_fy26 = {m: {} for m in range(1, 13)}   # month -> {holiday_name: [states]}
    holidays_fy27 = {m: {} for m in range(1, 13)}
    extreme_fy26 = {m: {} for m in range(1, 13)}    # month -> {date_str: [states]}
    extreme_fy27 = {m: {} for m in range(1, 13)}

    for s in states_in_filter:
        # Holidays
        for key, hmap in HOLIDAY_CACHE.items():
            if not key.startswith(s + '_'):
                continue
            for d_str, name in hmap.items():
                try:
                    d = datetime.strptime(d_str, "%Y-%m-%d")
                except Exception:
                    continue
                if FY26_START <= d <= FY26_END:
                    holidays_fy26[d.month].setdefault(name, []).append(s)
                elif FY27_START <= d <= FY27_END:
                    holidays_fy27[d.month].setdefault(name, []).append(s)
        # Extreme weather
        for key, wmap in WEATHER_CACHE.items():
            if not key.startswith(s + '_'):
                continue
            for d_str, w in wmap.items():
                if not isinstance(w, dict) or not w.get('extreme'):
                    continue
                try:
                    d = datetime.strptime(d_str, "%Y-%m-%d")
                except Exception:
                    continue
                gusts = w.get('gusts_max')
                precip = w.get('precip')
                detail = []
                if gusts and gusts >= EXTREME_GUSTS_KMH:
                    detail.append(f"gusts {gusts:.0f} km/h")
                if precip and precip >= EXTREME_PRECIP_MM:
                    detail.append(f"rain {precip:.0f} mm")
                detail_str = f"{s}: " + ", ".join(detail) if detail else s
                if FY26_START <= d <= FY26_END:
                    extreme_fy26[d.month].setdefault(d_str, []).append(detail_str)
                elif FY27_START <= d <= FY27_END:
                    extreme_fy27[d.month].setdefault(d_str, []).append(detail_str)

    # Order months as financial year (Jul..Jun)
    fy_order = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    monthly = []
    for i, m in enumerate(fy_order):
        fy26_m = monthly_fy26.get(m, 0)
        fy27_m = monthly_fy27.get(m, 0)
        fy27_base_m = monthly_fy27_base.get(m, 0)
        lfl_fy26_m = monthly_lfl_fy26.get(m, 0)
        lfl_fy27_m = monthly_lfl_fy27.get(m, 0)
        lfl_fy27_base_m = monthly_lfl_fy27_base.get(m, 0)
        growth_pct = ((fy27_m - fy26_m) / fy26_m * 100) if fy26_m > 0 else None
        base_growth_pct = ((fy27_base_m - fy26_m) / fy26_m * 100) if fy26_m > 0 else None
        lfl_growth_pct = ((lfl_fy27_m - lfl_fy26_m) / lfl_fy26_m * 100) if lfl_fy26_m > 0 else None
        lfl_base_growth_pct = ((lfl_fy27_base_m - lfl_fy26_m) / lfl_fy26_m * 100) if lfl_fy26_m > 0 else None
        # Multiplier contribution = total - base (in percentage points)
        mult_contrib = (growth_pct - base_growth_pct) if (growth_pct is not None and base_growth_pct is not None) else None
        monthly.append({
            'month_idx': m,
            'month_label': month_labels[i],
            'fy26_comparator': round(fy26_m, 2),
            'fy27_forecast': round(fy27_m, 2),
            'fy27_base': round(fy27_base_m, 2),
            'lfl_fy26': round(lfl_fy26_m, 2),
            'lfl_fy27': round(lfl_fy27_m, 2),
            'lfl_fy27_base': round(lfl_fy27_base_m, 2),
            'growth_pct': round(growth_pct, 2) if growth_pct is not None else None,
            'base_growth_pct': round(base_growth_pct, 2) if base_growth_pct is not None else None,
            'multiplier_lift_pct': round(mult_contrib, 2) if mult_contrib is not None else None,
            'lfl_growth_pct': round(lfl_growth_pct, 2) if lfl_growth_pct is not None else None,
            'lfl_base_growth_pct': round(lfl_base_growth_pct, 2) if lfl_base_growth_pct is not None else None,
            'fy26_holidays': sorted(holidays_fy26[m].keys()),
            'fy27_holidays': sorted(holidays_fy27[m].keys()),
            'fy26_holiday_count': len(holidays_fy26[m]),
            'fy27_holiday_count': len(holidays_fy27[m]),
            'fy26_extreme_days': sorted(extreme_fy26[m].keys()),
            'fy27_extreme_days': sorted(extreme_fy27[m].keys()),
            'fy26_extreme_count': len(extreme_fy26[m]),
            'fy27_extreme_count': len(extreme_fy27[m]),
            'fy26_extreme_detail': [
                {'date': d, 'detail': ' / '.join(sorted(set(extreme_fy26[m][d])))}
                for d in sorted(extreme_fy26[m].keys())
            ],
            'fy27_extreme_detail': [
                {'date': d, 'detail': ' / '.join(sorted(set(extreme_fy27[m][d])))}
                for d in sorted(extreme_fy27[m].keys())
            ],
        })

    total_growth_pct = ((fy27_total - fy26_total) / fy26_total * 100) if fy26_total > 0 else 0
    lfl_growth_pct = ((lfl_fy27_total - lfl_fy26_total) / lfl_fy26_total * 100) if lfl_fy26_total > 0 else 0
    # Prophet "natural" growth — what the forecast would be WITHOUT the 15-month multiplier applied
    base_growth_pct = ((fy27_base_total - fy26_total) / fy26_total * 100) if fy26_total > 0 else 0
    lfl_base_growth_pct = ((lfl_fy27_base_total - lfl_fy26_total) / lfl_fy26_total * 100) if lfl_fy26_total > 0 else 0
    # Lift = the points the multiplier added on top of the base
    multiplier_lift_pct = total_growth_pct - base_growth_pct
    lfl_multiplier_lift_pct = lfl_growth_pct - lfl_base_growth_pct

    return jsonify({
        'summary': {
            'fy27_forecast': round(fy27_total, 2),
            'fy27_base_forecast': round(fy27_base_total, 2),
            'fy26_comparator': round(fy26_total, 2),
            'fy26_completion_forecast': round(fy26_completion, 2),
            'lfl_fy26': round(lfl_fy26_total, 2),
            'lfl_fy27': round(lfl_fy27_total, 2),
            'lfl_fy27_base': round(lfl_fy27_base_total, 2),
            'total_growth_pct': round(total_growth_pct, 1),
            'base_growth_pct': round(base_growth_pct, 1),
            'multiplier_lift_pct': round(multiplier_lift_pct, 1),
            'lfl_growth_pct': round(lfl_growth_pct, 1),
            'lfl_base_growth_pct': round(lfl_base_growth_pct, 1),
            'lfl_multiplier_lift_pct': round(lfl_multiplier_lift_pct, 1),
            'venues_in_filter': venues_processed,
            'venues_fy26_existing_in_filter': venues_fy26_existing,
        },
        'monthly': monthly,
        'filters_applied': {'state': state, 'cluster': cluster, 'cohort': cohort, 'venue': venue_q, 'sales_type': sales_type},
        'filters_available': {
            'states': sorted(s for s in available_filters['states'] if s),
            'clusters': sorted(c for c in available_filters['clusters'] if c),
            'cohorts': sorted(c for c in available_filters['cohorts'] if c),
            'venues': sorted(available_filters['venues']),
        },
    })


@app.route("/venues-detail", methods=["GET"])
def venues_detail():
    """Per-venue FY26 vs FY27 detail with state/cluster grouping and movement commentary.
    Same filter params as /sales-detail: state, cluster, cohort, venue.
    """
    state = request.args.get('state', 'all')
    cluster = request.args.get('cluster', 'all')
    cohort = request.args.get('cohort', 'all')
    venue_q = request.args.get('venue', 'all')
    apply_guards = request.args.get('guardrails', 'on').lower() in ('on', '1', 'true', 'yes')

    results = SESSION_STATE.get('last_results') or []
    if not results:
        return jsonify({'error': 'No forecast results. Run /generate first.'}), 400

    servings_actual = SESSION_STATE.get('sales_servings') or {}
    location_bi = SESSION_STATE.get('location_bi') or {}
    venue_details = SESSION_STATE.get('venue_details') or {}

    FY25_START = datetime(2024, 7, 1)
    FY25_END = datetime(2025, 6, 30)
    FY26_START = datetime(2025, 7, 1)
    FY26_END = datetime(2026, 6, 30)
    FY27_START = datetime(2026, 7, 1)
    FY27_END = datetime(2027, 6, 30)
    LAST_ACTUAL = datetime(2026, 4, 30)
    FY_MONTH_ORDER = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]

    venue_rows = []

    for r in results:
        if 'error' in r or 'servings' not in r:
            continue
        venue = r['venue']
        if not _venue_filter_matches(venue, state, cluster, cohort, venue_q):
            continue

        v_state = get_venue_state(venue) or '—'
        loc = location_bi.get(venue, {}) or {}
        v_cluster = loc.get('cluster') or '—'
        v_cohort = _classify_cohort(venue)
        details = venue_details.get(venue, {}) or {}
        opening_date_str = details.get('opening_date', '') or ''

        # Monthly buckets for sparkline (Jul-Jun order)
        monthly_fy25 = {m: 0.0 for m in range(1, 13)}
        monthly_fy26 = {m: 0.0 for m in range(1, 13)}
        monthly_fy27 = {m: 0.0 for m in range(1, 13)}

        # FY25 + FY26: actuals from servings
        venue_actuals = servings_actual.get(venue, {})
        fy26 = 0.0
        fy26_trading_days = 0
        for d_str, sales_val in venue_actuals.items():
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d")
            except Exception:
                continue
            if not sales_val:
                continue
            if FY25_START <= d <= FY25_END:
                monthly_fy25[d.month] += sales_val
            elif FY26_START <= d <= LAST_ACTUAL:
                fy26 += sales_val
                monthly_fy26[d.month] += sales_val
                if sales_val > 0:
                    fy26_trading_days += 1

        # Walk the forecast for completion FC + FY27 totals (with and without multiplier)
        fy27 = 0.0
        fy27_base = 0.0
        fy27_days = 0
        atp_values = []
        transactions_total = 0.0

        # Resolve sales: scenario override (if any), else guardrails (if on), else raw Prophet
        sales, scenario_override, guardrail_note = _resolve_venue_sales(r, apply_guardrails=apply_guards)
        atp_arr = r['servings']['atp_forecast']
        trans_arr = r['servings']['transactions']
        mults = r['servings']['multipliers']
        for i, d_str in enumerate(r['forecast_dates']):
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d")
            except Exception:
                continue
            s_val = sales[i] if i < len(sales) else 0
            m_val = mults[i] if i < len(mults) and mults[i] > 0 else 1.0
            base = s_val / m_val
            if FY26_START <= d <= FY26_END and d > LAST_ACTUAL:
                fy26 += s_val
                monthly_fy26[d.month] += s_val
                if s_val > 0:
                    fy26_trading_days += 1
            elif FY27_START <= d <= FY27_END:
                fy27 += s_val
                fy27_base += base
                fy27_days += 1
                monthly_fy27[d.month] += s_val
                if i < len(atp_arr) and atp_arr[i] > 0:
                    atp_values.append(atp_arr[i])
                # Recompute transactions when scenario overrides sales
                if scenario_override:
                    if i < len(atp_arr) and atp_arr[i] > 0:
                        transactions_total += s_val / atp_arr[i]
                elif i < len(trans_arr):
                    transactions_total += trans_arr[i]

        growth_pct = ((fy27 - fy26) / fy26 * 100) if fy26 > 0 else None
        base_growth_pct = ((fy27_base - fy26) / fy26 * 100) if fy26 > 0 else None
        multiplier_lift_pct = (growth_pct - base_growth_pct) if growth_pct is not None and base_growth_pct is not None else None
        avg_atp = mean(atp_values) if atp_values else None

        # Count extreme weather days that hit this venue's state
        extreme_fy26_count = 0
        extreme_fy27_count = 0
        wx_arr = r.get('weather_extreme', [])
        for i, d_str in enumerate(r['forecast_dates']):
            if i >= len(wx_arr) or not wx_arr[i]:
                continue
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d")
            except Exception:
                continue
            if FY26_START <= d <= FY26_END:
                extreme_fy26_count += 1
            elif FY27_START <= d <= FY27_END:
                extreme_fy27_count += 1

        comment = _venue_movement_comment(
            venue, v_cohort, opening_date_str, fy26, fy27, fy27_base,
            growth_pct, base_growth_pct, multiplier_lift_pct,
            r['servings'].get('model', ''),
            extreme_fy26_count, extreme_fy27_count,
        )

        # Pull scenario override (if any) for return
        scenario_growth = None
        scenario_avg_daily = None
        scenario_comment = ''
        if scenario_override:
            scenario_growth = scenario_override.get('target_growth_pct')
            scenario_avg_daily = scenario_override.get('target_avg_daily')
            # Backward compat: old snapshots may have a `comments` array
            if 'comment' in scenario_override:
                scenario_comment = scenario_override.get('comment', '') or ''
            else:
                cs = scenario_override.get('comments') or []
                scenario_comment = (cs[0] if cs else '') or ''

        # Average daily $: FY26 over trading days, FY27 over forecast days
        avg_daily_fy26 = (fy26 / fy26_trading_days) if fy26_trading_days > 0 else None
        avg_daily_fy27 = (fy27 / fy27_days) if fy27_days > 0 else None

        # FY-ordered monthly arrays (Jul..Jun) for the sparkline
        fy25_monthly_arr = [round(monthly_fy25[m], 2) for m in FY_MONTH_ORDER]
        fy26_monthly_arr = [round(monthly_fy26[m], 2) for m in FY_MONTH_ORDER]
        fy27_monthly_arr = [round(monthly_fy27[m], 2) for m in FY_MONTH_ORDER]

        venue_rows.append({
            'venue': venue,
            'state': v_state,
            'cluster': v_cluster,
            'cohort': v_cohort,
            'opening_date': opening_date_str,
            'fy26_forecast': round(fy26, 2),
            'fy27_forecast': round(fy27, 2),
            'fy26_trading_days': fy26_trading_days,
            'fy27_days': fy27_days,
            'avg_daily_fy26': round(avg_daily_fy26, 2) if avg_daily_fy26 is not None else None,
            'avg_daily_fy27': round(avg_daily_fy27, 2) if avg_daily_fy27 is not None else None,
            'growth_pct': round(growth_pct, 1) if growth_pct is not None else None,
            'base_growth_pct': round(base_growth_pct, 1) if base_growth_pct is not None else None,
            'multiplier_lift_pct': round(multiplier_lift_pct, 1) if multiplier_lift_pct is not None else None,
            'avg_atp': round(avg_atp, 2) if avg_atp is not None else None,
            'transactions_fy27': round(transactions_total, 0),
            'model': r['servings'].get('model', ''),
            'rmse': r['servings'].get('rmse', 0),
            'comment': comment,
            'guardrail_note': guardrail_note or '',
            # Scenario fields (simplified to single comment)
            'scenario_target_growth_pct': scenario_growth,
            'scenario_target_avg_daily': scenario_avg_daily,
            'scenario_comment': scenario_comment,
            # Monthly arrays for the sparkline (Jul..Jun order)
            'monthly_fy25': fy25_monthly_arr,
            'monthly_fy26': fy26_monthly_arr,
            'monthly_fy27': fy27_monthly_arr,
        })

    # Sort: State -> Cluster -> Venue
    venue_rows.sort(key=lambda v: (v['state'], v['cluster'], v['venue']))

    # Group totals
    state_totals = {}
    cluster_totals = {}
    for v in venue_rows:
        st = v['state']
        cl = (v['state'], v['cluster'])
        for key, dct in [(st, state_totals), (cl, cluster_totals)]:
            t = dct.setdefault(key, {'fy26': 0, 'fy27': 0, 'transactions_fy27': 0, 'venue_count': 0})
            t['fy26'] += v['fy26_forecast']
            t['fy27'] += v['fy27_forecast']
            t['transactions_fy27'] += v['transactions_fy27']
            t['venue_count'] += 1
    state_summary = [
        {'state': st, **t, 'growth_pct': round((t['fy27'] - t['fy26']) / t['fy26'] * 100, 1) if t['fy26'] > 0 else None}
        for st, t in sorted(state_totals.items())
    ]
    cluster_summary = [
        {'state': st, 'cluster': cl, **t, 'growth_pct': round((t['fy27'] - t['fy26']) / t['fy26'] * 100, 1) if t['fy26'] > 0 else None}
        for (st, cl), t in sorted(cluster_totals.items())
    ]

    guardrailed_venues = sum(1 for v in venue_rows if v.get('guardrail_note'))
    return jsonify({
        'venues': venue_rows,
        'state_summary': state_summary,
        'cluster_summary': cluster_summary,
        'filters_applied': {'state': state, 'cluster': cluster, 'cohort': cohort, 'venue': venue_q, 'guardrails': apply_guards},
        'venue_count': len(venue_rows),
        'active_scenario': SESSION_STATE.get('active_scenario'),
        'scenarios': list((SESSION_STATE.get('scenarios') or {}).keys()),
        'guardrails_active': apply_guards,
        'venues_guardrailed': guardrailed_venues,
    })


@app.route("/venue-compare", methods=["GET"])
def venue_compare():
    """Side-by-side comparison data for up to N venues.
    Query: ?venues=v1|v2|v3|v4 (pipe-separated; max 8 enforced).
    Returns per-venue monthly arrays for FY26 and FY27, plus growth %.
    """
    raw = request.args.get('venues', '') or ''
    requested = [v.strip() for v in raw.split('|') if v.strip()][:8]
    if not requested:
        return jsonify({'error': 'Provide ?venues=v1|v2|... (pipe-separated)'}), 400

    results = SESSION_STATE.get('last_results') or []
    if not results:
        return jsonify({'error': 'No forecast results. Run /generate first.'}), 400

    servings_actual = SESSION_STATE.get('sales_servings') or {}
    location_bi = SESSION_STATE.get('location_bi') or {}

    FY26_START = datetime(2025, 7, 1)
    FY26_END = datetime(2026, 6, 30)
    FY27_START = datetime(2026, 7, 1)
    FY27_END = datetime(2027, 6, 30)
    LAST_ACTUAL = datetime(2026, 4, 30)

    fy_order = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    month_labels = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

    by_venue = {r['venue']: r for r in results if 'servings' in r}

    output_venues = []
    for venue in requested:
        if venue not in by_venue:
            output_venues.append({'venue': venue, 'error': 'not in forecast results'})
            continue
        r = by_venue[venue]
        loc = location_bi.get(venue, {}) or {}
        monthly_fy26 = {m: 0.0 for m in range(1, 13)}
        monthly_fy27 = {m: 0.0 for m in range(1, 13)}

        # FY26 actuals
        for d_str, sales_val in servings_actual.get(venue, {}).items():
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d")
            except Exception:
                continue
            if FY26_START <= d <= LAST_ACTUAL and sales_val:
                monthly_fy26[d.month] += sales_val

        # FY26 completion FC + FY27 from forecast result
        sales = r['servings']['sales']
        for i, d_str in enumerate(r['forecast_dates']):
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d")
            except Exception:
                continue
            s_val = sales[i] if i < len(sales) else 0
            if FY26_START <= d <= FY26_END and d > LAST_ACTUAL:
                monthly_fy26[d.month] += s_val
            elif FY27_START <= d <= FY27_END:
                monthly_fy27[d.month] += s_val

        monthly = []
        fy26_tot = 0.0
        fy27_tot = 0.0
        for i, m in enumerate(fy_order):
            f26 = monthly_fy26.get(m, 0)
            f27 = monthly_fy27.get(m, 0)
            growth = ((f27 - f26) / f26 * 100) if f26 > 0 else None
            monthly.append({
                'month_idx': m,
                'month_label': month_labels[i],
                'fy26': round(f26, 2),
                'fy27': round(f27, 2),
                'growth_pct': round(growth, 2) if growth is not None else None,
            })
            fy26_tot += f26
            fy27_tot += f27

        output_venues.append({
            'venue': venue,
            'state': get_venue_state(venue) or '',
            'cluster': loc.get('cluster') or '',
            'cohort': _classify_cohort(venue),
            'monthly': monthly,
            'fy26_total': round(fy26_tot, 2),
            'fy27_total': round(fy27_tot, 2),
            'growth_pct': round((fy27_tot - fy26_tot) / fy26_tot * 100, 1) if fy26_tot > 0 else None,
            'model': r['servings'].get('model', ''),
        })

    return jsonify({
        'venues': output_venues,
        'count': len(output_venues),
    })


@app.route("/data-sources", methods=["GET", "POST"])
def data_sources():
    """GET: report on weather + holiday data loaded.
    POST: pre-fetch weather + holidays for all distinct venue states.
    """
    if request.method == "POST":
        states = set()
        for key in ['location_bi', 'venue_details', 'multipliers']:
            d = SESSION_STATE.get(key) or {}
            for v in d.keys():
                s = get_venue_state(v)
                if s:
                    states.add(s)
        year_range = range(2020, FORECAST_END.year + 1)
        for s in sorted(states):
            load_holidays_for_state(s, year_range)
            load_weather_for_state(s, datetime(2020, 5, 1), FORECAST_END)
        return jsonify({
            'status': 'ok',
            'states_loaded': sorted(states),
            'holiday_cache_keys': list(HOLIDAY_CACHE.keys()),
            'weather_cache_keys': list(WEATHER_CACHE.keys()),
        })

    # GET — summarise current cache contents
    weather_summary = {}
    for key, m in WEATHER_CACHE.items():
        if not m:
            continue
        temps = [w['temp_max'] for w in m.values() if isinstance(w, dict) and w.get('temp_max') is not None]
        precips = [w['precip'] for w in m.values() if isinstance(w, dict) and w.get('precip') is not None]
        gusts = [w['gusts_max'] for w in m.values() if isinstance(w, dict) and w.get('gusts_max') is not None]
        n_extreme = sum(1 for w in m.values() if isinstance(w, dict) and w.get('extreme'))
        if not temps:
            continue
        weather_summary[key] = {
            'days': len(m),
            'min_date': min(m.keys()),
            'max_date': max(m.keys()),
            'min_temp': round(min(temps), 1),
            'max_temp': round(max(temps), 1),
            'avg_temp': round(sum(temps) / len(temps), 1),
            'total_rain_mm': round(sum(precips), 1) if precips else 0,
            'max_gust_kmh': round(max(gusts), 1) if gusts else 0,
            'extreme_weather_days': n_extreme,
        }
    holiday_summary = {}
    for key, m in HOLIDAY_CACHE.items():
        if not m:
            continue
        holiday_summary[key] = {
            'count': len(m),
            'examples': sorted(m.items())[:5],
            'all_dates': sorted(m.keys()),
        }
    return jsonify({
        'weather': weather_summary,
        'holidays': holiday_summary,
        'supported_states': list(STATE_COORDS.keys()),
        'state_city_map': {k: v[2] for k, v in STATE_COORDS.items()},
    })


# ── SCENARIOS: per-venue growth targets + comments, up to 3 named versions ────
FY26_START_DT = datetime(2025, 7, 1)
FY26_END_DT = datetime(2026, 6, 30)
FY27_START_DT = datetime(2026, 7, 1)
FY27_END_DT = datetime(2027, 6, 30)
LAST_ACTUAL_DT = datetime(2026, 4, 30)


def _scenario_venue_baseline_fy26(venue, r):
    """Return baseline FY26 $ for this venue = actuals (Jul-Apr) + completion FC (May/Jun) with original mults."""
    servings_actual = SESSION_STATE.get('sales_servings') or {}
    venue_actuals = servings_actual.get(venue, {})
    fy26 = 0.0
    for d_str, v in venue_actuals.items():
        try:
            d = datetime.strptime(d_str, "%Y-%m-%d")
        except Exception:
            continue
        if FY26_START_DT <= d <= LAST_ACTUAL_DT and v:
            fy26 += v
    sales = r['servings']['sales']
    for i, d_str in enumerate(r['forecast_dates']):
        try:
            d = datetime.strptime(d_str, "%Y-%m-%d")
        except Exception:
            continue
        if FY26_START_DT <= d <= FY26_END_DT and d > LAST_ACTUAL_DT and i < len(sales):
            fy26 += sales[i]
    return fy26


def _scenario_apply_to_result(r, override):
    """Return a NEW sales array for r where FY27 days are scaled so the venue
    hits the scenario target. Two target modes (whichever is set on override):
      - target_avg_daily: FY27 average daily $ = this value
      - target_growth_pct: FY27 total = FY26 baseline × (1 + growth/100)
    The calibration is a uniform scalar applied to the Prophet BASE (sales/mult),
    so Prophet's day-by-day phasing is preserved. Excessive ramps naturally pull
    back when cal_mult < 1.
    FY26 completion days are kept at their existing values.
    """
    if not override:
        return r['servings']['sales']

    target_avg = override.get('target_avg_daily')
    target_growth = override.get('target_growth_pct')

    # Convert to floats; treat empty/missing as None
    def _as_float(v):
        if v in (None, ''):
            return None
        try:
            return float(v)
        except Exception:
            return None
    target_avg = _as_float(target_avg)
    target_growth = _as_float(target_growth)

    if target_avg is None and target_growth is None:
        return r['servings']['sales']

    sales = r['servings']['sales']
    mults = r['servings']['multipliers']

    # Sum FY27 BASE forecast (sales / multiplier) so we can compute the calibration
    base_fy27 = 0.0
    fy27_idx = []
    for i, d_str in enumerate(r['forecast_dates']):
        try:
            d = datetime.strptime(d_str, "%Y-%m-%d")
        except Exception:
            continue
        if FY27_START_DT <= d <= FY27_END_DT:
            m = mults[i] if i < len(mults) and mults[i] > 0 else 1.0
            base_fy27 += (sales[i] / m) if i < len(sales) else 0
            fy27_idx.append(i)

    if base_fy27 <= 0 or not fy27_idx:
        return r['servings']['sales']

    # Resolve the target FY27 total $
    if target_avg is not None:
        # Average daily target × number of FY27 forecast days
        target_fy27 = target_avg * len(fy27_idx)
    else:
        fy26_baseline = _scenario_venue_baseline_fy26(r['venue'], r)
        if fy26_baseline <= 0:
            return r['servings']['sales']
        target_fy27 = fy26_baseline * (1 + target_growth / 100.0)

    cal_mult = target_fy27 / base_fy27

    new_sales = list(sales)
    for i in fy27_idx:
        m = mults[i] if i < len(mults) and mults[i] > 0 else 1.0
        new_sales[i] = (sales[i] / m) * cal_mult
    return new_sales


def _apply_baseline_guardrails(r):
    """Cap a venue's FY27 forecast within a sensible band of *annualized* FY26.
    This is the safety net for two Prophet pathologies:
      1. Recent dip extrapolated into runaway FY27 decline
      2. Partial-year FY26 venues (e.g. opened Dec 2025) producing apparent
         huge YoY growth simply because FY27 has 12 months vs FY26 has 5
    Returns (new_sales_array, note_or_None).
    Bands (configurable via SESSION_STATE['guardrail_settings']):
      Mature venues (≥330 FY26 trading days)   : 0.75x to 1.40x annualized FY26
      Ramping venues (<330 FY26 trading days)  : 0.70x to 1.50x annualized FY26
    """
    settings = SESSION_STATE.get('guardrail_settings') or {}
    mature_lo = float(settings.get('mature_lo', 0.75))
    mature_hi = float(settings.get('mature_hi', 1.40))
    ramp_lo   = float(settings.get('ramp_lo',   0.70))
    ramp_hi   = float(settings.get('ramp_hi',   1.50))
    min_fy26_days = int(settings.get('min_fy26_days_to_guard', 30))

    venue = r['venue']
    sales = list(r['servings']['sales'])
    servings_actual = SESSION_STATE.get('sales_servings') or {}

    # Compute FY26: actuals + completion forecast (May/Jun 2026)
    fy26 = 0.0
    fy26_days = 0
    for d_str, v in (servings_actual.get(venue) or {}).items():
        try:
            d = datetime.strptime(d_str, "%Y-%m-%d")
        except Exception:
            continue
        if FY26_START_DT <= d <= LAST_ACTUAL_DT and v:
            fy26 += v
            if v > 0:
                fy26_days += 1
    for i, d_str in enumerate(r['forecast_dates']):
        try:
            d = datetime.strptime(d_str, "%Y-%m-%d")
        except Exception:
            continue
        if FY26_START_DT <= d <= FY26_END_DT and d > LAST_ACTUAL_DT and i < len(sales):
            v = sales[i]
            fy26 += v
            if v > 0:
                fy26_days += 1

    if fy26_days < min_fy26_days:
        return sales, None   # Too little FY26 data — skip (likely a brand-new venue)

    annualized_fy26 = fy26 * (365.0 / fy26_days)

    fy27 = 0.0
    fy27_idx = []
    for i, d_str in enumerate(r['forecast_dates']):
        try:
            d = datetime.strptime(d_str, "%Y-%m-%d")
        except Exception:
            continue
        if FY27_START_DT <= d <= FY27_END_DT and i < len(sales):
            fy27 += sales[i]
            fy27_idx.append(i)

    if not fy27_idx or fy27 <= 0 or annualized_fy26 <= 0:
        return sales, None

    if fy26_days >= 330:
        lo_band, hi_band, kind = mature_lo, mature_hi, 'mature'
    else:
        lo_band, hi_band, kind = ramp_lo, ramp_hi, 'ramping'

    lo_cap = annualized_fy26 * lo_band
    hi_cap = annualized_fy26 * hi_band

    if lo_cap <= fy27 <= hi_cap:
        return sales, None

    if fy27 < lo_cap:
        target, action = lo_cap, 'floored'
        band = lo_band
    else:
        target, action = hi_cap, 'capped'
        band = hi_band
    scale = target / fy27

    new_sales = list(sales)
    for i in fy27_idx:
        new_sales[i] = sales[i] * scale

    pre_growth = (fy27 - fy26) / fy26 * 100 if fy26 > 0 else None
    post_growth = (target - fy26) / fy26 * 100 if fy26 > 0 else None
    note = (
        f"Guardrail {action} ({kind}, band {band:.2f}x annualized FY26): "
        f"raw forecast {pre_growth:+.1f}% → {post_growth:+.1f}%"
        if pre_growth is not None else f"Guardrail {action}"
    )
    return new_sales, note


def _resolve_venue_sales(r, apply_guardrails=True):
    """Return (sales_array, scenario_override, guardrail_note).
    Priority:
      1. Active scenario override → respect user target, no guardrail
      2. No scenario + guardrails on → apply baseline guardrails
      3. No scenario + guardrails off → raw forecast
    """
    override = _active_scenario_override(r['venue'])
    has_target = override and (
        override.get('target_growth_pct') not in (None, '') or
        override.get('target_avg_daily') not in (None, '')
    )
    if has_target:
        return _scenario_apply_to_result(r, override), override, None
    if apply_guardrails:
        sales, note = _apply_baseline_guardrails(r)
        return sales, override, note
    return r['servings']['sales'], override, None


def _active_scenario_override(venue):
    """Return the active scenario's override dict for venue, or None."""
    name = SESSION_STATE.get('active_scenario')
    if not name:
        return None
    sc = (SESSION_STATE.get('scenarios') or {}).get(name)
    if not sc:
        return None
    return (sc.get('venue_overrides') or {}).get(venue)


@app.route("/scenario/list", methods=["GET"])
def scenario_list():
    scenarios = SESSION_STATE.get('scenarios') or {}
    out = []
    for name, sc in scenarios.items():
        out.append({
            'name': name,
            'created_at': sc.get('created_at', ''),
            'venue_count': len(sc.get('venue_overrides', {})),
            'note': sc.get('note', ''),
        })
    out.sort(key=lambda x: x.get('created_at', ''))
    return jsonify({
        'scenarios': out,
        'active': SESSION_STATE.get('active_scenario'),
    })


@app.route("/scenario/create", methods=["POST"])
def scenario_create():
    body = request.get_json(silent=True) or {}
    name = (body.get('name') or '').strip()[:60]
    note = (body.get('note') or '').strip()[:500]
    if not name:
        return jsonify({'error': 'Provide a scenario name'}), 400
    scenarios = SESSION_STATE.get('scenarios') or {}
    if name in scenarios:
        return jsonify({'error': f'Scenario "{name}" already exists'}), 400
    # Enforce max 3 scenarios — drop the oldest if we'd exceed
    if len(scenarios) >= 3:
        oldest = sorted(scenarios.items(), key=lambda kv: kv[1].get('created_at', ''))[0][0]
        scenarios.pop(oldest, None)
        log.info(f"Scenarios capped at 3 — removed oldest: {oldest}")
    scenarios[name] = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'note': note,
        'venue_overrides': {},
    }
    SESSION_STATE['scenarios'] = scenarios
    SESSION_STATE['active_scenario'] = name
    return jsonify({'status': 'ok', 'name': name, 'active': name})


@app.route("/scenario/activate", methods=["POST"])
def scenario_activate():
    body = request.get_json(silent=True) or {}
    name = (body.get('name') or '').strip()
    scenarios = SESSION_STATE.get('scenarios') or {}
    if name and name not in scenarios:
        return jsonify({'error': f'Unknown scenario "{name}"'}), 404
    SESSION_STATE['active_scenario'] = name or None
    return jsonify({'status': 'ok', 'active': SESSION_STATE['active_scenario']})


@app.route("/scenario/delete", methods=["POST"])
def scenario_delete():
    body = request.get_json(silent=True) or {}
    name = (body.get('name') or '').strip()
    scenarios = SESSION_STATE.get('scenarios') or {}
    if name not in scenarios:
        return jsonify({'error': 'Not found'}), 404
    scenarios.pop(name, None)
    if SESSION_STATE.get('active_scenario') == name:
        SESSION_STATE['active_scenario'] = None
    return jsonify({'status': 'ok', 'deleted': name})


@app.route("/scenario/venue", methods=["POST"])
def scenario_venue_update():
    """Update a single venue's overrides in a scenario.
    Body: {
      scenario: str, venue: str,
      target_growth_pct: float|null,
      target_avg_daily: float|null,
      comment: str
    }
    If both target_growth_pct and target_avg_daily are provided, target_avg_daily wins.
    """
    body = request.get_json(silent=True) or {}
    sc_name = (body.get('scenario') or '').strip()
    venue = (body.get('venue') or '').strip()
    growth = body.get('target_growth_pct')
    avg_daily = body.get('target_avg_daily')
    comment = (body.get('comment') or '').strip()[:500]

    scenarios = SESSION_STATE.get('scenarios') or {}
    if sc_name not in scenarios:
        return jsonify({'error': f'Unknown scenario "{sc_name}"'}), 404
    if not venue:
        return jsonify({'error': 'Provide venue name'}), 400

    def _f(v):
        if v in (None, ''):
            return None
        try:
            return float(v)
        except Exception:
            return None

    scenarios[sc_name].setdefault('venue_overrides', {})[venue] = {
        'target_growth_pct': _f(growth),
        'target_avg_daily': _f(avg_daily),
        'comment': comment,
        'updated_at': datetime.now().isoformat(timespec='seconds'),
    }
    return jsonify({'status': 'ok', 'scenario': sc_name, 'venue': venue,
                    'override': scenarios[sc_name]['venue_overrides'][venue]})


@app.route("/scenario/get", methods=["GET"])
def scenario_get():
    """Return all venue overrides for a scenario."""
    name = request.args.get('name', '').strip()
    scenarios = SESSION_STATE.get('scenarios') or {}
    if name not in scenarios:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({
        'name': name,
        'scenario': scenarios[name],
    })


# ── SNAPSHOTS: save & restore named versions of the forecast ──────────────────
import re as _re_snap

def _safe_snapshot_name(raw):
    """Strip risky chars; only alnum/underscore/hyphen/space, then collapse spaces."""
    s = _re_snap.sub(r'[^A-Za-z0-9_\- ]', '', str(raw or '')).strip()
    s = _re_snap.sub(r'\s+', '_', s)
    return s[:80]


@app.route("/snapshot/list", methods=["GET"])
def snapshot_list():
    """Return all saved snapshots (most-recent first)."""
    files = []
    for fn in os.listdir(SNAPSHOTS_DIR):
        if not fn.endswith('.json'):
            continue
        full = os.path.join(SNAPSHOTS_DIR, fn)
        try:
            stat = os.stat(full)
            with open(full, 'r', encoding='utf-8') as f:
                head = json.load(f)
            files.append({
                'filename': fn,
                'name': head.get('name', fn),
                'saved_at': head.get('saved_at', ''),
                'venue_count': head.get('venue_count', 0),
                'note': head.get('note', ''),
                'size_kb': round(stat.st_size / 1024, 1),
            })
        except Exception as e:
            files.append({'filename': fn, 'error': str(e)})
    files.sort(key=lambda x: x.get('saved_at', ''), reverse=True)
    return jsonify({'snapshots': files})


@app.route("/snapshot/save", methods=["POST"])
def snapshot_save():
    """Save current forecast + uploaded data as a named snapshot.
    Body: { name: str, note: str (optional) }
    """
    body = request.get_json(silent=True) or {}
    name = _safe_snapshot_name(body.get('name', ''))
    note = (body.get('note', '') or '').strip()[:500]
    if not name:
        return jsonify({'error': 'Provide a name (letters / digits / underscores / hyphens / spaces).'}), 400

    results = SESSION_STATE.get('last_results') or []
    if not results:
        return jsonify({'error': 'No forecast results to save. Run /generate first.'}), 400

    snapshot = {
        'name': name,
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'note': note,
        'venue_count': sum(1 for r in results if 'servings' in r),
        'forecast_period': f"{iso(FORECAST_START)} to {iso(FORECAST_END)}",
        # Full session state — enough to fully restore the view
        'session': {
            'sales_servings': SESSION_STATE.get('sales_servings'),
            'sales_retail': SESSION_STATE.get('sales_retail'),
            'sales_discounts': SESSION_STATE.get('sales_discounts'),
            'atp_history': SESSION_STATE.get('atp_history'),
            'location_bi': SESSION_STATE.get('location_bi'),
            'venue_details': SESSION_STATE.get('venue_details'),
            'multipliers': SESSION_STATE.get('multipliers'),
            'atp_growth_template': SESSION_STATE.get('atp_growth_template'),
            'last_results': results,
            'scenarios': SESSION_STATE.get('scenarios'),
            'active_scenario': SESSION_STATE.get('active_scenario'),
        },
        'weather_cache_summary': {k: len(v) for k, v in WEATHER_CACHE.items()},
        'holiday_cache_summary': {k: len(v) for k, v in HOLIDAY_CACHE.items()},
    }

    filename = name + '.json'
    path = os.path.join(SNAPSHOTS_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, default=str)
    size_kb = round(os.path.getsize(path) / 1024, 1)
    log.info(f"Snapshot saved: {filename} ({size_kb} KB)")
    return jsonify({'status': 'ok', 'filename': filename, 'name': name, 'size_kb': size_kb})


@app.route("/snapshot/load", methods=["POST"])
def snapshot_load():
    """Restore a snapshot into SESSION_STATE.
    Body: { filename: str }
    """
    body = request.get_json(silent=True) or {}
    filename = _safe_snapshot_name(body.get('filename', '').replace('.json', '')) + '.json'
    path = os.path.join(SNAPSHOTS_DIR, filename)
    if not os.path.exists(path):
        return jsonify({'error': f'Snapshot not found: {filename}'}), 404

    with open(path, 'r', encoding='utf-8') as f:
        snapshot = json.load(f)

    sess = snapshot.get('session', {})
    for key, val in sess.items():
        SESSION_STATE[key] = val

    log.info(f"Snapshot loaded: {filename}")
    return jsonify({
        'status': 'ok',
        'filename': filename,
        'name': snapshot.get('name'),
        'saved_at': snapshot.get('saved_at'),
        'venue_count': snapshot.get('venue_count'),
        'note': snapshot.get('note', ''),
    })


@app.route("/snapshot/delete", methods=["POST"])
def snapshot_delete():
    body = request.get_json(silent=True) or {}
    filename = _safe_snapshot_name(body.get('filename', '').replace('.json', '')) + '.json'
    path = os.path.join(SNAPSHOTS_DIR, filename)
    if not os.path.exists(path):
        return jsonify({'error': 'Not found'}), 404
    os.remove(path)
    return jsonify({'status': 'ok', 'deleted': filename})


@app.route("/export/standalone", methods=["POST"])
def export_standalone():
    """Generate a self-contained HTML viewer with the current forecast embedded.
    Body: { name?: str, note?: str }
    Pre-computes the sales-detail + venues-detail views across reasonable filter
    combinations so the colleague can interact with the file in a browser
    without needing the server.
    """
    body = request.get_json(silent=True) or {}
    snapshot_name = (body.get('name') or 'Forecast').strip()[:80]
    note = (body.get('note') or '').strip()[:500]

    results = SESSION_STATE.get('last_results') or []
    if not results:
        return jsonify({'error': 'No forecast results to export. Run /generate first.'}), 400

    # Discover filter universe
    states = set()
    clusters = set()
    cohorts = set()
    venues = set()
    for r in results:
        if 'error' in r or 'servings' not in r:
            continue
        v = r['venue']
        s = get_venue_state(v) or ''
        if s:
            states.add(s)
        loc = (SESSION_STATE.get('location_bi') or {}).get(v) or {}
        if loc.get('cluster'):
            clusters.add(loc['cluster'])
        cohorts.add(_classify_cohort(v))
        venues.add(v)

    filter_options = {
        'states': sorted(states),
        'clusters': sorted(clusters),
        'cohorts': sorted(cohorts),
        'venues': sorted(venues),
    }

    # Pre-compute the most commonly-needed filter slices.
    # Avoid the full cross-product explosion; cover the slices a reviewer is likely to use:
    #   (sales_type) × {all,each state,each cohort} × {all,each cluster}
    sales_types = ['servings', 'retail', 'pos_discounts', 'all']
    state_filters = ['all'] + sorted(states)
    cluster_filters = ['all'] + sorted(clusters)
    cohort_filters = ['all'] + sorted(cohorts)

    sales_detail_by_filter = {}
    venues_detail_by_filter = {}

    from werkzeug.test import EnvironBuilder

    def _call_local_view(view_func, args):
        """Run a Flask view function with given query args, capturing JSON."""
        builder = EnvironBuilder(method='GET', query_string=args)
        env = builder.get_environ()
        with app.request_context(env):
            resp = view_func()
            if isinstance(resp, tuple):
                resp = resp[0]
            return resp.get_json()

    combos_built = 0
    for stype in sales_types:
        for st in state_filters:
            for co in cohort_filters:
                # cluster only loops when state is 'all' (cluster ∩ state already constrains)
                cluster_set = [c for c in cluster_filters if c == 'all' or st == 'all']
                for cl in cluster_set:
                    key = f"{stype}|{st}|{cl}|{co}|all"
                    args = {'sales_type': stype, 'state': st, 'cluster': cl, 'cohort': co, 'venue': 'all'}
                    try:
                        sales_detail_by_filter[key] = _call_local_view(sales_detail, args)
                        if stype == 'servings':
                            # Only build per-venue table for the default sales-type
                            venues_detail_by_filter[key] = _call_local_view(venues_detail, args).get('venues', [])
                        combos_built += 1
                    except Exception as e:
                        log.warning(f"Snapshot precompute failed for {key}: {e}")

    payload = {
        'meta': {
            'name': snapshot_name,
            'saved_at': datetime.now().isoformat(timespec='seconds'),
            'note': note,
            'venue_count': sum(1 for r in results if 'servings' in r),
            'forecast_period': f"{iso(FORECAST_START)} to {iso(FORECAST_END)}",
        },
        'filter_options': filter_options,
        'sales_detail_by_filter': sales_detail_by_filter,
        'venues_detail_by_filter': venues_detail_by_filter,
    }
    log.info(f"Standalone export: {combos_built} filter slices precomputed for '{snapshot_name}'")

    # Inject into template
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'standalone_viewer.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        tmpl = f.read()
    payload_json = json.dumps(payload, default=str)
    # The <script id="snapshotData"> uses textContent — inject as raw JSON between tags
    html = tmpl.replace('__SNAPSHOT_JSON__', payload_json)

    safe_name = _re_snap.sub(r'\s+', '_', _safe_snapshot_name(snapshot_name)) or 'forecast'
    out_path = os.path.join(RESULTS_DIR, f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    size_kb = round(os.path.getsize(out_path) / 1024, 1)
    log.info(f"Standalone HTML written: {os.path.basename(out_path)} ({size_kb} KB, {combos_built} slices)")

    return send_file(out_path, as_attachment=True, download_name=os.path.basename(out_path))


@app.route("/reset", methods=["POST"])
def reset():
    for k in SESSION_STATE:
        SESSION_STATE[k] = None
    return jsonify({'status': 'reset'})


if __name__ == "__main__":
    log.info("Starting Venu Cast forecasting server on http://localhost:5000")
    log.info(f"Forecast period: {iso(FORECAST_START)} to {iso(FORECAST_END)} ({FORECAST_DAYS} days)")
    app.run(host="0.0.0.0", port=5000, debug=False)
