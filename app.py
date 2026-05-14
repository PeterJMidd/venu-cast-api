"""
Venu Cast — Forecast API
Models: Prophet (primary) -> SARIMA (fallback) -> Holt-Winters (always available)

ENHANCEMENTS:
1. Weather × Day-of-Week interaction modeling
2. 15-month monthly forecast multipliers
3. New venue drivers with cannibalization
4. Cluster analysis integration
5. Multi-component sales forecasting (total, retail, discount)
6. Average ticket price forecasting with growth templates
7. Integrated transaction forecasts
"""
import os, io, json, math, logging, traceback, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class _BypassProphet(Exception):
    """Raised inside run_prophet to return early with a non-Prophet forecast."""
    def __init__(self, fi, fc, lo, hi, rmse, comps):
        self.fi, self.fc, self.lo, self.hi, self.rmse, self.comps = fi, fc, lo, hi, rmse, comps

app = Flask(__name__)
CORS(app)

# ── UTILS ──────────────────────────────────────────────────────────────────────
def safe(v):
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except Exception:
        return None

def iso(d): return d.strftime("%Y-%m-%d")
def add_days(d, n): return d + timedelta(days=n)
def mean(lst): return sum(lst) / len(lst) if lst else 0.0
def stdev(lst):
    if len(lst) < 2: return 1.0
    m = mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst)) or 1.0

# ── ENHANCEMENT 2: Monthly Multiplier System ────────────────────────────────────
def load_monthly_multipliers(multiplier_df):
    """
    Load 15-month multipliers from DataFrame.
    Returns dict: {venue_id: [m1, m2, ..., m15]}
    """
    result = {}
    for idx, row in multiplier_df.iterrows():
        venue_id = row.get('venue_id')
        if venue_id:
            multipliers = [
                float(row.get(f'month_{i}_multiplier', 1.0) or 1.0)
                for i in range(1, 16)
            ]
            result[venue_id] = multipliers
    return result

def get_monthly_multiplier(venue_id, forecast_date, multiplier_map, cycle_start=None):
    """
    Get monthly multiplier for a given date.
    cycle_start: datetime object marking where the 15-month cycle begins (default: Jan 1)
    """
    if venue_id not in multiplier_map:
        return 1.0
    
    if cycle_start is None:
        cycle_start = datetime(forecast_date.year, 1, 1)
    
    days_since_start = (forecast_date - cycle_start).days
    month_in_cycle = (days_since_start // 30) + 1
    
    # Cycle every 15 months
    idx = ((month_in_cycle - 1) % 15)
    return multiplier_map[venue_id][idx] if idx < len(multiplier_map[venue_id]) else 1.0

# ── ENHANCEMENT 3: New Venue Driver Framework ────────────────────────────────────
def load_venue_drivers(venue_details_df):
    """
    Load new venue driver information.
    Returns dict: {venue_id: {parent_venue, ramp_up_schedule, impacted_stores}}
    """
    result = {}
    for idx, row in venue_details_df.iterrows():
        venue_id = row.get('venue_id')
        if venue_id:
            ramp_up = [float(row.get(f'ramp_up_month_{i}', 1.0) or 1.0) for i in range(1, 13)]
            impacted = {}
            for i in range(1, 6):  # Support up to 5 impacted stores
                store = row.get(f'impacted_store_{i}')
                impact = row.get(f'impact_pct_{i}', 0)
                if store:
                    impacted[store] = float(impact or 0)
            
            result[venue_id] = {
                'parent_venue': row.get('parent_venue'),
                'opening_date': row.get('opening_date'),
                'ramp_up_schedule': ramp_up,
                'impacted_stores': impacted,
                'base_sales_estimate': float(row.get('base_sales_estimate', 0) or 0)
            }
    return result

def forecast_new_venue(parent_forecast, ramp_up_schedule, opening_date, forecast_dates, parent_mean):
    """
    Forecast new venue using parent venue pattern and ramp-up percentages.
    """
    result = []
    for date in forecast_dates:
        if date < opening_date:
            result.append(0)
        else:
            months_since_open = (date - opening_date).days / 30.0
            if months_since_open < 12:
                ramp_pct = ramp_up_schedule[int(months_since_open)]
            else:
                ramp_pct = ramp_up_schedule[-1]
            result.append(parent_mean * ramp_pct)
    return result

def apply_cannibalization(all_forecasts, impact_map, forecast_dates):
    """
    Apply venue cannibalization: reduce impacted venue forecasts.
    impact_map: {venue_id: {impacted_venue_id: impact_percentage}}
    """
    for venue_id, impact_info in impact_map.items():
        if venue_id in all_forecasts:
            for impacted_venue, impact_pct in impact_info.items():
                if impacted_venue in all_forecasts:
                    all_forecasts[impacted_venue] = [
                        f * (1 - impact_pct) for f in all_forecasts[impacted_venue]
                    ]
    return all_forecasts

# ── ENHANCEMENT 4: Cluster Analysis Integration ────────────────────────────────────
def load_cluster_data(cluster_df):
    """
    Load cluster membership and saturation data.
    Returns dict: {venue_id: {cluster, cluster_type, saturation}}
    """
    result = {}
    for idx, row in cluster_df.iterrows():
        venue_id = row.get('venue_id')
        if venue_id:
            result[venue_id] = {
                'cluster': row.get('Cat_Cluster'),
                'cluster_type': row.get('Cat_Cluster_Type'),
                'saturation': float(row.get('cluster_saturation_indicator', 0.5) or 0.5),
                'competitive_overlap': float(row.get('competitive_overlap_score', 0) or 0)
            }
    return result

def analyze_cluster_history(sales_history_dict, cluster_memberships):
    """
    Learn cannibalization effects from historical data within clusters.
    Returns: {(opening_venue, impacted_venue): learned_impact_pct}
    """
    # This is a simplified implementation
    # In production, analyze pre/post-opening trends for each venue
    cannibalization_matrix = {}
    
    # Placeholder: default 10% cannibalization within cluster
    for venue_id, cluster_info in cluster_memberships.items():
        cluster = cluster_info['cluster']
        overlap = cluster_info['competitive_overlap']
        
        # Scale cannibalization by competitive overlap
        default_cann = overlap * 0.15  # Up to 15% cannibalization
        
        # Find other venues in same cluster
        for other_venue, other_info in cluster_memberships.items():
            if other_venue != venue_id and other_info['cluster'] == cluster:
                cannibalization_matrix[(venue_id, other_venue)] = default_cann
    
    return cannibalization_matrix

def forecast_cluster(cluster_venues, base_forecasts, cannibalization_matrix, cluster_saturation):
    """
    Apply cluster-level adjustments to forecasts.
    """
    adjusted = {v: list(base_forecasts[v]) for v in cluster_venues if v in base_forecasts}
    
    for venue_a in cluster_venues:
        if venue_a in base_forecasts:
            for venue_b in cluster_venues:
                if venue_a != venue_b and (venue_a, venue_b) in cannibalization_matrix:
                    cann_rate = cannibalization_matrix[(venue_a, venue_b)]
                    effective_cann = cann_rate * cluster_saturation
                    if venue_b in adjusted:
                        adjusted[venue_b] = [
                            f * (1 - effective_cann) for f in adjusted[venue_b]
                        ]
    
    return adjusted

# ── ENHANCEMENT 5 & 6: Multi-Component Sales + ATP Forecasting ──────────────────
def validate_sales_components(total_sales, retail_sales, pos_discounts, tolerance=0.05):
    """
    Validate that components sum correctly: total ≈ retail - discounts
    Returns list of (venue, date, issues) tuples
    """
    issues = []
    # Simplified validation - in production, compare all three dataframes
    return issues

def forecast_sales_components(venue_id, dates, total_vals, retail_vals, discount_vals, 
                              forecast_days, holidays, weather_map, atp_history=None):
    """
    Forecast each sales component independently.
    Returns dict with keys: total_sales, retail_sales, pos_discounts
    """
    results = {}
    
    # Forecast each component
    for component, values in [
        ('total_sales', total_vals),
        ('retail_sales', retail_vals),
        ('pos_discounts', discount_vals)
    ]:
        fi, fc, lo, hi, rmse, model = forecast_with_fallback(
            dates, values, forecast_days, holidays, weather_map
        )
        results[component] = {
            'fitted': fi,
            'forecast': fc,
            'lower_90': lo,
            'upper_90': hi,
            'model_used': model,
            'rmse': rmse
        }
    
    return results

# ── ENHANCEMENT 1: Weather × Day-of-Week Interaction ────────────────────────────────
def add_weather_dow_interaction(df, dow_encoded):
    """
    Create interaction term: temperature × day-of-week
    dow_encoded: 0=Mon, 1=Tue, ..., 6=Sun
    """
    df['weather_dow_interaction'] = df['temp_max'] * dow_encoded
    return df

# ── ENHANCEMENT 7: Average Ticket Price Forecasting ──────────────────────────────────
def forecast_average_ticket_price(dates, atp_values, forecast_days, holidays, weather_map,
                                  growth_overrides=None):
    """
    Forecast ATP with optional user-provided growth overrides.
    """
    fi, fc, lo, hi, rmse, model = forecast_with_fallback(
        dates, atp_values, forecast_days, holidays, weather_map, atp_mode=True
    )
    
    # Apply growth overrides if provided
    if growth_overrides:
        fc = apply_atp_growth_overrides(fc, growth_overrides, dates[-1])
    
    return {
        'fitted': fi,
        'forecast': fc,
        'lower_90': lo,
        'upper_90': hi,
        'model_used': model,
        'rmse': rmse
    }

def apply_atp_growth_overrides(base_forecast, growth_template, start_date):
    """
    Apply month-by-month growth percentage overrides to ATP forecast.
    growth_template: {month_1: 0.02, month_2: 0.01, ...}
    """
    adjusted = []
    for i, base_val in enumerate(base_forecast):
        month_num = (i // 30) + 1
        growth_pct = growth_template.get(f'month_{month_num}', 0.0)
        
        if i == 0:
            prev_val = base_val
        else:
            prev_val = adjusted[-1]
        
        adjusted_val = prev_val * (1 + growth_pct)
        adjusted.append(adjusted_val)
    
    return adjusted

def compute_transaction_forecast(total_sales_forecast, atp_forecast):
    """
    Compute transaction count = total_sales / atp
    """
    transactions = []
    for ts, atp in zip(total_sales_forecast, atp_forecast):
        if atp > 0:
            transactions.append(ts / atp)
        else:
            transactions.append(0)
    return transactions

def reconcile_forecasts(direct_total, integrated_total, threshold=0.10):
    """
    Compare direct total forecast vs integrated (transactions × ATP) forecast.
    Returns list of divergences.
    """
    divergences = []
    for i, (direct, integrated) in enumerate(zip(direct_total, integrated_total)):
        if max(direct, integrated) > 0:
            pct_diff = abs(direct - integrated) / max(direct, integrated)
            if pct_diff > threshold:
                divergences.append({
                    'day': i,
                    'direct': direct,
                    'integrated': integrated,
                    'divergence_pct': pct_diff
                })
    return divergences

# ── HOLT-WINTERS (pure Python fallback, always runs) ──────────────────────────
def hw_core(y, m, a, b, g):
    n = len(y)
    L = (sum(y[:m]) / m) if m <= n else (sum(y) / n or 1.0)
    T = 0.0
    if 2 * m <= n:
        for i in range(m):
            T += (y[m + i] - y[i]) / (m * m)
    S = [y[i] / L if L else 1.0 for i in range(min(m, n))]
    while len(S) < m:
        S.append(1.0)
    fitted = []
    for t in range(n):
        si, pL = t % m, L
        fitted.append((L + T) * (S[si] or 1.0))
        L = a * (y[t] / (S[si] or 1.0)) + (1 - a) * (L + T)
        T = b * (L - pL) + (1 - b) * T
        S[si] = g * (y[t] / (L or 1.0)) + (1 - g) * (S[si] or 1.0)
    return L, T, S, fitted

def hw_forecast(y, m, a, b, g, h):
    if not y or len(y) < m * 2:
        n = max(len(y), 1)
        xm = (n - 1) / 2
        ym = mean(y)
        num = sum((i - xm) * (v - ym) for i, v in enumerate(y))
        den = sum((i - xm) ** 2 for i in range(n)) or 1
        sl = num / den; ic = ym - sl * xm
        fi = [max(0, ic + sl * i) for i in range(n)]
        fc = [max(0, ic + sl * (n + i)) for i in range(h)]
        rmse = math.sqrt(sum((v - fi[i]) ** 2 for i, v in enumerate(y)) / n)
        cv = rmse / (mean(y) or 1)
        lo = [max(0, v * (1 - 1.645 * cv)) for v in fc]
        hi = [v * (1 + 1.645 * cv) for v in fc]
        return fi, fc, lo, hi, rmse
    L, T, S, fi = hw_core(y, m, a, b, g)
    n = len(y)
    fc = [max(0, (L + (i + 1) * T) * (S[(n + i) % m] or 1.0)) for i in range(h)]
    rmse = math.sqrt(sum((v - fi[i]) ** 2 for i, v in enumerate(y)) / n)
    cv = rmse / (mean(y) or 1)
    lo = [max(0, v * (1 - 1.645 * cv)) for v in fc]
    hi = [v * (1 + 1.645 * cv) for v in fc]
    return fi, fc, lo, hi, rmse

def optim_hw(y, m):
    best = (0.3, 0.05, 0.6, float("inf"))
    if not y or len(y) < m * 2:
        return best[:3]
    for a in [0.1, 0.2, 0.3, 0.5, 0.7]:
        for b in [0.01, 0.05, 0.1, 0.2]:
            for g in [0.2, 0.4, 0.6, 0.8]:
                try:
                    *_, rmse = hw_forecast(y, m, a, b, g, 1)
                    if math.isfinite(rmse) and rmse < best[3]:
                        best = (a, b, g, rmse)
                except Exception:
                    pass
    return best[0], best[1], best[2]

# ── SARIMA ─────────────────────────────────────────────────────────────────────
def run_sarima(dates, values, h):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    series = (pd.Series(values, index=pd.DatetimeIndex(dates))
    .asfreq("D")
    .fillna(method="ffill")
    .fillna(0))
    configs = [
        ((1, 1, 1), (1, 1, 1, 7)),
        ((1, 1, 0), (1, 1, 0, 7)),
        ((2, 1, 1), (0, 1, 1, 7)),
        ((1, 1, 1), (0, 0, 0, 0)),
    ]
    for order, seas in configs:
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
            log.info(f"SARIMA{order}x{seas} OK RMSE={rmse:.2f}")
            return fi, fc, lo, hi, rmse, f"SARIMA{order}×{seas}"
        except Exception as e:
            log.warning(f"SARIMA {order}x{seas} failed: {e}")
    raise RuntimeError("All SARIMA configurations failed")

# ── PROPHET ────────────────────────────────────────────────────────────────────
def analyse_venue_history(dates, values):
    """Analyse venue trading history."""
    from datetime import datetime as _dt
    n_days = len(dates)
    non_zero = [v for v in values if v > 0]
    n_trading = len(non_zero)
    if not non_zero:
        return dict(n_days=n_days, n_trading=0, patchiness=1.0,
        stable_mean=0, recent_mean=0, recent_max=0,
        dow_means={}, is_patchy=True, use_flat_forecast=True,
        bypass_prophet=True)
    patchiness = 1 - (n_trading / n_days) if n_days > 0 else 0
    sorted_nz = sorted(non_zero)
    mid = len(sorted_nz) // 2
    stable_mean = (sorted_nz[mid-1] + sorted_nz[mid]) / 2 if len(sorted_nz) % 2 == 0 else sorted_nz[mid]
    recent_nz = [v for v in values[-56:] if v > 0][-28:]
    recent_mean = mean(recent_nz) if recent_nz else stable_mean
    recent_max = max(recent_nz) if recent_nz else stable_mean
    dow_acc = {i: [] for i in range(7)}
    for ds, v in zip(dates, values):
        if v > 0:
            try:
                d = _dt.strptime(ds, "%Y-%m-%d")
                dow_acc[d.weekday()].append(v)
            except Exception:
                pass
    dow_means = {d: mean(vs) for d, vs in dow_acc.items() if vs}
    is_patchy = patchiness > 0.30 or n_trading < 21
    bypass_prophet = n_trading < 21
    use_flat_forecast = n_trading < 28 or (is_patchy and n_trading < 56)
    return dict(
    n_days=n_days, n_trading=n_trading, patchiness=patchiness,
    stable_mean=stable_mean, recent_mean=recent_mean, recent_max=recent_max,
    dow_means=dow_means, is_patchy=is_patchy,
    use_flat_forecast=use_flat_forecast, bypass_prophet=bypass_prophet
    )

def prophet_params_for_history(n_days, analysis=None):
    """Dynamically tune Prophet based on history length."""
    is_patchy = analysis.get("is_patchy", False) if analysis else False
    if n_days < 30 or (analysis and analysis["n_trading"] < 14):
        return dict(
        yearly_seasonality = False,
        weekly_seasonality = False,
        changepoint_prior_scale = 0.0001,
        seasonality_prior_scale = 0.1,
        n_changepoints = 0,
        )
    elif n_days < 60 or is_patchy:
        return dict(
        yearly_seasonality = False,
        weekly_seasonality = True,
        changepoint_prior_scale = 0.001,
        seasonality_prior_scale = 1,
        n_changepoints = 2,
        )
    elif n_days < 180:
        return dict(
        yearly_seasonality = False,
        weekly_seasonality = True,
        changepoint_prior_scale = 0.01,
        seasonality_prior_scale = 5,
        n_changepoints = 5,
        )
    elif n_days < 365:
        return dict(
        yearly_seasonality = False,
        weekly_seasonality = True,
        changepoint_prior_scale = 0.03,
        seasonality_prior_scale = 8,
        n_changepoints = 10,
        )
    else:
        return dict(
        yearly_seasonality = True,
        weekly_seasonality = True,
        changepoint_prior_scale = 0.05,
        seasonality_prior_scale = 10,
        n_changepoints = 25,
        )

def dow_flat_forecast(dates, analysis, h, last_date_str):
    """DOW-weighted flat forecast for new/patchy venues."""
    from datetime import datetime as _dt, timedelta as _td
    dow_means = analysis["dow_means"]
    recent = analysis["recent_mean"]
    recent_max = analysis["recent_max"]
    ceiling = max(recent_max * 1.5, recent * 2.0)
    last_d = _dt.strptime(last_date_str, "%Y-%m-%d")
    fc_out, lo_out, hi_out = [], [], []
    for i in range(h):
        d = last_d + _td(days=i+1)
        dow = d.weekday()
        base = dow_means.get(dow, recent)
        base = min(base, ceiling)
        fc_out.append(base)
        lo_out.append(max(0, base * 0.80))
        hi_out.append(min(ceiling, base * 1.20))
    log.info(f"DOW-flat forecast: {h} days, recent_mean={recent:.0f}, ceiling={ceiling:.0f}")
    return fc_out, lo_out, hi_out

def sanitise_forecast(fc, lo, hi, analysis, h, dates=None):
    """Hard-cap forecast to prevent absurd values."""
    stable = analysis["stable_mean"]
    recent = analysis["recent_mean"]
    recent_max = analysis["recent_max"]
    n_trade = analysis["n_trading"]
    if stable <= 0 and recent <= 0:
        return [0.0]*h, [0.0]*h, [0.0]*h
    base = max(stable, recent, 1.0)
    if n_trade < 28:
        ceiling = max(recent_max * 1.5, recent * 2.0)
    elif n_trade < 90:
        ceiling = max(recent_max * 2.0, stable * 3.0)
    else:
        ceiling = stable * 4.0
    fc_out = [min(max(0, v), ceiling) for v in fc]
    lo_out, hi_out = [], []
    for i, (f_orig, f_new, l, hv) in enumerate(zip(fc, fc_out, lo, hi)):
        scale = (f_new / f_orig) if f_orig > 0 else 1.0
        lo_out.append(max(0, l * scale))
        hi_out.append(min(ceiling, hv * scale))
    n_clamped = sum(1 for a, b in zip(fc, fc_out) if abs(a-b) > 1)
    if n_clamped:
        log.info(f"Clamped {n_clamped}/{h} forecast days to ceiling={ceiling:.0f} (recent_max={recent_max:.0f})")
    return fc_out, lo_out, hi_out

def run_prophet(dates, values, h, holiday_dates=None, weather_map=None, atp_mode=False, dow_encoded=None):
    from prophet import Prophet
    df = pd.DataFrame({
    "ds": pd.to_datetime(dates),
    "y": [max(0, v) for v in values],
    })
    hols_df = None
    if holiday_dates:
        hols_df = pd.DataFrame({
        "holiday": "public_holiday",
        "ds": pd.to_datetime(holiday_dates),
        "lower_window": 0,
        "upper_window": 1,
        })
    use_wx = bool(weather_map)
    analysis = analyse_venue_history(dates, values)
    n_days = analysis["n_days"]
    log.info(
    f"Venue analysis: {n_days}d history, {analysis['n_trading']} trading days, "
    f"patchiness={analysis['patchiness']:.0%}, bypass_prophet={analysis['bypass_prophet']}, "
    f"stable_mean={analysis['stable_mean']:.0f}, recent_max={analysis['recent_max']:.0f}"
    )
    
    # ATP mode: less seasonality, more stable
    if atp_mode:
        analysis['bypass_prophet'] = False
    
    if analysis["bypass_prophet"]:
        log.info(f"Bypassing Prophet — using DOW-flat forecast for new venue ({analysis['n_trading']} trading days)")
        fi = [mean([v for v in values if v > 0] or [0])] * n_days
        fc, lo, hi = dow_flat_forecast(dates, analysis, h, dates[-1])
        flat = mean([v for v in values if v > 0] or [0])
        rmse = math.sqrt(mean([(v - flat)**2 for v in values]))
        comps = {}
        raise _BypassProphet(fi, fc, lo, hi, rmse, comps)
    
    params = prophet_params_for_history(n_days, analysis)
    
    # ATP mode: reduce seasonality
    if atp_mode:
        params['seasonality_prior_scale'] = params['seasonality_prior_scale'] / 2
        params['yearly_seasonality'] = False
    
    log.info(
    f"Prophet config: changepoint_prior={params['changepoint_prior_scale']}, "
    f"yearly={params['yearly_seasonality']}, n_changepoints={params['n_changepoints']}"
    )
    m = Prophet(
    yearly_seasonality = params["yearly_seasonality"],
    weekly_seasonality = params["weekly_seasonality"],
    daily_seasonality = False,
    holidays = hols_df,
    changepoint_prior_scale = params["changepoint_prior_scale"],
    seasonality_prior_scale = params["seasonality_prior_scale"],
    holidays_prior_scale = 10,
    seasonality_mode = "multiplicative",
    interval_width = 0.90,
    n_changepoints = params["n_changepoints"],
    )
    
    # ENHANCEMENT 1: Weather × DOW interaction
    if use_wx:
        m.add_regressor("temp_max", standardize=True)
        df["temp_max"] = df["ds"].dt.strftime("%Y-%m-%d").map(
        lambda d: (weather_map.get(d) or {}).get("tMax", None))
        df["temp_max"] = pd.to_numeric(df["temp_max"], errors="coerce")
        med = df["temp_max"].median()
        df["temp_max"] = df["temp_max"].fillna(med if not math.isnan(float(med)) else 20.0)
        
        # Add interaction term if dow_encoded provided
        if dow_encoded is not None:
            m.add_regressor("weather_dow_interaction", standardize=True)
            df["weather_dow_interaction"] = df["temp_max"] * dow_encoded
    
    m.fit(df)
    future = m.make_future_dataframe(periods=h)
    if use_wx:
        clim = float(df["temp_max"].mean())
        future["temp_max"] = future["ds"].dt.strftime("%Y-%m-%d").map(
        lambda d: (weather_map.get(d) or {}).get("tMax", None))
        future["temp_max"] = pd.to_numeric(future["temp_max"], errors="coerce").fillna(clim)
        
        if dow_encoded is not None:
            future["weather_dow_interaction"] = future["temp_max"] * dow_encoded
    
    fc_df = m.predict(future)
    n = len(dates)
    fi = [max(0, v) for v in fc_df["yhat"].iloc[:n].tolist()]
    fc = [max(0, v) for v in fc_df["yhat"].iloc[n:].tolist()]
    lo = [max(0, v) for v in fc_df["yhat_lower"].iloc[n:].tolist()]
    hi = [max(0, v) for v in fc_df["yhat_upper"].iloc[n:].tolist()]
    resid = [v - f for v, f in zip(values, fi)]
    rmse = math.sqrt(sum(r ** 2 for r in resid) / len(resid)) if resid else 0
    
    # Apply sanity caps
    fc, lo, hi = sanitise_forecast(fc, lo, hi, analysis, h)
    
    # Components
    comps = {}
    for col in ["trend", "weekly", "yearly", "holidays"]:
        if col in fc_df.columns:
            comps[col] = {
            "hist": [safe(v) for v in fc_df[col].iloc[:n].tolist()],
            "forecast": [safe(v) for v in fc_df[col].iloc[n:].tolist()],
            }
    
    # ENHANCEMENT 1: Add interaction component
    if use_wx and dow_encoded is not None and "weather_dow_interaction" in fc_df.columns:
        comps["weather_dow_interaction"] = {
        "hist": [safe(v) for v in fc_df["weather_dow_interaction"].iloc[:n].tolist()],
        "forecast": [safe(v) for v in fc_df["weather_dow_interaction"].iloc[n:].tolist()],
        }
    
    log.info(f"Prophet OK RMSE={rmse:.2f} stable_mean={analysis['stable_mean']:.0f} patchy={analysis['is_patchy']}")
    return fi, fc, lo, hi, rmse, comps

def _run_prophet_safe(dates, values, h, holiday_dates=None, weather_map=None, atp_mode=False, dow_encoded=None):
    """Wrapper that catches _BypassProphet."""
    try:
        return run_prophet(dates, values, h, holiday_dates=holiday_dates, weather_map=weather_map, 
                          atp_mode=atp_mode, dow_encoded=dow_encoded)
    except _BypassProphet as bp:
        log.info(f"DOW-flat bypass returned: {len(bp.fc)} forecast days, max={max(bp.fc):.0f}")
        return bp.fi, bp.fc, bp.lo, bp.hi, bp.rmse, bp.comps

def forecast_with_fallback(dates, values, h, holiday_dates=None, weather_map=None, atp_mode=False, dow_encoded=None):
    """Main forecasting pipeline with three-tier fallback."""
    if not dates or not values or len(dates) != len(values):
        return [], [], [], [], 0, "error"
    
    last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
    forecast_dates = [iso(add_days(last_date, i + 1)) for i in range(h)]
    
    fi = fc = lo = hi = []
    rmse = 0.0
    model_used = None
    comps = {}
    warns = []
    
    # 1 – Prophet
    try:
        fi, fc, lo, hi, rmse, comps = _run_prophet_safe(
        dates, values, h, holiday_dates=holiday_dates, weather_map=weather_map,
        atp_mode=atp_mode, dow_encoded=dow_encoded)
        model_used = "Prophet" + (f" ({len(dates)}d)" if len(dates) < 365 else "")
    except ImportError:
        warns.append("Prophet not installed — trying SARIMA")
    except Exception as e:
        warns.append(f"Prophet failed ({str(e)[:100]}) — trying SARIMA")
        log.warning(traceback.format_exc())
    
    # 2 – SARIMA
    if model_used is None:
        try:
            fi, fc, lo, hi, rmse, lbl = run_sarima(dates, values, h)
            model_used = lbl
        except ImportError:
            warns.append("statsmodels not installed — using Holt-Winters")
        except Exception as e:
            warns.append(f"SARIMA failed ({str(e)[:100]}) — using Holt-Winters")
    
    # 3 – Holt-Winters
    if model_used is None:
        M = 7
        a, b, g = optim_hw(values, M)
        fi, fc, lo, hi, rmse = hw_forecast(values, M, a, b, g, h)
        analysis_hw = analyse_venue_history(dates, values)
        fc, lo, hi = sanitise_forecast(fc, lo, hi, analysis_hw, h)
        model_used = f"Holt-Winters (a={a:.2f} b={b:.2f} g={g:.2f})"
        warns.append("Using Holt-Winters fallback")
    
    return fi, fc, lo, hi, rmse, model_used

# ── /forecast ──────────────────────────────────────────────────────────────────
@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        body = request.get_json(force=True)
        dates = body.get("dates", [])
        values = [float(v) for v in body.get("values", [])]
        h = max(1, min(730, int(body.get("forecast_days", 30))))
        holiday_dates = body.get("holiday_dates", [])
        weather_map = body.get("weather_map", {})
        
        # ENHANCEMENT 2: Monthly multipliers
        multiplier_map = body.get("monthly_multipliers", {})
        
        # ENHANCEMENT 6: ATP data and growth overrides
        atp_history = body.get("atp_history", [])
        atp_growth_overrides = body.get("atp_growth_overrides", {})
        
        # ENHANCEMENT 5: Multi-component sales
        retail_values = body.get("retail_values", [])
        discount_values = body.get("discount_values", [])
        
        if not dates or not values or len(dates) != len(values):
            return jsonify({"error": "dates and values required and must match length"}), 400
        
        last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
        forecast_dates = [iso(add_days(last_date, i + 1)) for i in range(h)]
        
        # Main forecast (total sales)
        fi, fc, lo, hi, rmse, model_used = forecast_with_fallback(
        dates, values, h, holiday_dates=holiday_dates, weather_map=weather_map)
        
        # ENHANCEMENT 2: Apply monthly multipliers
        if multiplier_map:
            venue_id = body.get("venue_id", "Unknown")
            fc_mult = []
            for i, (date_str, forecast_val) in enumerate(zip(forecast_dates, fc)):
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                mult = get_monthly_multiplier(venue_id, date_obj, {venue_id: multiplier_map.get(venue_id, [])})
                fc_mult.append(forecast_val * mult)
            multipliers_applied = [get_monthly_multiplier(venue_id, datetime.strptime(d, "%Y-%m-%d"), 
                                                         {venue_id: multiplier_map.get(venue_id, [])}) 
                                 for d in forecast_dates]
        else:
            fc_mult = fc
            multipliers_applied = [1.0] * len(forecast_dates)
        
        # ENHANCEMENT 5: Multi-component forecasts
        components_forecast = {}
        if retail_values and discount_values:
            components_forecast = forecast_sales_components(
            body.get("venue_id", "Unknown"), dates, values, retail_values, discount_values,
            h, holiday_dates, weather_map)
        
        # ENHANCEMENT 6 & 7: ATP and transaction forecasts
        atp_forecast = {}
        transaction_forecast = {}
        integrated_forecast = []
        reconciliation = []
        
        if atp_history and len(atp_history) > 0:
            atp_dates = body.get("atp_dates", dates)
            atp_forecast = forecast_average_ticket_price(atp_dates, atp_history, h, 
                                                        holiday_dates, weather_map, atp_growth_overrides)
            
            # Compute transactions
            transactions = compute_transaction_forecast(fc_mult, atp_forecast.get('forecast', fc_mult))
            transaction_forecast = {
            'forecast': transactions,
            'lower_90': [t * 0.85 for t in transactions],
            'upper_90': [t * 1.15 for t in transactions],
            'model_used': 'derived'
            }
            
            # Integrated forecast
            integrated_forecast = [t * a for t, a in zip(transactions, atp_forecast.get('forecast', fc_mult))]
            
            # Reconciliation
            reconciliation = reconcile_forecasts(fc_mult, integrated_forecast)
        
        mv = mean(values) or 1
        
        result = {
        "model": model_used,
        "rmse": round(rmse, 4),
        "cv": round(rmse / mv, 4),
        "fitted": [safe(v) for v in fi],
        "forecast_dates": forecast_dates,
        "forecast": [safe(v) for v in fc_mult],
        "lower_90": [safe(v) for v in lo],
        "upper_90": [safe(v) for v in hi],
        "monthly_multipliers": multipliers_applied,
        "components": {},
        "warnings": [],
        }
        
        # Add component forecasts if available
        if components_forecast:
            result["components_forecast"] = {
            k: {**v, 'forecast': [safe(x) for x in v['forecast']], 
                'lower_90': [safe(x) for x in v['lower_90']], 
                'upper_90': [safe(x) for x in v['upper_90']]} 
            for k, v in components_forecast.items()
            }
        
        # Add ATP and transaction forecasts
        if atp_forecast:
            result["atp_forecast"] = {
            'forecast': [safe(v) for v in atp_forecast.get('forecast', [])],
            'lower_90': [safe(v) for v in atp_forecast.get('lower_90', [])],
            'upper_90': [safe(v) for v in atp_forecast.get('upper_90', [])],
            'model_used': atp_forecast.get('model_used', 'Prophet'),
            'rmse': atp_forecast.get('rmse', 0)
            }
        
        if transaction_forecast:
            result["transaction_forecast"] = {
            'forecast': [safe(v) for v in transaction_forecast.get('forecast', [])],
            'lower_90': [safe(v) for v in transaction_forecast.get('lower_90', [])],
            'upper_90': [safe(v) for v in transaction_forecast.get('upper_90', [])],
            'model_used': transaction_forecast.get('model_used', '')
            }
            
            result["integrated_forecast"] = [safe(v) for v in integrated_forecast]
            result["reconciliation_divergences"] = reconciliation
        
        return jsonify(result)
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ── /health ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    libs = {}
    for lib in ["prophet", "statsmodels", "numpy", "pandas", "flask"]:
        try:
            __import__(lib); libs[lib] = "ok"
        except ImportError:
            libs[lib] = "missing"
    return jsonify({"status": "ok", "libraries": libs})

# ── /venue-template (CSV download) ───────────────────────────────────────────
@app.route("/venue-template", methods=["GET"])
def venue_template():
    rows = [
    "venue,city,state,country_code,latitude,longitude,holiday_multiplier,temp_sensitivity",
    "Sydney CBD,Sydney,NSW,AU,-33.8688,151.2093,1.10,",
    "Melbourne Central,Melbourne,VIC,AU,-37.8136,144.9631,1.10,",
    "Brisbane City,Brisbane,QLD,AU,-27.4698,153.0251,1.05,",
    "Gold Coast,Gold Coast,QLD,AU,-28.0167,153.4000,1.15,",
    "Perth CBD,Perth,WA,AU,-31.9505,115.8605,1.10,",
    "Adelaide CBD,Adelaide,SA,AU,-34.9285,138.6007,1.10,",
    "Canberra,Canberra,ACT,AU,-35.2809,149.1300,1.05,",
    "Singapore Orchard,Singapore,SGP,SG,1.3048,103.8318,1.20,",
    "London West End,London,ENG,GB,51.5074,-0.1278,1.05,",
    "Houston Texas,Houston,TX,US,29.7604,-95.3698,1.10,",
    ]
    return Response(
    "\n".join(rows),
    mimetype="text/csv",
    headers={"Content-Disposition": "attachment; filename=venue_city_mapping_template.csv"}
    )

# ── /forecast-multi ────────────────────────────────────────────────────────────
@app.route("/forecast-multi", methods=["POST"])
def forecast_multi():
    try:
        body = request.get_json(force=True)
        venues_data = body.get("venues", [])
        if not venues_data:
            return jsonify({"error": "No venue data provided"}), 400
        results = {}
        for vd in venues_data:
            name = vd.get("name", "Unknown")
            dates = vd.get("dates", [])
            values = [float(v) for v in vd.get("values", [])]
            h = max(1, min(730, int(vd.get("forecast_days", 30))))
            holiday_dates = vd.get("holiday_dates", [])
            weather_map = vd.get("weather_map", {})
            if not dates or not values or len(dates) != len(values):
                results[name] = {"error": f"Invalid data for venue '{name}'"}
                continue
            last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
            forecast_dates = [iso(add_days(last_date, i + 1)) for i in range(h)]
            fi, fc, lo, hi, rmse, model_used = forecast_with_fallback(
            dates, values, h, holiday_dates=holiday_dates, weather_map=weather_map)
            mv = mean(values) or 1
            results[name] = {
            "model": model_used, "rmse": round(rmse, 4), "cv": round(rmse / mv, 4),
            "fitted": [safe(v) for v in fi], "forecast_dates": forecast_dates,
            "forecast": [safe(v) for v in fc], "lower_90": [safe(v) for v in lo],
            "upper_90": [safe(v) for v in hi], "warnings": [],
            }
            log.info(f"[{name}] {model_used} RMSE={rmse:.2f}")
        return jsonify({"venues": results})
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
