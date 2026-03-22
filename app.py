"""
Venu Cast — Forecast API
Models: Prophet (primary) -> SARIMA (fallback) -> Holt-Winters (always available)
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

# cmdstan is pre-installed via requirements.txt (cmdstanpy)
# Do NOT call install_cmdstan() at startup — it blocks port binding on Render

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
            fi  = [max(0, v) for v in res.fittedvalues.tolist()]
            fcr = res.get_forecast(steps=h)
            fc  = [max(0, v) for v in fcr.predicted_mean.tolist()]
            ci  = fcr.conf_int(alpha=0.10)
            lo  = [max(0, v) for v in ci.iloc[:, 0].tolist()]
            hi  = [max(0, v) for v in ci.iloc[:, 1].tolist()]
            rmse = math.sqrt(sum(r ** 2 for r in res.resid.tolist()) / len(res.resid))
            log.info(f"SARIMA{order}x{seas} OK  RMSE={rmse:.2f}")
            return fi, fc, lo, hi, rmse, f"SARIMA{order}×{seas}"
        except Exception as e:
            log.warning(f"SARIMA {order}x{seas} failed: {e}")
    raise RuntimeError("All SARIMA configurations failed")


# ── PROPHET ────────────────────────────────────────────────────────────────────

# No trim needed — CSV only contains rows for active trading days


def analyse_venue_history(dates, values):
    """
    Analyse a venue's trading history to detect:
    - n_days: calendar days of history
    - n_trading: actual trading days with sales
    - patchiness: fraction of days with no sales (0=consistent, 1=all gaps)
    - consistency_score: how consistent the non-zero days are (CoV-based)
    - stable_mean: robust average of consistent trading days (median of non-zero)
    - recent_mean: mean of last 28 days of non-zero sales
    - is_patchy: True if venue has large gaps or very few trading days
    """
    n_days = len(dates)
    non_zero = [v for v in values if v > 0]
    n_trading = len(non_zero)

    if not non_zero:
        return dict(n_days=n_days, n_trading=0, patchiness=1.0,
                    consistency_score=0, stable_mean=0, recent_mean=0,
                    is_patchy=True, use_flat_forecast=True)

    patchiness = 1 - (n_trading / n_days) if n_days > 0 else 0

    # Coefficient of variation of non-zero days
    m = mean(non_zero)
    s = stdev(non_zero)
    cov = s / m if m > 0 else 0

    # Stable mean = median of non-zero values (robust to outlier opening days)
    sorted_nz = sorted(non_zero)
    mid = len(sorted_nz) // 2
    stable_mean = (sorted_nz[mid-1] + sorted_nz[mid]) / 2 if len(sorted_nz) % 2 == 0 else sorted_nz[mid]

    # Recent mean = last 28 trading days
    recent_nz = [v for d, v in zip(dates[-56:], values[-56:]) if v > 0][-28:]
    recent_mean = mean(recent_nz) if recent_nz else stable_mean

    # Patchy = more than 30% gaps OR fewer than 21 trading days
    is_patchy = patchiness > 0.30 or n_trading < 21

    # Use flat forecast if very new or very patchy
    use_flat_forecast = n_trading < 14 or (is_patchy and n_trading < 28)

    return dict(
        n_days=n_days, n_trading=n_trading, patchiness=patchiness,
        cov=cov, stable_mean=stable_mean, recent_mean=recent_mean,
        is_patchy=is_patchy, use_flat_forecast=use_flat_forecast
    )


def prophet_params_for_history(n_days, analysis=None):
    """
    Dynamically tune Prophet based on history length and trading consistency.
    Patchy/new venues get very conservative flat-trend models.
    """
    is_patchy = analysis.get("is_patchy", False) if analysis else False

    if n_days < 30 or (analysis and analysis["n_trading"] < 14):
        # Too little data for Prophet to learn anything useful
        return dict(
            yearly_seasonality      = False,
            weekly_seasonality      = False,
            changepoint_prior_scale = 0.0001,
            seasonality_prior_scale = 0.1,
            n_changepoints          = 0,
        )
    elif n_days < 60 or is_patchy:
        # Very new or patchy: near-flat trend, weekly cycle only
        return dict(
            yearly_seasonality      = False,
            weekly_seasonality      = True,
            changepoint_prior_scale = 0.001,
            seasonality_prior_scale = 1,
            n_changepoints          = 2,
        )
    elif n_days < 180:
        # New venue: conservative trend
        return dict(
            yearly_seasonality      = False,
            weekly_seasonality      = True,
            changepoint_prior_scale = 0.01,
            seasonality_prior_scale = 5,
            n_changepoints          = 5,
        )
    elif n_days < 365:
        # Growing venue: moderate, no full yearly cycle yet
        return dict(
            yearly_seasonality      = False,
            weekly_seasonality      = True,
            changepoint_prior_scale = 0.03,
            seasonality_prior_scale = 8,
            n_changepoints          = 10,
        )
    else:
        # Mature venue: full model
        return dict(
            yearly_seasonality      = True,
            weekly_seasonality      = True,
            changepoint_prior_scale = 0.05,
            seasonality_prior_scale = 10,
            n_changepoints          = 25,
        )


def sanitise_forecast(fc, lo, hi, analysis, h):
    """
    Apply sanity caps to prevent a forecast running above a normal range.

    Rules:
    1. No daily forecast value > 3x the venue's stable_mean (absolute cap)
    2. For patchy/new venues, clamp to a flat band around recent_mean
    3. CI band cannot exceed 2x the forecast value
    4. No negative values
    """
    stable   = analysis["stable_mean"]
    recent   = analysis["recent_mean"]
    is_patch = analysis["is_patchy"]
    n_trade  = analysis["n_trading"]

    if stable <= 0:
        return [0.0]*h, [0.0]*h, [0.0]*h

    # Absolute ceiling: 3x stable mean for mature, 2x for patchy/new
    ceiling = stable * (2.0 if (is_patch or n_trade < 60) else 3.0)

    # For very new / patchy venues, anchor to a flat band around recent mean
    # Allow ±40% variation around recent mean
    if analysis["use_flat_forecast"]:
        base   = recent if recent > 0 else stable
        fc_out = [min(max(base * 0.6, v), base * 1.4) for v in fc]
        lo_out = [max(0, v * 0.75) for v in fc_out]
        hi_out = [min(ceiling, v * 1.25) for v in fc_out]
        log.info(f"Flat-band forecast applied: base={base:.0f}, ceiling={ceiling:.0f}")
        return fc_out, lo_out, hi_out

    # For normal venues: cap at ceiling, ensure CI not wider than 100% of fc
    fc_out = [min(max(0, v), ceiling) for v in fc]
    lo_out = [max(0, min(l, f)) for l, f in zip(lo, fc_out)]
    hi_out = [min(ceiling, max(h_v, f)) for h_v, f in zip(hi, fc_out)]

    return fc_out, lo_out, hi_out


def run_prophet(dates, values, h, holiday_dates=None, weather_map=None):

    from prophet import Prophet
    df = pd.DataFrame({
        "ds": pd.to_datetime(dates),
        "y":  [max(0, v) for v in values],
    })
    hols_df = None
    if holiday_dates:
        hols_df = pd.DataFrame({
            "holiday":      "public_holiday",
            "ds":           pd.to_datetime(holiday_dates),
            "lower_window": 0,
            "upper_window": 1,
        })
    use_wx = bool(weather_map)

    # Analyse venue history for patchy/new venue detection
    analysis = analyse_venue_history(dates, values)
    n_days   = analysis["n_days"]
    params   = prophet_params_for_history(n_days, analysis)

    log.info(
        f"Prophet config: {n_days}d history, {analysis['n_trading']} trading days, "
        f"patchiness={analysis['patchiness']:.0%}, patchy={analysis['is_patchy']}, "
        f"stable_mean={analysis['stable_mean']:.0f}, "
        f"changepoint_prior={params['changepoint_prior_scale']}, yearly={params['yearly_seasonality']}"
    )

    m = Prophet(
        yearly_seasonality      = params["yearly_seasonality"],
        weekly_seasonality      = params["weekly_seasonality"],
        daily_seasonality       = False,
        holidays                = hols_df,
        changepoint_prior_scale = params["changepoint_prior_scale"],
        seasonality_prior_scale = params["seasonality_prior_scale"],
        holidays_prior_scale    = 10,
        seasonality_mode        = "multiplicative",
        interval_width          = 0.90,
        n_changepoints          = params["n_changepoints"],
    )
    if use_wx:
        m.add_regressor("temp_max", standardize=True)
        df["temp_max"] = df["ds"].dt.strftime("%Y-%m-%d").map(
            lambda d: (weather_map.get(d) or {}).get("tMax", None))
        df["temp_max"] = pd.to_numeric(df["temp_max"], errors="coerce")
        med = df["temp_max"].median()
        df["temp_max"] = df["temp_max"].fillna(med if not math.isnan(float(med)) else 20.0)

    m.fit(df)
    future = m.make_future_dataframe(periods=h)
    if use_wx:
        clim = float(df["temp_max"].mean())
        future["temp_max"] = future["ds"].dt.strftime("%Y-%m-%d").map(
            lambda d: (weather_map.get(d) or {}).get("tMax", None))
        future["temp_max"] = pd.to_numeric(future["temp_max"], errors="coerce").fillna(clim)

    fc_df = m.predict(future)
    n   = len(dates)
    fi  = [max(0, v) for v in fc_df["yhat"].iloc[:n].tolist()]
    fc  = [max(0, v) for v in fc_df["yhat"].iloc[n:].tolist()]
    lo  = [max(0, v) for v in fc_df["yhat_lower"].iloc[n:].tolist()]
    hi  = [max(0, v) for v in fc_df["yhat_upper"].iloc[n:].tolist()]
    resid = [v - f for v, f in zip(values, fi)]
    rmse  = math.sqrt(sum(r ** 2 for r in resid) / len(resid)) if resid else 0

    # Apply sanity caps — prevent forecasts running above normal range
    fc, lo, hi = sanitise_forecast(fc, lo, hi, analysis, h)

    # Components
    comps = {}
    for col in ["trend", "weekly", "yearly", "holidays"]:
        if col in fc_df.columns:
            comps[col] = {
                "hist":     [safe(v) for v in fc_df[col].iloc[:n].tolist()],
                "forecast": [safe(v) for v in fc_df[col].iloc[n:].tolist()],
            }
    log.info(f"Prophet OK  RMSE={rmse:.2f}  stable_mean={analysis['stable_mean']:.0f}  patchy={analysis['is_patchy']}")
    return fi, fc, lo, hi, rmse, comps


# ── /forecast ──────────────────────────────────────────────────────────────────

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        body          = request.get_json(force=True)
        dates         = body.get("dates", [])
        values        = [float(v) for v in body.get("values", [])]
        h             = max(1, min(730, int(body.get("forecast_days", 30))))
        holiday_dates = body.get("holiday_dates", [])
        weather_map   = body.get("weather_map", {})

        if not dates or not values or len(dates) != len(values):
            return jsonify({"error": "dates and values required and must match length"}), 400

        last_date      = datetime.strptime(dates[-1], "%Y-%m-%d")
        forecast_dates = [iso(add_days(last_date, i + 1)) for i in range(h)]

        fi = fc = lo = hi = []
        rmse = 0.0
        model_used = None
        comps = {}
        warns = []

        # 1 ── Prophet
        try:
            fi, fc, lo, hi, rmse, comps = run_prophet(
                dates, values, h,
                holiday_dates=holiday_dates,
                weather_map=weather_map)
            model_used = "Prophet" + (f" ({len(dates)}d)" if len(dates) < 365 else "")
        except ImportError:
            warns.append("Prophet not installed — trying SARIMA")
        except Exception as e:
            warns.append(f"Prophet failed ({str(e)[:100]}) — trying SARIMA")
            log.warning(traceback.format_exc())

        # 2 ── SARIMA
        if model_used is None:
            try:
                fi, fc, lo, hi, rmse, lbl = run_sarima(dates, values, h)
                model_used = lbl
            except ImportError:
                warns.append("statsmodels not installed — using Holt-Winters")
            except Exception as e:
                warns.append(f"SARIMA failed ({str(e)[:100]}) — using Holt-Winters")

        # 3 ── Holt-Winters
        if model_used is None:
            M = 7
            a, b, g = optim_hw(values, M)
            fi, fc, lo, hi, rmse = hw_forecast(values, M, a, b, g, h)
            model_used = f"Holt-Winters (α={a:.2f} β={b:.2f} γ={g:.2f})"
            warns.append("Using Holt-Winters fallback")

        mv = mean(values) or 1
        return jsonify({
            "model":          model_used,
            "rmse":           round(rmse, 4),
            "cv":             round(rmse / mv, 4),
            "fitted":         [safe(v) for v in fi],
            "forecast_dates": forecast_dates,
            "forecast":       [safe(v) for v in fc],
            "lower_90":       [safe(v) for v in lo],
            "upper_90":       [safe(v) for v in hi],
            "components":     comps,
            "warnings":       warns,
        })

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


# ── /venue-template  (CSV download) ───────────────────────────────────────────

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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


# ── /forecast-multi  (per-venue independent forecasts) ────────────────────────

@app.route("/forecast-multi", methods=["POST"])
def forecast_multi():
    try:
        body        = request.get_json(force=True)
        venues_data = body.get("venues", [])
        if not venues_data:
            return jsonify({"error": "No venue data provided"}), 400
        results = {}
        for vd in venues_data:
            name          = vd.get("name", "Unknown")
            dates         = vd.get("dates", [])
            values        = [float(v) for v in vd.get("values", [])]
            h             = max(1, min(730, int(vd.get("forecast_days", 30))))
            holiday_dates = vd.get("holiday_dates", [])
            weather_map   = vd.get("weather_map", {})
            if not dates or not values or len(dates) != len(values):
                results[name] = {"error": f"Invalid data for venue '{name}'"}
                continue
            last_date      = datetime.strptime(dates[-1], "%Y-%m-%d")
            forecast_dates = [iso(add_days(last_date, i + 1)) for i in range(h)]
            fi = fc = lo = hi = []
            rmse = 0.0; model_used = None; comps = {}; warns = []

            try:
                fi, fc, lo, hi, rmse, comps = run_prophet(dates, values, h, holiday_dates=holiday_dates, weather_map=weather_map)
                model_used = "Prophet" + (f" ({len(dates)}d)" if len(dates) < 365 else "")
            except ImportError:
                warns.append("Prophet not installed")
            except Exception as e:
                warns.append(f"Prophet failed ({str(e)[:80]}) -- trying SARIMA")
                log.warning(f"[{name}] Prophet: {e}")
            if model_used is None:
                try:
                    fi, fc, lo, hi, rmse, lbl = run_sarima(dates, values, h)
                    model_used = lbl
                except Exception as e:
                    warns.append(f"SARIMA failed ({str(e)[:80]}) -- using Holt-Winters")
            if model_used is None:
                M = 7
                a, b, g = optim_hw(values, M)
                fi, fc, lo, hi, rmse = hw_forecast(values, M, a, b, g, h)
                # Apply sanity caps to HW fallback too
                analysis_hw = analyse_venue_history(dates, values)
                fc, lo, hi = sanitise_forecast(fc, lo, hi, analysis_hw, h)
                model_used = f"Holt-Winters (a={a:.2f} b={b:.2f} g={g:.2f})"
                warns.append("Using Holt-Winters fallback")
            mv = mean(values) or 1
            results[name] = {
                "model": model_used, "rmse": round(rmse, 4), "cv": round(rmse / mv, 4),
                "fitted": [safe(v) for v in fi], "forecast_dates": forecast_dates,
                "forecast": [safe(v) for v in fc], "lower_90": [safe(v) for v in lo],
                "upper_90": [safe(v) for v in hi], "components": comps, "warnings": warns,
            }
            log.info(f"[{name}] {model_used}  RMSE={rmse:.2f}")
        return jsonify({"venues": results})
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
