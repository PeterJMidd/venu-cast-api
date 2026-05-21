# Architecture Reference

## Data Flow

```
User uploads 7 Excel files
        │
        ▼
┌──────────────────────────────────┐
│   excel-parser.js (SheetJS)      │  Parses + normalises venue names
└──────────────────────────────────┘
        │
        ├──▶ sales_history (daily, 24mo)
        ├──▶ venue_details + ramp_up
        ├──▶ avg_ticket (12 budget months)
        ├──▶ labour assumptions
        ├──▶ cogs assumptions (% by category)
        ├──▶ rent assumptions
        └──▶ prior_pnl (12mo, line items)
                 │
                 ▼
┌──────────────────────────────────┐
│  weather.js                      │  Open-Meteo archive API
│  - 24mo historical weather       │
│  - temp band → sales index       │
└──────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  sales-forecast.js               │
│  ┌─ API path: /forecast-multi ───┼──▶ venu-cast-api (Prophet/SARIMA)
│  └─ Local: seasonality model     │
│  + ramp-up multiplier per day    │
└──────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  Daily forecasts (venue × date)  │
└──────────────────────────────────┘
                 │
        ┌────────┼────────┬────────────┐
        ▼        ▼        ▼            ▼
   cogs-calc labour-calc rent-calc  pnl-builder
   (food,    (SPLH→hrs   (base +    (assemble
    pkg,     × rate +    pct rent   daily P&L,
    retail,  oncosts +   when sales aggregate to
    disc%)   mgmt)       > threshold) monthly)
        │        │        │            │
        └────────┴────┬───┴────────────┘
                      ▼
        ┌──────────────────────────────────┐
        │  daily_forecast (~36,500 rows)   │
        │  monthly_summary (~1,200 rows)   │
        └──────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   charts.js     export.js     supabase-client.js
   (dashboards)  (.xlsx)       (batch upserts)
```

## Calculation Order

The order matters because rent's percentage component needs monthly sales:

1. **Sales forecast** — produces gross_sales = net_sales (no GST in this model)
2. **COGS** — applied to net_sales (food/packaging/retail) and gross_sales (discounts)
3. **Gross profit** — net_sales − cogs_total
4. **Labour** — derived from net_sales via SPLH
5. **Rent (two-pass)**:
   - Pass 1: aggregate monthly sales per venue
   - Pass 2: if monthly sales > threshold, calculate excess × rate, distribute evenly across days
6. **Venue contribution** — gross_profit − labour_total − occupancy_total

## Seasonality Model (Local Fallback)

When the API is unavailable, the forecast engine uses three multiplicative indices:

```
forecast = base_daily × dow_index × month_index × holiday_adj × weather_index × ramp_up
```

- **dow_index** — average sales by day-of-week (Mon-Sun) ÷ overall avg
- **month_index** — average sales by month (Jan-Dec) ÷ overall avg
- **holiday_adj** — 1.15 default for public holidays, 1.10 for school holidays (recalculated from history if data available)
- **weather_index** — sales index by 5°C temperature band, looked up by expected month temperature
- **ramp_up** — 0 before opening, custom 1-18 month curve from venue_details, then 1.0

## Ramp-Up Logic

```
months_open = (forecast_date.year - opening_date.year) * 12
            + (forecast_date.month - opening_date.month)

if forecast_date < opening_date:
    multiplier = 0  (venue not open yet, sales = 0)
elif months_open >= 18:
    multiplier = 1.0  (mature venue)
else:
    multiplier = ramp_up_data[venue][months_open]
```

## Supabase Write Strategy

For 100 venues × 365 days = ~36,500 daily rows per run:

1. Create `budget_runs` row → get `run_id`
2. Upsert `venues` (idempotent on venue_name)
3. Upsert assumption tables tagged with `run_id`
4. Batch-upsert `daily_forecast` in chunks of 500 (~73 round trips)
5. Upsert `monthly_summary` (~1,200 rows, single batch)

Total upload time: ~30-60 seconds depending on connection.

## Venue Name Matching

All 7 templates must reference the same venues. The parser normalises names:
- Trim whitespace
- Lowercase
- Collapse multiple spaces to single

`venue_key = "Bondi Junction"` and `"bondi  junction"` both → `"bondi junction"`.

The Venue Details template is the master list. Cross-template validation flags any venue in another template that isn't in the master.

## Why Prophet for Forecasting?

The venu-cast-api uses Prophet (with SARIMA + Holt-Winters fallbacks) because:
- **Multiple seasonalities** — captures both weekly (DOW) and yearly cycles natively
- **Holiday effects** — built-in holiday regressors
- **Trend changepoints** — detects regime shifts (e.g. post-renovation)
- **Confidence intervals** — provides 90% CI for risk-aware budgeting
- **Robust to missing data** — handles closure days, partial months

The local seasonality fallback in the budget model is a simpler multiplicative decomposition that's "good enough" when the API is offline.
