# Frozen Yoghurt QSR — Full Stack Forecasting & Budget System

This package contains three integrated components that together form an end-to-end sales forecasting and budget modelling system for a 100-venue frozen yoghurt QSR.

```
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│   budget-model/     │─────▶│   venu-cast-api/    │      │      Supabase       │
│   (this build)      │      │   (Flask + Prophet) │      │   (PostgreSQL)      │
│   HTML/JS SPA       │      │   on Render         │      │   Budget DB         │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
         │                                                          ▲
         │                   ┌─────────────────────┐                │
         └──────────────────▶│   venu-cast/        │                │
                             │   Original SPA      │────────────────┘
                             │   (forecast UI)     │
                             └─────────────────────┘
```

---

## Components

### 1. `budget-model/` — Full P&L Budget Application
The new end-to-end budget model. Reads 7 Excel templates, calls the venu-cast-api for sales forecasts, calculates COGS/labour/rent on a daily basis, and pushes results to Supabase.

**Files:**
- `index.html` — Single-page app with 6 tabs (Upload, Venues, Sales Forecast, P&L Budget, Dashboard, Export)
- `css/styles.css` — Application styling
- `js/config.js` — Configuration: API URLs, state coords, P&L line items
- `js/calendars.js` — Australian public + school holidays by state (2026-2027)
- `js/excel-parser.js` — SheetJS parsers for all 7 Excel templates
- `js/supabase-client.js` — Supabase client with batched upserts
- `js/weather.js` — Open-Meteo historical weather + temperature band correlation
- `js/engine/sales-forecast.js` — Calls venu-cast-api, falls back to local seasonality model
- `js/engine/cogs-calc.js` — COGS by food/packaging/retail/discounts %
- `js/engine/labour-calc.js` — SPLH-driven labour, on-costs, management
- `js/engine/rent-calc.js` — Base rent, outgoings, % rent, marketing levy
- `js/engine/pnl-builder.js` — Orchestrator + monthly aggregation + variance
- `js/charts.js` — Chart.js dashboards (seasonality, weather, P&L, heatmap)
- `js/export.js` — Excel exports + Supabase push + template generator
- `js/app.js` — Tab routing, file uploads, button handlers
- `schema.sql` — Run this in Supabase SQL Editor before pushing data

### 2. `venu-cast-api/` — Forecasting API (Flask + Prophet)
Existing Render-hosted forecasting backend. The budget model calls `/forecast-multi` to generate Prophet/SARIMA/Holt-Winters forecasts per venue.

- `app.py` — Flask app with `/forecast`, `/forecast-multi`, `/health`, `/venue-template` endpoints
- `requirements.txt` — Prophet, Pandas, NumPy, Scikit-learn, Statsmodels, Holidays
- `Procfile`, `build.sh` — Render deployment configuration

### 3. `venu-cast/` — Original Forecasting SPA
Existing single-venue forecasting frontend. Useful for ad-hoc forecasting. The budget model is a superset that includes its own forecast logic.

---

## Setup Guide

### Step 1: Deploy Supabase
1. Create a new Supabase project at https://supabase.com
2. Open the SQL Editor and run `budget-model/schema.sql`
3. Note your Project URL and `anon` key (Settings → API)

### Step 2: Deploy venu-cast-api on Render
The API is already deployed at your Render URL. If you need to redeploy:
1. Push `venu-cast-api/` to a GitHub repo
2. Connect that repo to Render as a Web Service
3. Build command: `bash build.sh`
4. Start command: `gunicorn app:app`
5. Note the deployed URL (e.g. `https://venu-cast-api.onrender.com`)

### Step 3: Run the Budget Model
**Option A — GitHub Pages (recommended):**
1. Push `budget-model/` to a GitHub repo (or commit to your venu-cast repo)
2. Enable GitHub Pages on that repo (Settings → Pages → main branch)
3. Open the published URL

**Option B — Local:**
```bash
cd budget-model
python -m http.server 8080
# Open http://localhost:8080
```

### Step 4: Use the App
1. Open the budget model in your browser
2. **Export tab** → enter Supabase URL + anon key → Test Connection
3. **Upload tab** → enter your Render API URL (e.g. `https://venu-cast-api.onrender.com`)
4. **Upload tab** → click "Download All Templates" to get blank Excel files
5. Fill in the 7 templates with your venue data
6. Drop each file into its upload card
7. Click **Generate Budget**
8. Review results in **Sales Forecast**, **P&L Budget**, **Dashboard** tabs
9. **Export tab** → "Push to Supabase" or download Excel reports

---

## Excel Templates Required

| # | Template | Contents |
|---|----------|----------|
| 1 | Sales History | Daily sales by venue, last 24 months (rows=venues, columns=dates) |
| 2 | Prior P&L | 12-month line-by-line P&L (one sheet per venue) |
| 3 | Venue Details | Venue name, state, opening date + 18-month ramp-up |
| 4 | Average Ticket | Monthly avg ticket by venue (rows=venues, columns=12 budget months) |
| 5 | Labour | SPLH, hourly rate, on-costs %, mgmt salary, mgmt on-costs % |
| 6 | COGS | Food %, packaging %, retail %, sale discounts % |
| 7 | Rent | Base rent monthly, outgoings, % rent threshold + rate, marketing levy % |

The "Download All Templates" button in the Export tab generates a single .xlsx workbook with one sheet per template, pre-populated with example data.

---

## How the Forecast Works

For each venue × day in the budget period:

1. **Base sales** — call venu-cast-api `/forecast-multi` with 24 months of history → Prophet picks up trend, weekly + yearly seasonality
2. **Local fallback** — if API unavailable: trailing avg × DOW index × month index × holiday adjustment × weather index
3. **Ramp-up multiplier** — for venues open < 18 months, apply user-supplied multiplier; before opening date, sales = 0
4. **Transactions** — forecast_sales ÷ avg_ticket_for_month
5. **COGS** — net_sales × food/packaging/retail %; gross_sales × discounts %
6. **Labour** — net_sales ÷ SPLH × hourly rate × (1 + on-costs); + monthly mgmt salary daily allocated
7. **Rent** — base + outgoings (daily); % rent triggered when monthly cumulative sales > threshold; marketing levy on net sales
8. **Contribution** — Net Sales − COGS − Labour − Occupancy

Results are aggregated to monthly summary and joined against prior-year P&L for variance analysis.

---

## Architecture Notes

- **Stateless frontend** — all calculation runs in the browser, no server state
- **Supabase = output store** — assumptions and results persist there with run versioning (`run_id`)
- **Render API = forecast brain** — Prophet handles complex seasonality better than the local fallback
- **Open-Meteo = weather** — free, no API key, used for historical correlation
- **Static calendars** — Australian public + school holidays embedded in `js/calendars.js`

---

## Troubleshooting

- **API call hangs**: Render free tier sleeps after 15 min of inactivity — first call takes ~30s to wake
- **CORS errors**: ensure venu-cast-api has `flask-cors` enabled (already configured)
- **Supabase batch failures**: reduce `SUPABASE_BATCH_SIZE` in `js/config.js` (default 500)
- **Wrong venue names across templates**: validation requires exact match (case-insensitive, whitespace-trimmed) against the Venue Details master list

---

## Repos

- Backend API: https://github.com/PeterJMidd/venu-cast-api
- Original frontend: https://github.com/PeterJMidd/venu-cast
- Budget model: (this folder — push to its own repo)
