# Venu Cast — Yo-Chi FY27 Budget Forecast Model

A Flask + Prophet sales forecasting application for the **Yo-Chi** Australian frozen yogurt chain. Produces FY26 completion and FY27 budget at venue / cluster / state / network level using Prophet (primary), SARIMA + Holt-Winters (fallbacks), and DOW-Flat + peer seasonality (new venues).

**Status (16 May 2026):** V1 release. All FY27 budgeting features working. Reference output verified at network +22% growth FY26→FY27 with realistic monthly seasonality (Jan summer peak, Jul winter trough, Apr Easter lift, Sep/Oct spring break lift).

---

## The two folders

There are **two copies** of this project on disk:

| Folder | Purpose | Use when |
|---|---|---|
| **`C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Budget FY 2027 V1\`** | **Clean V1 release** — self-contained, runs out-of-the-box with bundled data templates | You want to **run** the model or **review** the V1 deliverable |
| `C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Playwright\` | Development / iteration project — has rollback snapshots from each change | You want to see **change history**, run dev tests, or iterate further |

Both `app.py` files are byte-identical right now (last synced 16 May 2026 4:00 PM).

**To review the model**, point your new Claude project at `Budget FY 2027 V1\` — it has everything needed without the dev clutter.

---

## What's in the clean release (`Budget FY 2027 V1\`)

```
Budget FY 2027 V1/
├── app.py                           # 205.8 KB — Flask + forecast engine (~4500 lines)
├── requirements.txt                 # pip dependencies (Flask, Prophet, pandas, holidays, etc.)
├── run.bat                          # Windows launcher — auto-installs deps, opens browser
├── README.md                        # Quick start + model overview
│
├── templates/
│   ├── index.html                   # 134.2 KB — main dashboard UI
│   └── standalone_viewer.html       # offline viewer for exported forecasts
│
├── data_templates/                  # bundled real input data (1.5 MB total)
│   ├── 01_Sales_History_Template.xlsx     # daily Servings/Retail/Discounts per venue
│   ├── Average ticket history.xlsx        # monthly ATP per venue
│   ├── D03_Location_BI.xlsx               # venue → cluster / state / cannibalisation
│   ├── Venue Details.xlsx                 # new-venue templates (ramp-up, opening dates)
│   └── venue_city_mapping_updated_1.csv   # venue → city + 15-month multipliers
│
├── tests/                           # 4 verification scripts (run with server up)
│   ├── test_overlay.py              # peer seasonality overlay for sub-365 venues
│   ├── test_seasonality.py          # new-venue seasonality inheritance
│   ├── test_short_history_fix.py    # Indooroopilly/Kawana 90-day Prophet bypass
│   └── test_softening.py            # Prophet softening (β=1.0 → 0.5 comparison)
│
├── docs/
│   └── venu_cast_infographic.html   # one-page model architecture explainer
│
├── results/                         # exported forecasts land here
└── uploads/                         # uploaded files cached here
```

---

## How to run it

**Windows:** double-click `run.bat`. First run installs Python deps (~5 min). Subsequent runs start in seconds. Browser opens to <http://localhost:5000/>.

**Mac/Linux:**
```bash
cd "Budget FY 2027 V1"
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then in the dashboard:
1. **Upload** → drag the 5 files from `data_templates/`
2. **Generate Forecast** (top right) — takes ~2 min for 100+ venues
3. Explore the dashboard tabs (Sales Detail, Per-Venue, Weather & Holidays, etc.)

---

## How the model works (one paragraph)

For each venue / day, a cascading fit runs: **Prophet** (with state-specific weather + AU public holidays + AU school holidays) for venues with ≥90 trading days; **SARIMA** as fallback; **Holt-Winters** as final fallback; **DOW-Flat + peer seasonality + holiday lifts** for new venues (<90 days). Sub-365-day venues get a **peer-borrowed monthly seasonality overlay** (clamped 0.65–1.35x, 35% monthly floor) so they inherit cluster/state seasonal patterns. School holidays are split into 4 named Prophet categories (autumn / winter / spring / summer) so the model learns different coefficients per season — warm-weather breaks (Apr, Sep–Oct, Dec–Jan) lift sales 10–70% above monthly average; winter (Jul) gets a smaller bump because cold weather offsets the kids-out-of-school effect. The cascade output is then **softened** toward annualised FY26 (β=0.7 default), **cannibalisation** is netted per cluster, **baseline guardrails** cap FY27 to 0.80–1.30× annualised FY26 for mature venues (0.75–1.30× for ramping), and a **Universal LFL** target applies network-wide YoY growth with user-specified monthly weights. For **FY27 new openings** (no history), the ramp-up template generates Servings; Retail and POS Discounts are derived by applying cluster-peer ratios (e.g. Retail = 0.3% × Servings for SYD cluster).

For the full architecture diagram, open `docs/venu_cast_infographic.html` in any browser.

---

## Tunables (live, no re-generate needed)

| Control | Default | Range | Effect |
|---|---|---|---|
| Soften Prophet (β) | 0.7 (Moderate) | 0.5–1.0 | Blends Prophet output with annualised FY26 baseline |
| Universal LFL target | OFF | −10% to +20% | Network-wide YoY growth target with monthly weights |
| Cannibalisation | ON | on/off | Nets the impact of new openings on existing cluster peers |
| Guardrails | ON | on/off | Caps FY27 to band of annualised FY26 |

---

## Key API endpoints

- `GET /health` — server health + which session data is loaded
- `POST /upload` — multipart upload of all five input files
- `POST /data-sources` — parse uploaded files + pre-fetch weather/holidays
- `GET /data-sources` — current weather + public-holiday + school-holiday coverage
- `POST /generate` — run forecast for `{venues: [...]}` (omit for all)
- `GET /venues-detail?cannibalisation=on&guardrails=on` — per-venue FY26 vs FY27 with monthly arrays + holiday metadata
- `GET /sales-detail?cohort=...&sales_type=servings|retail|pos_discounts` — filterable sales table with school-holiday and public-holiday counts per month per FY
- `POST /softening/set` — `{enabled, beta}`
- `POST /universal-lfl/set` — `{enabled, target_pct, monthly_weights[12]}`

---

## Model labels — what each tag means

Every venue's `model` field tells you exactly what's in the fit:

| Tag | Meaning |
|---|---|
| `Prophet` | Facebook Prophet primary fit |
| `+Wx` | Weather regressors (temp, precip, gusts, extreme-day flag) |
| `+PubHol` | Australian public holidays per venue's state |
| `+SchHol` | School holidays per venue's state, split into 4 seasonal coefficients |
| `+PeerSeas` | Peer-borrowed monthly seasonality overlay (venues <365 trading days) |
| `(clamped Xd)` | Sanity ceiling clamped X days during forecast horizon |
| `DOW-Flat` | Day-of-week flat fallback (new venues <90 days) |
| `FutureVenue-Template` | FY27 New Opening — uses Venue Details template |
| `Servings×N.NNN (peer-derived)` | Retail or Discounts stream derived from cluster peer ratios (new venues only) |

Example: `Prophet+Wx+PubHol+SchHol+PeerSeas (clamped 35d)` means Noosa-style mature beach venue with weather, holidays, peer overlay, and 35 ceiling clamps.

---

## V1 release decisions worth knowing

1. **School holiday data range: 2024–2028.** Earlier years not loaded (sales history only goes back ~2 years, so 2020–2023 school holidays were unused weight). Sources: state Department of Education websites. NSW uses Eastern division. Year-on-year date shifts ARE captured (e.g. Easter pulling VIC autumn break into March 2027).

2. **School holiday lift fallback: +18%, with +5% warmth bonus for Apr/Sep/Oct/Dec/Jan breaks** (so +23.9% effective on warm-weather school days, +18% on July). Public holiday lift: +20%. These fallbacks apply ONLY to new venues using DOW-Flat — Prophet learns the actual per-season coefficient from history for mature venues.

3. **Sub-365 venue seasonality** uses *monthly rescale* (not daily multiply): Prophet's annual total preserved, monthly totals scaled to peer seasonal shape, daily texture (DOW/weather/holiday) preserved. Clamped 0.65–1.35x with 35% monthly floor (no $0 months).

4. **Prophet tiers** by trading days: <30 → DOW-Flat-only; 30–90 → very flat trend; 90–179 → near-flat trend (changepoint=0.001); 180–364 → flat trend + peer overlay (changepoint=0.003); 365+ → full Prophet with yearly seasonality + 12 Fourier terms.

5. **FY27 New Openings** now have realistic Retail and POS Discount forecasts derived from cluster-peer ratios (previously $0).

---

## Running the verification tests

With the Flask app running, in another terminal:

```bash
python tests/test_overlay.py            # sub-365 venues — should show Jan peak, Jun trough, 35-55% variation
python tests/test_seasonality.py        # new venues inherit peer seasonal shape
python tests/test_short_history_fix.py  # Indooroopilly/Kawana don't crash to $0
python tests/test_softening.py          # softening β=1.0 vs 0.5 comparison
```

Healthy output: all sub-365 venues peak Jan, trough Jun, 35–55% intra-year variation, no $0 months.

---

## Reference output (V1, full network, 16 May 2026)

| Metric | Value |
|---|---|
| Venues forecast | 85 (52 mature + 19 FY26-new + 14 FY27-new) |
| Network FY26 | $226.05M |
| Network FY27 | $276.17M |
| Network growth | **+22.2%** |
| Peak month | **Jan ($28.25M)** |
| Trough month | **Aug ($20.15M)** |
| Apr (Easter) | $25.06M |
| Beach venues (Noosa/Manly/Bondi) variation | 80–107% (clamped) |
| Sub-365 venues variation | 35–55% (peer overlay) |
| Mature venues variation | 18–46% (organic Prophet) |

---

## If you need to iterate

The dev project is at `C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Playwright\`. It has:
- Numbered rollback snapshots (`rollback_20260516_*\`) for every major change
- Additional tests from earlier iterations (`test_cannibalisation*.py`, `test_universal_lfl*.py`, etc.)
- The same `app.py` (byte-identical) and `templates/`

Make changes in the dev project, then `Copy-Item` to `Budget FY 2027 V1\` to publish.

---

*Generated 16 May 2026.*
