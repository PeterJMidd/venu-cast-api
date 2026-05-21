"""Re-test the venues that previously had runaway forecasts (Burwood, Castle Towers, Cronulla, Eastland)
   plus mature venues to make sure they still forecast correctly.
"""
import os, time, csv, io
from collections import defaultdict
from datetime import datetime
import requests

DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'

# Wait for server
for _ in range(15):
    try:
        r = requests.get(f'{BASE}/health', timeout=3)
        if r.status_code == 200:
            break
    except Exception:
        time.sleep(2)

print("=" * 78)
print("PROBLEM-VENUE RE-TEST after trim_history + n_trading routing + ramp ceiling")
print("=" * 78)

print("\n[1] Upload templates...")
files = {
    'sales_history': open(os.path.join(DATA_DIR, '01_Sales_History_Template.xlsx'), 'rb'),
    'atp_history': open(os.path.join(DATA_DIR, 'Average ticket history.xlsx'), 'rb'),
    'location_bi': open(os.path.join(DATA_DIR, 'D03_Location_BI.xlsx'), 'rb'),
    'venue_details': open(os.path.join(DATA_DIR, 'Venue Details.xlsx'), 'rb'),
    'multipliers': open(os.path.join(DATA_DIR, 'venue_city_mapping_updated_1.csv'), 'rb'),
}
r = requests.post(f'{BASE}/upload', files=files, timeout=120)
for fh in files.values():
    fh.close()
print(f"    Upload: {r.status_code}")

print("\n[2] Pre-fetch weather + holidays...")
r = requests.post(f'{BASE}/data-sources', timeout=300)
print(f"    Done: {r.json()['states_loaded']}")

# Test mix: 4 previously-wild venues + 2 mature + 1 future
test_venues = [
    'Yo-Chi Burwood',         # opens 2025-10-25 - 6 months history
    'Yo-Chi Castle Towers',   # opens 2025-11-01 - 6 months
    'Yo-Chi Cronulla',        # opens 2025-07-07 - 10 months
    'Yo-Chi Eastland',        # opens 2025-08-11 - 9 months
    'Yo-Chi Albert St',       # mature (5 years)
    'Yo-Chi Balaclava',       # mature (5 years)
    'Yo-Chi New Venue 01',    # future template
]

print(f"\n[3] Forecast for {len(test_venues)} venues...")
r = requests.post(f'{BASE}/generate', json={'venues': test_venues}, timeout=600)
print(f"    Status: {r.status_code}")
result = r.json()
print(f"    {result['summary']['successful']}/{result['summary']['total_venues']} successful")
print(f"    Models: {result['summary']['models_used']}")

print("\n[4] Monthly $ totals per venue (forecast):")
print(f"    {'Venue':<25} {'Model':<35} {'May':>10} {'Jun':>10} {'Jul':>10} {'Aug':>10} {'Sep':>10} {'Oct':>10} {'Nov':>10} {'Dec':>10}")

for r_v in result['results']:
    if 'servings' not in r_v:
        continue
    v = r_v['venue']
    s = r_v['servings']
    sales = s['sales']
    monthly = defaultdict(float)
    for date_str, val in zip(r_v['forecast_dates'], sales):
        d = datetime.strptime(date_str, "%Y-%m-%d")
        if d.year == 2026:
            monthly[d.month] += val
    line = f"    {v:<25} {s['model']:<35}"
    for m in [5, 6, 7, 8, 9, 10, 11, 12]:
        line += f" {monthly[m]:>10,.0f}"
    print(line)

print("\n[5] Per-venue analysis: max month vs avg month (should be < ~1.5x for sensible forecast):")
for r_v in result['results']:
    if 'servings' not in r_v:
        continue
    v = r_v['venue']
    sales = r_v['servings']['sales']
    monthly = defaultdict(float)
    for date_str, val in zip(r_v['forecast_dates'], sales):
        d = datetime.strptime(date_str, "%Y-%m-%d")
        if d.year == 2026 and d.month >= 5:
            monthly[d.month] += val
    if not monthly:
        continue
    vals = list(monthly.values())
    mx = max(vals)
    mn = min(v for v in vals if v > 0) if any(v > 0 for v in vals) else 0
    avg = sum(vals) / len(vals)
    ratio = mx / avg if avg > 0 else 0
    flag = "[OK]" if ratio < 1.6 else ("[WARN]" if ratio < 2.5 else "[FAIL]")
    print(f"    {flag} {v:<25} min ${mn:>9,.0f} avg ${avg:>9,.0f} max ${mx:>9,.0f} max/avg={ratio:.2f}x")

print("\n[6] Specifically: Burwood/Castle Towers/Cronulla/Eastland (the bad ones)")
for v_target in ['Yo-Chi Burwood', 'Yo-Chi Castle Towers', 'Yo-Chi Cronulla', 'Yo-Chi Eastland']:
    for r_v in result['results']:
        if r_v.get('venue') != v_target or 'servings' not in r_v:
            continue
        sales = r_v['servings']['sales']
        # Find Nov 2026 forecast
        nov_total = sum(v for d, v in zip(r_v['forecast_dates'], sales) if d.startswith('2026-11'))
        dec_total = sum(v for d, v in zip(r_v['forecast_dates'], sales) if d.startswith('2026-12'))
        first_year_total = sum(sales[:365])
        print(f"    {v_target}: model={r_v['servings']['model']}")
        print(f"        Nov 2026 total: ${nov_total:,.0f}, Dec 2026 total: ${dec_total:,.0f}")
        print(f"        Year-1 total May 26 - Apr 27: ${first_year_total:,.0f}")
        break

print("\n" + "=" * 78)
print("DONE")
print("=" * 78)
