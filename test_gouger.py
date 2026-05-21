"""Retest Gouger St and similar mature-with-step-change venues."""
import os, time, requests
from collections import defaultdict
from datetime import datetime

DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'

for _ in range(15):
    try:
        if requests.get(f'{BASE}/health', timeout=3).status_code == 200:
            break
    except Exception:
        time.sleep(2)

print("[1] Upload + pre-fetch...")
files = {
    'sales_history': open(os.path.join(DATA_DIR, '01_Sales_History_Template.xlsx'), 'rb'),
    'atp_history': open(os.path.join(DATA_DIR, 'Average ticket history.xlsx'), 'rb'),
    'location_bi': open(os.path.join(DATA_DIR, 'D03_Location_BI.xlsx'), 'rb'),
    'venue_details': open(os.path.join(DATA_DIR, 'Venue Details.xlsx'), 'rb'),
    'multipliers': open(os.path.join(DATA_DIR, 'venue_city_mapping_updated_1.csv'), 'rb'),
}
requests.post(f'{BASE}/upload', files=files, timeout=120)
for fh in files.values():
    fh.close()
requests.post(f'{BASE}/data-sources', timeout=300)

# Test a mix
venues = ['Yo-Chi Gouger St', 'Yo-Chi Glenelg', 'Yo-Chi Bondi', 'Yo-Chi Carlton',
          'Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Burwood', 'Yo-Chi Castle Towers',
          'Yo-Chi Cronulla', 'Yo-Chi Eastland', 'Yo-Chi Barangaroo']

print(f"\n[2] Forecast for {len(venues)} venues...")
r = requests.post(f'{BASE}/generate', json={'venues': venues}, timeout=900)
result = r.json()
print(f"    {result['summary']['successful']}/{result['summary']['total_venues']} successful")

print(f"\n[3] Recent actuals vs forecast (May-Aug 2026)")
print(f"    {'Venue':<25} {'Apr Actual':>11} {'May FC':>10} {'Jun FC':>10} {'Jul FC':>10} {'Aug FC':>10} {'May/Apr':>9} {'Model':<35}")
servings_actual = {}
# Fetch CSV to get actuals
r = requests.get(f'{BASE}/download-csv', timeout=120)
import csv, io
rows = list(csv.DictReader(io.StringIO(r.text)))
apr_actuals = defaultdict(float)
for row in rows:
    if row['type'] == 'actual' and row['date'] >= '2026-04-01' and row['date'] <= '2026-04-30' and row['servings_sales']:
        apr_actuals[row['venue']] += float(row['servings_sales'])

for r_v in result['results']:
    if 'servings' not in r_v:
        continue
    v = r_v['venue']
    s = r_v['servings']
    monthly = defaultdict(float)
    for date_str, val in zip(r_v['forecast_dates'], s['sales']):
        d = datetime.strptime(date_str, "%Y-%m-%d")
        if d.year == 2026:
            monthly[d.month] += val
    apr_act = apr_actuals.get(v, 0)
    may_fc = monthly[5]
    ratio = may_fc / apr_act if apr_act > 0 else 0
    flag = "[OK]" if 0.7 <= ratio <= 1.3 else ("[WARN]" if 0.5 <= ratio <= 1.5 else "[FAIL]")
    print(f"    {flag} {v:<22} {apr_act:>10,.0f}  {may_fc:>10,.0f} {monthly[6]:>10,.0f} {monthly[7]:>10,.0f} {monthly[8]:>10,.0f} {ratio:>8.2f}x  {s['model']:<30}")

print("\n[4] Year totals comparison:")
for r_v in result['results']:
    if 'servings' not in r_v:
        continue
    v = r_v['venue']
    sales = r_v['servings']['sales']
    year_total = sum(sales[:365])
    # Compute actual year total (May 2025 - Apr 2026)
    actual_year = sum(float(row['servings_sales']) for row in rows
                      if row['venue'] == v and row['type'] == 'actual'
                      and '2025-05-01' <= row['date'] <= '2026-04-30'
                      and row['servings_sales'])
    delta = year_total - actual_year
    pct = (delta / actual_year * 100) if actual_year > 0 else 0
    flag = "[OK]" if abs(pct) < 15 else ("[WARN]" if abs(pct) < 30 else "[FAIL]")
    print(f"    {flag} {v:<25} FY26 actual ${actual_year:>11,.0f}  FY27 forecast ${year_total:>11,.0f}  delta {pct:+6.1f}%")
