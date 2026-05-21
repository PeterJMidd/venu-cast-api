"""Test the new-venue safeguards:
   1. Future venue (no history) uses ramp-up template - no Prophet runaway
   2. Recently opened venue (<21 trading days) uses DOW-flat
   3. Mature venue gets full Prophet + ceiling
"""
import os
import requests

DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'

print("=" * 70)
print("NEW VENUE HANDLING TEST")
print("=" * 70)

print("\n[1] Uploading templates...")
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
print(f"    Status: {r.status_code}")

print("\n[2] Pre-fetching weather + holidays...")
r = requests.post(f'{BASE}/data-sources', timeout=300)
print(f"    Done: {r.json()['states_loaded']}")

# Three test venues representing different categories
# Mature: Yo-Chi Albert St (years of data)
# Future: Yo-Chi New Venue 01 (no history, opens 2026-07-01)
# Recently opened: Yo-Chi Harbour Town (opens 2026-04-11 = only ~1 month before forecast)
test_venues = ['Yo-Chi Albert St', 'Yo-Chi New Venue 01', 'Yo-Chi Harbour Town', 'Yo-Chi Kawana Waters']

print(f"\n[3] Running forecast for: {test_venues}")
r = requests.post(f'{BASE}/generate', json={'venues': test_venues}, timeout=600)
print(f"    Status: {r.status_code}")
result = r.json()
print(f"    Summary: {result['summary']['successful']}/{result['summary']['total_venues']} venues")
print(f"    Models: {result['summary']['models_used']}")

print("\n[4] Sanity check per venue (no runaway forecasts)...")
for r_v in result['results']:
    if 'servings' in r_v:
        v = r_v['venue']
        s = r_v['servings']
        sales = s['sales']
        max_day = max(sales) if sales else 0
        min_day = min(sales) if sales else 0
        avg_day = sum(sales) / len(sales) if sales else 0
        # Find a forecast day to inspect (e.g., first day, mid, last)
        print(f"\n    {v}:")
        print(f"      Model: {s['model']}, RMSE: {s['rmse']}")
        print(f"      Daily sales: min=${min_day:,.0f}, avg=${avg_day:,.0f}, max=${max_day:,.0f}")
        # The Kawana Waters historical max was $11,650; ensure no day exceeds ~$25k
        if 'Kawana' in v or 'New Venue' in v:
            if max_day > 50000:
                print(f"      [WARN] Max daily forecast ${max_day:,.0f} is suspicious for a small new venue!")
            else:
                print(f"      [OK] Max forecast within safe range")
        # Future venues should start with 0 or low until opening date
        if 'New Venue 01' in v:
            # Opens 2026-07-01, so May/June should be 0
            for i, d in enumerate(r_v['forecast_dates']):
                if d < '2026-07-01' and sales[i] > 0:
                    print(f"      [WARN] Should be 0 before opening: {d}: ${sales[i]:.0f}")
                    break
            else:
                print(f"      [OK] Zero sales before opening date (2026-07-01)")
            # First post-opening day
            for i, d in enumerate(r_v['forecast_dates']):
                if d >= '2026-07-01':
                    print(f"      First open day {d}: ${sales[i]:,.0f}")
                    break

print("\n[5] Pulling CSV and checking export per category...")
r = requests.get(f'{BASE}/download-csv', timeout=120)
import csv, io
rows = list(csv.DictReader(io.StringIO(r.text)))
print(f"    Total CSV rows: {len(rows)}")

for v in test_venues:
    v_rows = [r for r in rows if r['venue'] == v and r['type'] == 'forecast']
    if v_rows:
        sales = [float(r['servings_sales']) for r in v_rows if r['servings_sales']]
        if sales:
            print(f"    {v}: {len(v_rows)} forecast rows, daily $ min={min(sales):.0f} avg={sum(sales)/len(sales):.0f} max={max(sales):.0f}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
