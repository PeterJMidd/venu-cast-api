"""Verify weather + state-based holidays are loaded and influence the forecast."""
import os
import requests
import json

DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'

print("=" * 70)
print("WEATHER + STATE HOLIDAY INTEGRATION TEST")
print("=" * 70)

# 1) Upload templates
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

# 2) Pre-fetch weather + holidays
print("\n[2] Pre-fetching weather + holidays for all venue states...")
r = requests.post(f'{BASE}/data-sources', timeout=300)
print(f"    Status: {r.status_code}")
data = r.json()
print(f"    States loaded: {data['states_loaded']}")
print(f"    Weather cache keys: {len(data['weather_cache_keys'])}")
print(f"    Holiday cache keys: {len(data['holiday_cache_keys'])}")

# 3) Inspect data sources
print("\n[3] Inspecting loaded weather + holidays...")
r = requests.get(f'{BASE}/data-sources', timeout=15)
ds = r.json()
print("    Weather coverage:")
for state, info in list(ds['weather'].items())[:4]:
    print(f"      {state}: {info['days']} days, {info['min_date']} to {info['max_date']}, range {info['min_temp']}-{info['max_temp']}C (avg {info['avg_temp']}C)")
print("    Holiday coverage:")
for state, info in list(ds['holidays'].items())[:4]:
    print(f"      {state}: {info['count']} holidays")
    for date, name in info['examples'][:3]:
        print(f"        {date}: {name}")

# 4) Run forecast for 3 venues across different states
print("\n[4] Running forecast for 3 venues in different states...")
# Need to pick venues with sales AND multipliers AND a state
r = requests.get(f'{BASE}/venues', timeout=15)
vlist = r.json()['venues']
test_venues = []
states_seen = set()
for v in vlist:
    s = v.get('state') or ''
    if s in ('VIC', 'NSW', 'QLD') and s not in states_seen and v.get('multipliers'):
        test_venues.append(v['venue'])
        states_seen.add(s)
    if len(test_venues) == 3:
        break
print(f"    Test venues: {test_venues}")

r = requests.post(f'{BASE}/generate', json={'venues': test_venues}, timeout=600)
print(f"    Status: {r.status_code}")
result = r.json()
summary = result['summary']
print(f"    Successful: {summary['successful']}/{summary['total_venues']}")
print(f"    Models used: {summary['models_used']}")
print(f"    States processed: {summary['states_processed']}")

# 5) Verify per-venue weather + holiday data
print("\n[5] Verifying per-venue weather + holiday metadata...")
for r_v in result['results']:
    if 'servings' not in r_v:
        continue
    v = r_v['venue']
    state = r_v.get('state', '?')
    wx = r_v.get('weather_temp_max', [])
    hols = r_v.get('holiday_names', [])
    wx_vals = [t for t in wx if t is not None]
    hol_days = [(d, n) for d, n in zip(r_v['forecast_dates'], hols) if n]
    print(f"\n    {v} (state={state}, model={r_v['servings']['model']}):")
    print(f"      Weather days in forecast window: {len(wx_vals)}/{len(wx)}")
    if wx_vals:
        print(f"      Forecast temp range: {min(wx_vals):.1f}C to {max(wx_vals):.1f}C (avg {sum(wx_vals)/len(wx_vals):.1f}C)")
    print(f"      Holidays in forecast window: {len(hol_days)}")
    for d, n in hol_days[:5]:
        print(f"        {d}: {n}")

# 6) Compare model name — should include Wx and Hol
print("\n[6] Verifying model uses both weather + holidays...")
for r_v in result['results']:
    if 'servings' in r_v:
        mdl = r_v['servings']['model']
        has_wx = 'Wx' in mdl
        has_hol = 'Hol' in mdl
        print(f"    {r_v['venue']}: {mdl} to wx={has_wx}, holidays={has_hol}")

# 7) CSV export must include weather + holiday columns
print("\n[7] Verifying CSV export includes weather + holidays...")
r = requests.get(f'{BASE}/download-csv', timeout=30)
text = r.text
header = text.split('\n', 1)[0]
print(f"    CSV header: {header}")
has_wx = 'weather_temp_max' in header
has_hol = 'public_holiday' in header
print(f"    Has weather column: {has_wx}")
print(f"    Has public_holiday column: {has_hol}")

# Find a holiday row
print("\n    Sample rows with public holidays:")
lines = text.split('\n')
holiday_lines = [l for l in lines if l and ',' in l and 'public_holiday' not in l]
holiday_rows = [l for l in holiday_lines[:5000] if l.strip() and l.split(',')[-2].strip() and not l.split(',')[-2].strip() == '']
shown = 0
for l in holiday_rows:
    if shown >= 5:
        break
    print(f"      {l[:160]}")
    shown += 1

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
