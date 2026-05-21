"""Test the /sales-detail endpoint with filters."""
import os, time, requests
from collections import defaultdict

DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'

for _ in range(15):
    try:
        if requests.get(f'{BASE}/health', timeout=3).status_code == 200:
            break
    except Exception:
        time.sleep(2)

print("[1] Upload + pre-fetch + generate full forecast for 10 venues...")
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

# Mix of mature + new-opened + future venues to populate cohorts
venues = ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Carlton', 'Yo-Chi Gouger St',
          'Yo-Chi Burwood', 'Yo-Chi Castle Towers', 'Yo-Chi Cronulla',
          'Yo-Chi New Venue 01', 'Yo-Chi New Venue 02', 'Yo-Chi Eastland']
r = requests.post(f'{BASE}/generate', json={'venues': venues}, timeout=900)
print(f"    Status: {r.status_code}")

print("\n[2] Test /sales-detail with no filters (all venues)...")
r = requests.get(f'{BASE}/sales-detail', timeout=30)
data = r.json()
print(f"    Status: {r.status_code}")

if 'summary' in data:
    s = data['summary']
    print(f"\n    KPIs:")
    print(f"      FY27 Forecast Sales:      ${s['fy27_forecast']:>14,.0f}")
    print(f"      FY26 Comparator:          ${s['fy26_comparator']:>14,.0f}")
    print(f"      FY26 Completion FC:       ${s['fy26_completion_forecast']:>14,.0f}")
    print(f"      Total Growth:             {s['total_growth_pct']:>14.1f}%")
    print(f"      LFL Growth:               {s['lfl_growth_pct']:>14.1f}%")
    print(f"      Venues in filter:         {s['venues_in_filter']} (FY26 existing: {s['venues_fy26_existing_in_filter']})")

    print(f"\n    Filters available:")
    for k, v in data['filters_available'].items():
        print(f"      {k}: {v[:5]}{' ...' if len(v) > 5 else ''}")

    print(f"\n    Monthly (FY order Jul-Jun):")
    print(f"    {'Month':<6} {'FY26 Comparator':>18} {'FY27 Forecast':>18} {'LFL FY27':>18}")
    for m in data['monthly']:
        print(f"    {m['month_label']:<6} {m['fy26_comparator']:>18,.0f} {m['fy27_forecast']:>18,.0f} {m['lfl_fy27']:>18,.0f}")

print("\n[3] Test /sales-detail with state=VIC filter...")
r = requests.get(f'{BASE}/sales-detail?state=VIC', timeout=30)
data = r.json()
if 'summary' in data:
    s = data['summary']
    print(f"    VIC FY27: ${s['fy27_forecast']:,.0f}, FY26: ${s['fy26_comparator']:,.0f}")
    print(f"    Venues: {s['venues_in_filter']}, Growth: {s['total_growth_pct']:.1f}%")

print("\n[4] Test /sales-detail with cohort='FY26 Existing' filter...")
r = requests.get(f'{BASE}/sales-detail?cohort=FY26%20Existing', timeout=30)
data = r.json()
if 'summary' in data:
    s = data['summary']
    print(f"    Cohort 'FY26 Existing' FY27: ${s['fy27_forecast']:,.0f}, FY26: ${s['fy26_comparator']:,.0f}")
    print(f"    Venues: {s['venues_in_filter']}, Growth: {s['total_growth_pct']:.1f}%")

print("\n[5] Test /sales-detail with cohort='FY27 New Opening' filter...")
r = requests.get(f'{BASE}/sales-detail?cohort=FY27%20New%20Opening', timeout=30)
data = r.json()
if 'summary' in data:
    s = data['summary']
    print(f"    Cohort 'FY27 New Opening' FY27: ${s['fy27_forecast']:,.0f}, FY26: ${s['fy26_comparator']:,.0f}")
    print(f"    Venues: {s['venues_in_filter']} (should be just the New Venue ones)")

print("\n[6] Single-venue filter...")
r = requests.get(f'{BASE}/sales-detail?venue=Yo-Chi%20Albert%20St', timeout=30)
data = r.json()
if 'summary' in data:
    s = data['summary']
    print(f"    Yo-Chi Albert St FY27: ${s['fy27_forecast']:,.0f}, FY26: ${s['fy26_comparator']:,.0f}")
    print(f"    Growth: {s['total_growth_pct']:.1f}%")

print("\nDONE")
