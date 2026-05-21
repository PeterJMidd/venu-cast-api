"""End-to-end test: upload all 5 templates, generate forecast, verify outputs."""
import os
import requests
import json

DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'

print("=" * 70)
print("END-TO-END TEST: Real Templates -> Forecast -> Output")
print("=" * 70)

# 1) Health check
print("\n[1] Health check...")
r = requests.get(f'{BASE}/health', timeout=10)
print(f"    Status: {r.status_code}")
print(f"    Libs: {r.json()['libs']}")

# 2) Upload all templates
print("\n[2] Uploading all 5 templates...")
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
data = r.json()
print(f"    Upload summary:")
for k, v in data.get('summary', {}).items():
    print(f"      {k}: {v}")
if data.get('errors'):
    print(f"    [ERRORS]: {data['errors']}")

# 3) Get venues list
print("\n[3] Fetching venues list...")
r = requests.get(f'{BASE}/venues', timeout=15)
vdata = r.json()
print(f"    Total venues: {vdata['count']}")
print(f"    First 3 venues:")
for v in vdata['venues'][:3]:
    print(f"      - {v['venue']} | cluster: {v.get('cluster', 'n/a')} | state: {v.get('state', 'n/a')}")

# 4) ATP growth template
print("\n[4] Checking ATP growth template...")
r = requests.get(f'{BASE}/atp-growth', timeout=10)
g = r.json()
sample_venue = next((k for k in g['template'].keys() if k != '_default'), None)
print(f"    Sample venue: {sample_venue}")
if sample_venue:
    print(f"    Sample growth: {g['template'][sample_venue]}")

# 5) Generate forecast for 3 venues (pick venues that have sales history AND multipliers)
print("\n[5] Generating forecast for 3 venues (May 2026 - June 2027)...")
# Pick first 3 venues starting with "Yo-Chi" that have multipliers loaded
yo_chi_venues = [v['venue'] for v in vdata['venues'] if v['venue'].startswith('Yo-Chi') and v.get('multipliers')]
test_venues = yo_chi_venues[:3]
print(f"    Test venues: {test_venues}")

payload = {
    'venues': test_venues,
    'holiday_dates': ['2026-06-09', '2026-12-25', '2026-12-26', '2027-01-01', '2027-01-26']
}
r = requests.post(f'{BASE}/generate', json=payload, timeout=600)
print(f"    Status: {r.status_code}")
if r.status_code != 200:
    print(f"    ERROR: {r.text[:500]}")
else:
    result = r.json()
    summary = result['summary']
    print(f"    Successful: {summary['successful']}/{summary['total_venues']}")
    print(f"    Forecast period: {summary['forecast_period']} ({summary['forecast_days']} days)")
    print(f"    Models used: {summary['models_used']}")

    # Sample first venue's output
    first = result['results'][0]
    if 'servings' in first:
        s = first['servings']
        print(f"\n    First venue: {first['venue']}")
        print(f"      Model: {s['model']}, RMSE: {s['rmse']}")
        print(f"      Sample day 0 ({first['forecast_dates'][0]}):")
        print(f"        Servings: ${s['sales'][0]:.2f}, multiplier: {s['multipliers'][0]:.4f}")
        print(f"        ATP: ${s['atp_forecast'][0]:.2f}, Transactions: {s['transactions'][0]:.1f}")
        print(f"      Sample day 200 ({first['forecast_dates'][200]}):")
        print(f"        Servings: ${s['sales'][200]:.2f}, multiplier: {s['multipliers'][200]:.4f}")
        print(f"        ATP: ${s['atp_forecast'][200]:.2f}, Transactions: {s['transactions'][200]:.1f}")
        if 'retail' in first:
            print(f"      Retail day 0: ${first['retail']['sales'][0]:.2f}")
        if 'discounts' in first:
            print(f"      Discounts day 0: ${first['discounts']['sales'][0]:.2f}")

        # Verify multiplier applied
        unique_mults = len(set([round(m, 4) for m in s['multipliers']]))
        print(f"      Unique multipliers: {unique_mults} (should be <=15 if multipliers vary)")

        # Verify transactions = sales/atp
        d0_sales = s['sales'][0]
        d0_atp = s['atp_forecast'][0]
        d0_trans = s['transactions'][0]
        calc_trans = d0_sales / d0_atp if d0_atp > 0 else 0
        err = abs(calc_trans - d0_trans) / max(calc_trans, d0_trans, 1)
        print(f"      Transaction reconciliation: {d0_sales:.2f}/{d0_atp:.2f}={calc_trans:.2f} vs reported {d0_trans:.2f} (err={err:.4%})")

# 6) Test CSV download
print("\n[6] Testing CSV download...")
r = requests.get(f'{BASE}/download-csv', timeout=30)
print(f"    Status: {r.status_code}")
print(f"    Size: {len(r.content):,} bytes")
print(f"    First 250 chars:")
print(f"    {r.text[:250]}")

print("\n" + "=" * 70)
print("E2E TEST COMPLETE")
print("=" * 70)
