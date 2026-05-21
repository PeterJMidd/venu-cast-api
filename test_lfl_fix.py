"""Verify Total Growth == LFL Growth when only FY26 Existing cohort is selected."""
import os, time, requests
DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'
for _ in range(15):
    try:
        if requests.get(f'{BASE}/health', timeout=3).status_code == 200:
            break
    except Exception:
        time.sleep(2)

print("[1] Upload + pre-fetch + generate forecast for mature venues only...")
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

# A mix that includes both FY26 Existing and FY26/FY27 new openings
venues = ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Carlton', 'Yo-Chi Gouger St',
          'Yo-Chi Burwood', 'Yo-Chi Castle Towers', 'Yo-Chi New Venue 01']
requests.post(f'{BASE}/generate', json={'venues': venues}, timeout=900)

print("\n[2] All venues (no filter):")
r = requests.get(f'{BASE}/sales-detail', timeout=30)
s = r.json()['summary']
print(f"    FY26: ${s['fy26_comparator']:>14,.0f}")
print(f"    FY27: ${s['fy27_forecast']:>14,.0f}")
print(f"    Total Growth: {s['total_growth_pct']:.1f}%   LFL Growth: {s['lfl_growth_pct']:.1f}%")
print(f"    Venues: {s['venues_in_filter']} (FY26 Existing: {s['venues_fy26_existing_in_filter']})")

print("\n[3] Cohort=FY26 Existing — Total Growth and LFL Growth should now MATCH:")
r = requests.get(f'{BASE}/sales-detail?cohort=FY26%20Existing', timeout=30)
s = r.json()['summary']
print(f"    FY26: ${s['fy26_comparator']:>14,.0f}   LFL FY26: ${s['lfl_fy26']:>14,.0f}")
print(f"    FY27: ${s['fy27_forecast']:>14,.0f}   LFL FY27: ${s['lfl_fy27']:>14,.0f}")
print(f"    Total Growth: {s['total_growth_pct']:.1f}%   LFL Growth: {s['lfl_growth_pct']:.1f}%")
print(f"    Venues: {s['venues_in_filter']}")
if abs(s['total_growth_pct'] - s['lfl_growth_pct']) < 0.01:
    print("    [OK] Total Growth == LFL Growth (as expected)")
else:
    print(f"    [WARN] Total Growth != LFL Growth (delta {abs(s['total_growth_pct'] - s['lfl_growth_pct']):.2f}%)")

print("\n[4] Monthly LFL fields:")
m = r.json()['monthly']
print(f"    {'Month':<5}  {'FY26':>13} {'FY27':>13} {'LFL FY26':>13} {'LFL FY27':>13} {'Growth%':>10} {'LFL Growth%':>13}")
for mo in m:
    g = f"{mo['growth_pct']:.1f}%" if mo['growth_pct'] is not None else '—'
    lg = f"{mo['lfl_growth_pct']:.1f}%" if mo['lfl_growth_pct'] is not None else '—'
    print(f"    {mo['month_label']:<5}  {mo['fy26_comparator']:>13,.0f} {mo['fy27_forecast']:>13,.0f} {mo['lfl_fy26']:>13,.0f} {mo['lfl_fy27']:>13,.0f} {g:>10} {lg:>13}")

print("\nDONE")
