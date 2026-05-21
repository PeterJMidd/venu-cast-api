"""Verify Prophet base + multiplier lift = total growth."""
import os, time, requests
DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'
for _ in range(15):
    try:
        if requests.get(f'{BASE}/health', timeout=3).status_code == 200:
            break
    except Exception:
        time.sleep(2)

# To demonstrate the multiplier effect, temporarily inflate multipliers to 1.04
# Use a sample CSV with adjusted multipliers
print("[1] Upload + pre-fetch...")
files = {
    'sales_history': open(os.path.join(DATA_DIR, '01_Sales_History_Template.xlsx'), 'rb'),
    'atp_history': open(os.path.join(DATA_DIR, 'Average ticket history.xlsx'), 'rb'),
    'location_bi': open(os.path.join(DATA_DIR, 'D03_Location_BI.xlsx'), 'rb'),
    'venue_details': open(os.path.join(DATA_DIR, 'Venue Details.xlsx'), 'rb'),
    'multipliers': open(os.path.join(DATA_DIR, 'venue_city_mapping_updated_1.csv'), 'rb'),
}
requests.post(f'{BASE}/upload', files=files, timeout=120)
for fh in files.values(): fh.close()
requests.post(f'{BASE}/data-sources', timeout=300)

# Run a small forecast
print("[2] Forecast 4 venues...")
venues = ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Barangaroo', 'Yo-Chi Carlton']
requests.post(f'{BASE}/generate', json={'venues': venues}, timeout=900)

print("\n[3] Sales detail with growth attribution:")
r = requests.get(f'{BASE}/sales-detail', timeout=30)
s = r.json()['summary']

print(f"\n    Total Growth:        {s['total_growth_pct']:>7.1f}%")
print(f"    Prophet base growth: {s['base_growth_pct']:>7.1f}%   (forecast WITHOUT multiplier)")
print(f"    Multiplier lift:     {s['multiplier_lift_pct']:>7.1f}%   (extra uplift from 15-month multipliers)")
print(f"    Reconciliation: base + lift = {s['base_growth_pct'] + s['multiplier_lift_pct']:.1f}%  vs  total {s['total_growth_pct']:.1f}%")

print(f"\n    LFL Total Growth:      {s['lfl_growth_pct']:>7.1f}%")
print(f"    LFL Prophet base:      {s['lfl_base_growth_pct']:>7.1f}%")
print(f"    LFL Multiplier lift:   {s['lfl_multiplier_lift_pct']:>7.1f}%")
print(f"    Reconciliation: {s['lfl_base_growth_pct'] + s['lfl_multiplier_lift_pct']:.1f}% vs {s['lfl_growth_pct']:.1f}%")

print(f"\n    FY27 Forecast (with mults):    ${s['fy27_forecast']:>14,.0f}")
print(f"    FY27 Forecast (Prophet base):  ${s['fy27_base_forecast']:>14,.0f}")
print(f"    Multiplier net $ contribution: ${s['fy27_forecast'] - s['fy27_base_forecast']:>14,.0f}")

print("\n[4] Monthly breakdown:")
monthly = r.json()['monthly']
print(f"    {'Month':<5} {'FY26':>11} {'FY27':>11} {'FY27 base':>11} {'Total %':>8} {'Base %':>8} {'Lift':>7}")
for m in monthly[:6]:
    g = f"{m['growth_pct']:.1f}%" if m['growth_pct'] is not None else '-'
    bg = f"{m['base_growth_pct']:.1f}%" if m['base_growth_pct'] is not None else '-'
    lift = f"{m['multiplier_lift_pct']:.1f}" if m['multiplier_lift_pct'] is not None else '-'
    print(f"    {m['month_label']:<5} {m['fy26_comparator']:>11,.0f} {m['fy27_forecast']:>11,.0f} {m['fy27_base']:>11,.0f} {g:>8} {bg:>8} {lift:>7}")

print("\nDONE")
