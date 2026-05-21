"""Verify guardrail changes catch the extreme venues.
   Note: seasonality prior changes (Prophet) require a re-forecast, so this test
   verifies only the immediate guardrail changes (which take effect on next view request).
"""
import os, time, requests
DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'

for _ in range(15):
    try:
        if requests.get(f'{BASE}/health', timeout=3).status_code == 200:
            break
    except Exception:
        time.sleep(2)

print("[1] Upload + forecast the problematic NSW + QLD venues...")
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
problem_venues = ['Yo-Chi Cronulla', 'Yo-Chi Circular Quay', 'Yo-Chi Surry Hills',
                  'Yo-Chi Castle Towers', 'Yo-Chi Lane Cove', 'Yo-Chi Macquarie',
                  'Yo-Chi Rouse Hill', 'Yo-Chi Southbank']
requests.post(f'{BASE}/generate', json={'venues': problem_venues}, timeout=900)

print("\n[2] Compare guardrails OFF vs ON (with new TIGHTER bands):")
def fetch(g):
    r = requests.get(f'{BASE}/venues-detail?guardrails={g}', timeout=20)
    return {v['venue']: v for v in r.json()['venues']}

off = fetch('off')
on = fetch('on')

print(f"\n    {'Venue':<26} {'FY26':>11} {'OFF FY27':>11} {'OFF %':>7} {'ON FY27':>11} {'ON %':>7}  Note")
for v in problem_venues:
    o = off.get(v); n = on.get(v)
    if not o or not n: continue
    note = (n.get('guardrail_note', '') or '')[:70]
    print(f"    {v:<26} ${o['fy26_forecast']:>9,.0f} ${o['fy27_forecast']:>9,.0f} {o['growth_pct']:>6.1f}% ${n['fy27_forecast']:>9,.0f} {n['growth_pct']:>6.1f}%  {note}")

print("\n[3] What each tightened band catches:")
print("    Mature  band: 0.80x to 1.30x annualized FY26  (was 0.75x to 1.40x)")
print("    Ramping band: 0.75x to 1.30x annualized FY26  PLUS recent run-rate x 1.20")
print()
print("    Expected after fixes:")
print("    - Cronulla: floor lifts from -25% to -20% baseline; the seasonality fix")
print("      requires re-forecast to take effect on Prophet's fitted patterns")
print("    - Circular Quay (+41.6%) → should be clamped to ~+30%")
print("    - Rouse Hill (+38.3%) → should be clamped to ~+30%")
print("    - Southbank: tighter via recent-run-rate × 1.20 instead of annualised")
print()
print("    NOTE: To get the FULL benefit (stronger seasonality, tighter trend),")
print("    you need to RE-RUN the forecast in the UI. The guardrails above are immediate.")

print("\nDONE")
