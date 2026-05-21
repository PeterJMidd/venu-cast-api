"""Verify Prophet softening pulls extremes toward FY26 baseline."""
import os, time, requests
DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'

for _ in range(15):
    try:
        if requests.get(f'{BASE}/health', timeout=3).status_code == 200:
            break
    except Exception:
        time.sleep(2)

print("[1] Upload + forecast the extreme venues...")
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
requests.post(f'{BASE}/generate', json={'venues': ['Yo-Chi Cronulla', 'Yo-Chi Manly', 'Yo-Chi Circular Quay',
                                                    'Yo-Chi Newtown', 'Yo-Chi Bondi']}, timeout=900)

# Turn off all adjustments except softening to isolate its effect
requests.post(f'{BASE}/universal-lfl/set', json={'enabled': False, 'target_pct': 0, 'monthly_weights': [0]*12}, timeout=10)

target = ['Yo-Chi Cronulla', 'Yo-Chi Manly', 'Yo-Chi Circular Quay', 'Yo-Chi Newtown', 'Yo-Chi Bondi']

print("\n[2] Compare softening levels (no LFL, guardrails+cannibalisation ON):")
print(f"\n    {'Venue':<22} {'FY26':>10} {'β=1.0 (raw)':>13} {'β=0.85':>10} {'β=0.7':>10} {'β=0.5':>10}")
for beta in [1.0, 0.85, 0.7, 0.5]:
    requests.post(f'{BASE}/softening/set',
                  json={'enabled': beta < 0.99, 'beta': beta}, timeout=10)
    r = requests.get(f'{BASE}/venues-detail?cannibalisation=on&guardrails=on', timeout=20)
    venues_d = {v['venue']: v for v in r.json()['venues']}
    if beta == 1.0:
        labels = {v: f"${venues_d[v]['fy27_forecast']/1000:>11,.0f}k" for v in target if v in venues_d}
        raw_growths = {v: venues_d[v]['growth_pct'] for v in target if v in venues_d}
    else:
        for v in target:
            if v in venues_d:
                pass  # collect later
for beta in [1.0, 0.85, 0.7, 0.5]:
    requests.post(f'{BASE}/softening/set', json={'enabled': beta < 0.99, 'beta': beta}, timeout=10)

# Re-fetch cleanly per beta
print(f"    {'Venue':<22} {'FY26':>11}   β=1.0     β=0.85    β=0.7     β=0.5")
rows = {}
for beta in [1.0, 0.85, 0.7, 0.5]:
    requests.post(f'{BASE}/softening/set', json={'enabled': beta < 0.99, 'beta': beta}, timeout=10)
    r = requests.get(f'{BASE}/venues-detail?cannibalisation=on&guardrails=on', timeout=20)
    venues_d = {v['venue']: v for v in r.json()['venues']}
    for v in target:
        if v not in rows: rows[v] = {}
        if v in venues_d:
            rows[v][beta] = (venues_d[v]['fy27_forecast'], venues_d[v]['growth_pct'])

for v in target:
    if v not in rows or 1.0 not in rows[v]: continue
    fy26 = next((venues_d[v]['fy26_forecast'] for venues_d in [{v: rows[v]}] if False), None)
    # just get growth %s
    r_raw = rows[v].get(1.0, (0, 0))
    r_85 = rows[v].get(0.85, (0, 0))
    r_7 = rows[v].get(0.7, (0, 0))
    r_5 = rows[v].get(0.5, (0, 0))
    print(f"    {v:<22}  growth %  {r_raw[1]:>+7.1f}% {r_85[1]:>+9.1f}% {r_7[1]:>+9.1f}% {r_5[1]:>+9.1f}%")
    print(f"    {'':<22}  FY27 $    ${r_raw[0]/1000:>5,.0f}k  ${r_85[0]/1000:>7,.0f}k  ${r_7[0]/1000:>7,.0f}k  ${r_5[0]/1000:>7,.0f}k")

# Reset to default
requests.post(f'{BASE}/softening/set', json={'enabled': True, 'beta': 0.7}, timeout=10)
print("\nReset to default β=0.7 (moderate). DONE")
