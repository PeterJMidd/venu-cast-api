"""Final test: use Yo-Chi Balaclava as a synthetic 'impactor' with a fake FY27 opening date.
This proves the cannibalisation math without needing a true new venue.
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

print("[1] Upload + generate forecast (Albert St + Balaclava + Carlton)...")
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
requests.post(f'{BASE}/generate',
              json={'venues': ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Carlton']},
              timeout=900)

print("\n[2] Baseline FY27 (cannibalisation OFF):")
r = requests.get(f'{BASE}/venues-detail?cannibalisation=off', timeout=15)
baseline = {v['venue']: v['fy27_forecast'] for v in r.json()['venues']}
for v, val in baseline.items():
    print(f"    {v}: ${val:,.0f}")

print("\n[3] Inject test impact: Balaclava (synthetic opening 2026-09-01) → Albert St [-10,-7,-5,-3,-2,-1]:")
r = requests.post(f'{BASE}/venue-impacts/set', json={
    'impactor': 'Yo-Chi Balaclava',
    'opening_date': '2026-09-01',
    'impacts': [
        {'impacted_venue': 'Yo-Chi Albert St',
         'monthly_impact_pcts': [-10, -7, -5, -3, -2, -1]}
    ]
}, timeout=10)
print(f"    Status: {r.status_code}, set: {r.json().get('status')}")

print("\n[4] With cannibalisation ON:")
r = requests.get(f'{BASE}/venues-detail?cannibalisation=on', timeout=15)
data_on = {v['venue']: v for v in r.json()['venues']}
for vname in ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Carlton']:
    v = data_on[vname]
    delta = v['fy27_forecast'] - baseline[vname]
    pct = delta / baseline[vname] * 100 if baseline[vname] else 0
    flag = '[OK impacted]' if vname == 'Yo-Chi Albert St' else '[OK unchanged]'
    if vname == 'Yo-Chi Albert St' and delta >= 0:
        flag = '[FAIL — should be negative]'
    if vname != 'Yo-Chi Albert St' and abs(delta) > 1:
        flag = '[FAIL — should be 0]'
    print(f"    {vname}: ${v['fy27_forecast']:,.0f}  Δ ${delta:>+10,.0f} ({pct:+.2f}%)  {flag}")
    if v.get('guardrail_note'):
        print(f"        note: '{v['guardrail_note']}'")

print("\n[5] Sanity check: Albert St delta vs expected:")
# Albert ~$500K/month average. Sep -10%, Oct -7%, Nov -5%, Dec -3%, Jan -2%, Feb -1%.
# Each month's actual draw depends on Prophet's monthly forecast, not exactly $500K — so we just check it's
# in the right ballpark (~$130K-$170K hit total).
albert_delta = data_on['Yo-Chi Albert St']['fy27_forecast'] - baseline['Yo-Chi Albert St']
albert_pct = albert_delta / baseline['Yo-Chi Albert St'] * 100
print(f"    Albert St hit: ${albert_delta:,.0f} ({albert_pct:+.2f}% annual)")
print(f"    Expected: ~$-140K to $-180K, around -2% to -3% (28% / 12 ≈ -2.3%)")
if -200000 < albert_delta < -80000:
    print(f"    [OK] in expected range")
else:
    print(f"    [WARN] outside expected range")

print("\n[6] Toggle back OFF → should return exactly to baseline:")
r = requests.get(f'{BASE}/venues-detail?cannibalisation=off', timeout=15)
for v in r.json()['venues']:
    delta = v['fy27_forecast'] - baseline[v['venue']]
    print(f"    {v['venue']}: ${v['fy27_forecast']:,.0f}  Δ ${delta:>+10,.0f}  {'[OK]' if abs(delta) < 1 else '[FAIL]'}")

print("\n[7] Scenario still wins over cannibalisation:")
requests.post(f'{BASE}/scenario/create', json={'name': 'Cann Override'}, timeout=10)
requests.post(f'{BASE}/scenario/venue', json={
    'scenario': 'Cann Override', 'venue': 'Yo-Chi Albert St',
    'target_growth_pct': 5.0, 'comment': 'Test'}, timeout=10)
r = requests.get(f'{BASE}/venues-detail?cannibalisation=on', timeout=15)
for v in r.json()['venues']:
    if v['venue'] == 'Yo-Chi Albert St':
        print(f"    Albert St with +5% scenario + cannibalisation ON:")
        print(f"    FY27 ${v['fy27_forecast']:,.0f}, growth {v['growth_pct']}%  ({'[OK]' if abs(v['growth_pct'] - 5.0) < 0.2 else '[FAIL]'})")
        print(f"    note: '{v.get('guardrail_note','')}'  (should be empty — scenario wins)")

print("\nDONE")
