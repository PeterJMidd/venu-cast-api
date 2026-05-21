"""Inject real impact values and verify the cannibalisation math.
   New Venue 01 (opens 2026-07-01) → Albert St -10% M1, -7% M2, -5% M3, -3% M4, -2% M5, -1% M6.
   Expected: Albert St's FY27 totals drop by a known amount.
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

print("[1] Upload + generate forecast...")
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
              json={'venues': ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Venue_72']},
              timeout=900)

print("\n[2] Baseline (no impacts, cannibalisation ON):")
r = requests.get(f'{BASE}/venues-detail?cannibalisation=on', timeout=15)
albert_base = next(v for v in r.json()['venues'] if v['venue'] == 'Yo-Chi Albert St')['fy27_forecast']
print(f"    Albert St FY27: ${albert_base:,.0f}")

print("\n[3] Inject impact: New Venue 01 → Albert St [-10, -7, -5, -3, -2, -1]:")
r = requests.post(f'{BASE}/venue-impacts/set', json={
    'impactor': 'Yo-Chi Venue_72',
    'impacts': [
        {'impacted_venue': 'Yo-Chi Albert St',
         'monthly_impact_pcts': [-10, -7, -5, -3, -2, -1]}
    ]
}, timeout=10)
print(f"    Status: {r.status_code}, set: {r.json()}")

print("\n[4] With cannibalisation ON (impacts now populated):")
r = requests.get(f'{BASE}/venues-detail?cannibalisation=on', timeout=15)
data_on = r.json()
albert_on = next(v for v in data_on['venues'] if v['venue'] == 'Yo-Chi Albert St')
print(f"    Albert St FY27: ${albert_on['fy27_forecast']:,.0f}")
print(f"    Guardrail/note: '{albert_on.get('guardrail_note','')}'")

print("\n[5] With cannibalisation OFF (fall-back — sanity check):")
r = requests.get(f'{BASE}/venues-detail?cannibalisation=off', timeout=15)
data_off = r.json()
albert_off = next(v for v in data_off['venues'] if v['venue'] == 'Yo-Chi Albert St')
print(f"    Albert St FY27: ${albert_off['fy27_forecast']:,.0f}")
print(f"    (should equal baseline ${albert_base:,.0f}: {'[OK]' if abs(albert_off['fy27_forecast'] - albert_base) < 1 else '[FAIL]'})")

delta = albert_on['fy27_forecast'] - albert_off['fy27_forecast']
pct = delta / albert_off['fy27_forecast'] * 100
print(f"\n    Delta ON - OFF: ${delta:,.0f} ({pct:+.1f}%)  (expected negative)")
print(f"    {'[OK]' if delta < 0 else '[FAIL]'}")

# Estimated impact: opens 2026-07-01 → impacts Jul-Dec 2026
# Albert St monthly ~ $500K; expected hit on Jul = 10%, Aug 7%, Sep 5%, Oct 3%, Nov 2%, Dec 1%
# Avg ~ 4.7% × half-year = effectively 2.3% of annual = -$140K hit
expected_hit_pct = (10 + 7 + 5 + 3 + 2 + 1) / 12 / 100  # 6 months at avg 4.67% = ~2.3% annual
expected_hit = -albert_base * expected_hit_pct
print(f"    Expected hit: ${expected_hit:,.0f} (-{expected_hit_pct*100:.1f}% annual)")

print("\n[6] Balaclava (not impacted) should be unchanged:")
bal_on = next(v for v in data_on['venues'] if v['venue'] == 'Yo-Chi Balaclava')['fy27_forecast']
bal_off = next(v for v in data_off['venues'] if v['venue'] == 'Yo-Chi Balaclava')['fy27_forecast']
print(f"    Balaclava ON ${bal_on:,.0f}  vs  OFF ${bal_off:,.0f}  delta ${bal_on - bal_off:,.0f}")
print(f"    {'[OK]' if abs(bal_on - bal_off) < 1 else '[FAIL — should be 0]'}")

print("\n[7] Verify scenario still overrides cannibalisation:")
requests.post(f'{BASE}/scenario/create', json={'name': 'Cann Override'}, timeout=10)
requests.post(f'{BASE}/scenario/venue',
              json={'scenario': 'Cann Override', 'venue': 'Yo-Chi Albert St',
                    'target_growth_pct': 8.0, 'comment': 'Test override'}, timeout=10)
r = requests.get(f'{BASE}/venues-detail?cannibalisation=on', timeout=15)
ao = next(v for v in r.json()['venues'] if v['venue'] == 'Yo-Chi Albert St')
print(f"    Albert St with +8% scenario + cannibalisation ON:")
print(f"    FY27 ${ao['fy27_forecast']:,.0f}, growth {ao['growth_pct']}% (expected ~8%)")
print(f"    {'[OK]' if abs(ao['growth_pct'] - 8.0) < 0.2 else '[FAIL]'} — scenario beats cannibalisation as designed")

print("\nDONE")
