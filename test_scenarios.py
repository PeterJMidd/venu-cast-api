"""E2E test: scenarios with growth targets actually hit the target, comments persist, switching works."""
import os, time, requests
DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'

for _ in range(15):
    try:
        if requests.get(f'{BASE}/health', timeout=3).status_code == 200:
            break
    except Exception:
        time.sleep(2)

print("[1] Upload + pre-fetch + generate...")
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
venues = ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Carlton', 'Yo-Chi Gouger St']
requests.post(f'{BASE}/generate', json={'venues': venues}, timeout=900)

print("\n[2] Get FY26 baselines (no scenario):")
r = requests.get(f'{BASE}/venues-detail', timeout=15)
data = r.json()
baselines = {}
for v in data['venues']:
    baselines[v['venue']] = v['fy26_forecast']
    print(f"    {v['venue']}: FY26 ${v['fy26_forecast']:,.0f}, FY27 ${v['fy27_forecast']:,.0f}, growth {v['growth_pct']}%")

print("\n[3] Create scenario 'Base Case':")
r = requests.post(f'{BASE}/scenario/create', json={'name': 'Base Case'}, timeout=10)
print(f"    Status: {r.status_code}, response: {r.json()}")

print("\n[4] Set Albert St target +8%, Carlton target +5%, Gouger -3% with comments:")
for venue, growth, comments in [
    ('Yo-Chi Albert St', 8.0, ['Strong Brisbane growth', 'New menu launch', '', '', '']),
    ('Yo-Chi Carlton', 5.0, ['Recovery from Q4 dip', '', '', '', '']),
    ('Yo-Chi Gouger St', -3.0, ['Renovation Q1', 'Competitor opening Mar', '', '', '']),
]:
    r = requests.post(f'{BASE}/scenario/venue',
                      json={'scenario': 'Base Case', 'venue': venue,
                            'target_growth_pct': growth, 'comments': comments},
                      timeout=10)
    print(f"    {venue}: target={growth}%, status={r.status_code}")

print("\n[5] Verify Base Case results — targets should be hit exactly:")
r = requests.get(f'{BASE}/venues-detail', timeout=15)
data = r.json()
print(f"    Active scenario: {data['active_scenario']}")
print(f"\n    {'Venue':<22} {'FY26 $':>12} {'FY27 $':>12} {'Actual %':>10} {'Target %':>10} {'Hit?':>6}")
for v in data['venues']:
    target = v['scenario_target_growth_pct']
    if target is not None:
        delta = abs(v['growth_pct'] - target)
        hit = '[OK]' if delta < 0.5 else '[FAIL]'
    else:
        hit = '-'
    t_str = f"{target}%" if target is not None else '-'
    print(f"    {v['venue']:<22} ${v['fy26_forecast']:>10,.0f} ${v['fy27_forecast']:>10,.0f} {v['growth_pct']:>9.1f}% {t_str:>10} {hit:>6}")
    if v['scenario_target_growth_pct'] is not None:
        print(f"        Comments: {' | '.join(c for c in v['scenario_comments'] if c)}")

print("\n[6] Create 'Stretch' scenario and set Albert +15%:")
requests.post(f'{BASE}/scenario/create', json={'name': 'Stretch'}, timeout=10)
requests.post(f'{BASE}/scenario/venue',
              json={'scenario': 'Stretch', 'venue': 'Yo-Chi Albert St',
                    'target_growth_pct': 15.0, 'comments': ['Aggressive case']}, timeout=10)
r = requests.get(f'{BASE}/venues-detail', timeout=15)
data = r.json()
for v in data['venues']:
    if v['venue'] == 'Yo-Chi Albert St':
        print(f"    Active scenario: {data['active_scenario']}")
        print(f"    Albert St FY27 ${v['fy27_forecast']:,.0f} (growth {v['growth_pct']}%, target {v['scenario_target_growth_pct']}%)")

print("\n[7] Switch back to 'Base Case' — Albert should return to +8%:")
requests.post(f'{BASE}/scenario/activate', json={'name': 'Base Case'}, timeout=10)
r = requests.get(f'{BASE}/venues-detail', timeout=15)
data = r.json()
for v in data['venues']:
    if v['venue'] == 'Yo-Chi Albert St':
        print(f"    Albert St FY27 ${v['fy27_forecast']:,.0f} (growth {v['growth_pct']}%, target {v['scenario_target_growth_pct']}%)")

print("\n[8] Deactivate scenario — back to raw forecast:")
requests.post(f'{BASE}/scenario/activate', json={'name': ''}, timeout=10)
r = requests.get(f'{BASE}/venues-detail', timeout=15)
data = r.json()
print(f"    Active: {data['active_scenario']}")
for v in data['venues'][:2]:
    print(f"    {v['venue']}: FY27 ${v['fy27_forecast']:,.0f} (growth {v['growth_pct']}%)")

print("\n[9] Save snapshot — should include scenarios:")
r = requests.post(f'{BASE}/snapshot/save',
                  json={'name': 'With Scenarios Test', 'note': 'Base Case + Stretch'}, timeout=30)
print(f"    Saved: {r.json()}")

print("\n[10] Reset, reload — scenarios should be restored:")
requests.post(f'{BASE}/reset', timeout=10)
r = requests.get(f'{BASE}/scenario/list', timeout=10)
print(f"    After reset: scenarios={r.json()}")
requests.post(f'{BASE}/snapshot/load', json={'filename': 'With Scenarios Test'}, timeout=30)
r = requests.get(f'{BASE}/scenario/list', timeout=10)
data = r.json()
print(f"    After load: {len(data['scenarios'])} scenarios, active={data['active']}")
for s in data['scenarios']:
    print(f"      - {s['name']}: {s['venue_count']} venue overrides")

print("\n[11] Cleanup snapshot:")
requests.post(f'{BASE}/snapshot/delete', json={'filename': 'With Scenarios Test'}, timeout=10)
print("    OK")

print("\nDONE")
