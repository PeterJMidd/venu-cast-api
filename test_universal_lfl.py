"""Verify the universal LFL target hits exactly, distributing the gap by monthly weights."""
import os, time, requests
DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'

for _ in range(15):
    try:
        if requests.get(f'{BASE}/health', timeout=3).status_code == 200:
            break
    except Exception:
        time.sleep(2)

print("[1] Upload + run forecast on 3 mature LFL venues + 1 new venue (excluded from LFL pool)...")
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
              json={'venues': ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Carlton', 'Yo-Chi Burwood']},
              timeout=900)

print("\n[2] Baseline LFL (universal LFL DISABLED):")
r = requests.post(f'{BASE}/universal-lfl/set',
                  json={'enabled': False, 'target_pct': 0,
                        'monthly_weights': [0]*12}, timeout=10)
r = requests.get(f'{BASE}/sales-detail?cohort=FY26%20Existing', timeout=15)
s = r.json()['summary']
print(f"    LFL FY26: ${s['lfl_fy26']:,.0f}")
print(f"    LFL FY27 (pre): ${s['lfl_fy27']:,.0f}")
print(f"    LFL growth %: {s['lfl_growth_pct']:.2f}% (this is the BASELINE — i.e. what model produces today)")
baseline_lfl_growth = s['lfl_growth_pct']

print("\n[3] Enable Universal LFL: target +3%, weights = example (0,0,0,5,10,10,10,10,10,15,15,15):")
weights = [0, 0, 0, 5, 10, 10, 10, 10, 10, 15, 15, 15]
print(f"    Weight sum: {sum(weights)}")
r = requests.post(f'{BASE}/universal-lfl/set',
                  json={'enabled': True, 'target_pct': 3,
                        'monthly_weights': weights}, timeout=10)
print(f"    Set status: {r.status_code}  config: {r.json()['config']['target_pct']}% target, weights normalized={r.json()['config']['monthly_weights']}")

print("\n[4] LFL after adjustment — should hit exactly +3%:")
r = requests.get(f'{BASE}/sales-detail?cohort=FY26%20Existing', timeout=15)
data = r.json()
s = data['summary']
print(f"    LFL FY26: ${s['lfl_fy26']:,.0f}")
print(f"    LFL FY27 (adjusted): ${s['lfl_fy27']:,.0f}")
print(f"    LFL growth %: {s['lfl_growth_pct']:.2f}% (TARGET: +3.00%)")
hit = abs(s['lfl_growth_pct'] - 3.0) < 0.05
print(f"    {'[OK]' if hit else '[FAIL]'}")

if 'universal_lfl' in data and data['universal_lfl']:
    ul = data['universal_lfl']
    print(f"\n    Diagnostic:")
    print(f"      Pre-adjustment FY27: ${ul['lfl_fy27_pre_total']:,.0f}")
    print(f"      Target FY27: ${ul['lfl_fy27_target']:,.0f}")
    print(f"      Gap to bridge: ${ul['gap']:,.0f}")
    print(f"      Venues in LFL pool: {ul['venues_in_pool']}")

print("\n[5] Verify monthly distribution — check share of growth comes from late months:")
m = r.json()['monthly']
print(f"    {'Month':<5} {'FY26':>12} {'FY27 (LFL)':>14} {'FY27 LFL Growth':>16} {'Weight':>8}")
fy_order = ['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun']
total_pre = sum(mo['lfl_fy26'] for mo in m if mo['lfl_fy26'])
total_post = sum(mo['lfl_fy27'] for mo in m if mo['lfl_fy27'])
for i, mo in enumerate(m):
    lg = mo.get('lfl_growth_pct')
    g = f'{lg:.1f}%' if lg is not None else '-'
    w = weights[i] if i < len(weights) else 0
    print(f"    {mo['month_label']:<5} {mo['lfl_fy26']:>12,.0f} {mo['lfl_fy27']:>14,.0f} {g:>16} {w:>7}%")

print(f"\n    Total LFL FY27: ${total_post:,.0f}   FY26: ${total_pre:,.0f}   growth: {(total_post - total_pre) / total_pre * 100:.2f}%")

print("\n[6] Verify NEW venue (Burwood, FY26 New Opening) is NOT affected by universal LFL:")
r = requests.get(f'{BASE}/venues-detail', timeout=15)
for v in r.json()['venues']:
    if v['venue'] in ['Yo-Chi Burwood']:
        print(f"    {v['venue']}: cohort={v['cohort']}, FY27 ${v['fy27_forecast']:,.0f} (universal LFL should NOT touch this)")
        if 'guardrail_note' in v and 'Universal LFL' in (v.get('guardrail_note') or ''):
            print(f"    [FAIL] LFL note appeared on non-LFL venue: {v['guardrail_note']}")
        else:
            print(f"    [OK] non-LFL venue untouched")

print("\n[7] Disable Universal LFL → should return to baseline:")
requests.post(f'{BASE}/universal-lfl/set',
              json={'enabled': False, 'target_pct': 0, 'monthly_weights': [0]*12}, timeout=10)
r = requests.get(f'{BASE}/sales-detail?cohort=FY26%20Existing', timeout=15)
s = r.json()['summary']
print(f"    LFL growth: {s['lfl_growth_pct']:.2f}% (should equal baseline {baseline_lfl_growth:.2f}%)")
print(f"    {'[OK]' if abs(s['lfl_growth_pct'] - baseline_lfl_growth) < 0.05 else '[FAIL]'}")

print("\n[8] Re-enable with target 0% (zero gap) — should leave LFL unchanged:")
requests.post(f'{BASE}/universal-lfl/set',
              json={'enabled': True, 'target_pct': baseline_lfl_growth,
                    'monthly_weights': [100/12]*12}, timeout=10)
r = requests.get(f'{BASE}/sales-detail?cohort=FY26%20Existing', timeout=15)
s = r.json()['summary']
print(f"    Target = baseline {baseline_lfl_growth:.2f}% → growth: {s['lfl_growth_pct']:.2f}%")
print(f"    {'[OK]' if abs(s['lfl_growth_pct'] - baseline_lfl_growth) < 0.1 else '[FAIL]'}")

print("\nDONE")
