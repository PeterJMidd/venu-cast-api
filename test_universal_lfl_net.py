"""Verify the universal LFL hits target on NET (servings + retail + discounts)
   and ALSO sensibly on servings-only (slightly different but proportional).
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

print("[1] Upload + forecast 3 mature venues...")
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

# Baseline: LFL disabled
requests.post(f'{BASE}/universal-lfl/set',
              json={'enabled': False, 'target_pct': 0, 'monthly_weights': [0]*12}, timeout=10)

print("\n[2] Baseline LFL growth — by sales type (LFL DISABLED):")
baselines = {}
for st in ['servings', 'retail', 'pos_discounts', 'all']:
    r = requests.get(f'{BASE}/sales-detail?cohort=FY26%20Existing&sales_type={st}', timeout=15)
    s = r.json()['summary']
    baselines[st] = s['lfl_growth_pct']
    print(f"    {st:<14} LFL FY26 ${s['lfl_fy26']:>14,.0f}  FY27 ${s['lfl_fy27']:>14,.0f}  growth {s['lfl_growth_pct']:>+6.2f}%")

print("\n[3] Enable Universal LFL +3% with Q4-Q3 ramp weights:")
r = requests.post(f'{BASE}/universal-lfl/set',
                  json={'enabled': True, 'target_pct': 3,
                        'monthly_weights': [0, 0, 0, 5, 10, 10, 10, 10, 10, 15, 15, 15]}, timeout=10)
print(f"    Config set: {r.json()['config']['target_pct']}% target")

print("\n[4] After LFL adjustment — by sales type:")
print(f"    {'Sales Type':<14} {'LFL FY26':>14} {'LFL FY27':>14} {'Growth':>8} {'vs Target':>12}")
for st in ['servings', 'retail', 'pos_discounts', 'all']:
    r = requests.get(f'{BASE}/sales-detail?cohort=FY26%20Existing&sales_type={st}', timeout=15)
    s = r.json()['summary']
    target_check = '[OK]' if abs(s['lfl_growth_pct'] - 3.0) < 0.05 else f"[OFF by {s['lfl_growth_pct']-3:.2f}pp]"
    print(f"    {st:<14} ${s['lfl_fy26']:>13,.0f} ${s['lfl_fy27']:>13,.0f} {s['lfl_growth_pct']:>+7.2f}% {target_check:>12}")

print("\n[5] Expected behaviour:")
print("    - 'all' (net): exactly +3.00%  (this is what the multiplier was sized for)")
print("    - 'servings', 'retail', 'pos_discounts': also +3.00% (same multiplier applied to each)")

print("\nDONE")
