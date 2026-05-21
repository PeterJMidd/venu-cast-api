"""Verify LFL target hits exactly +3% on WHICHEVER sales_type the user views."""
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

print("\n[2] Enable Universal LFL: target +3% with Q4-Q3 ramp weights:")
requests.post(f'{BASE}/universal-lfl/set',
              json={'enabled': True, 'target_pct': 3,
                    'monthly_weights': [0,0,0,5,10,10,10,10,10,15,15,15]}, timeout=10)

print("\n[3] LFL growth by sales_type — each should hit +3% on its own basis:")
print(f"    {'View':<18} {'LFL FY26':>14} {'LFL FY27':>14} {'Growth':>8} {'Status':>8}")
for st in ['servings', 'retail', 'pos_discounts', 'all']:
    r = requests.get(f'{BASE}/sales-detail?cohort=FY26%20Existing&sales_type={st}', timeout=15)
    s = r.json()['summary']
    g = s['lfl_growth_pct']
    target_check = '[OK]' if abs(g - 3.0) < 0.05 else f'[off]'
    print(f"    {st:<18} ${s['lfl_fy26']:>13,.0f} ${s['lfl_fy27']:>13,.0f} {g:>+7.2f}% {target_check:>8}")

print("\nDONE")
