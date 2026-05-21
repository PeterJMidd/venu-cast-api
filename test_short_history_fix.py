"""Verify the bypass threshold fix for short-history venues."""
import os, time, requests
DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'

for _ in range(15):
    try:
        if requests.get(f'{BASE}/health', timeout=3).status_code == 200:
            break
    except Exception:
        time.sleep(2)

print("[1] Upload + forecast the problematic short-history venues...")
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
venues = ['Yo-Chi Indooroopilly', 'Yo-Chi Kawana Waters', 'Yo-Chi Chippendale',
          'Yo-Chi Albert St', 'Yo-Chi Balaclava']  # mix of new + mature
requests.post(f'{BASE}/generate', json={'venues': venues}, timeout=900)

# Turn off softening/LFL to isolate the model output
requests.post(f'{BASE}/universal-lfl/set', json={'enabled': False, 'target_pct': 0, 'monthly_weights': [0]*12}, timeout=10)
requests.post(f'{BASE}/softening/set', json={'enabled': False, 'beta': 1.0}, timeout=10)

print("\n[2] FY27 by month (no softening, no LFL — just Prophet + guardrails):")
r = requests.get(f'{BASE}/venues-detail?cannibalisation=on&guardrails=on', timeout=20)
data = r.json()
months = ['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun']
for vname in venues:
    v = next((x for x in data['venues'] if x['venue'] == vname), None)
    if not v: continue
    print(f"\n  === {v['venue']} ({v.get('cohort')}, model={v.get('model')}) ===")
    print(f"  FY26: ${v['fy26_forecast']:,.0f}  FY27: ${v['fy27_forecast']:,.0f}  Growth: {v['growth_pct']}%")
    print(f"  Avg daily: FY26 ${v.get('avg_daily_fy26',0):,.0f}/d  FY27 ${v.get('avg_daily_fy27',0):,.0f}/d")
    fy27 = v.get('monthly_fy27') or []
    out = '  FY27: ' + ' '.join(f'{months[i]}:${fy27[i]/1000:>4.0f}k' for i in range(12))
    print(out)

# Re-enable softening for final state
requests.post(f'{BASE}/softening/set', json={'enabled': True, 'beta': 0.7}, timeout=10)
print('\nDONE — softening re-enabled')
