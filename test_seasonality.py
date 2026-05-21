"""Verify new venues now get peer-borrowed seasonality + school-holiday lifts."""
import os, time, requests
DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'

for _ in range(15):
    try:
        if requests.get(f'{BASE}/health', timeout=3).status_code == 200:
            break
    except Exception:
        time.sleep(2)

print("[1] Upload + forecast: new venues + mature peers for comparison...")
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
# Generate forecast for new venues PLUS mature peers (needed for peer-seasonality lookup)
all_v = requests.get(f'{BASE}/venues', timeout=15).json()['venues']
target_new = ['Yo-Chi Indooroopilly', 'Yo-Chi Kawana Waters']
target_mature = ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Bondi', 'Yo-Chi Cronulla']
requests.post(f'{BASE}/generate', json={'venues': target_new + target_mature}, timeout=900)

requests.post(f'{BASE}/universal-lfl/set', json={'enabled': False, 'target_pct': 0, 'monthly_weights': [0]*12}, timeout=10)
requests.post(f'{BASE}/softening/set', json={'enabled': False, 'beta': 1.0}, timeout=10)

print("\n[2] FY27 monthly for new venues (look for variation, not flat):")
r = requests.get(f'{BASE}/venues-detail', timeout=20)
data = r.json()
months = ['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun']
for vname in target_new:
    v = next((x for x in data['venues'] if x['venue'] == vname), None)
    if not v: continue
    print(f"\n  === {v['venue']} ({v.get('state')}, cluster={v.get('cluster')}, model={v.get('model')}) ===")
    fy27 = v.get('monthly_fy27') or []
    if fy27:
        avg = sum(fy27)/len(fy27)
        for i in range(12):
            ratio = fy27[i]/avg if avg > 0 else 1
            bar = '#' * int(ratio * 20)
            print(f"  {months[i]:<5}  ${fy27[i]/1000:>5,.0f}k  ratio {ratio:.2f}  {bar}")

# Also compare with a mature beach peer for visual reference
print("\n[3] For comparison — mature peer (Cronulla) seasonality pattern:")
v = next((x for x in data['venues'] if x['venue'] == 'Yo-Chi Cronulla'), None)
if v:
    fy27 = v.get('monthly_fy27') or []
    avg = sum(fy27)/len(fy27) if fy27 else 1
    for i in range(12):
        ratio = fy27[i]/avg if avg > 0 else 1
        bar = '#' * int(ratio * 20)
        print(f"  {months[i]:<5}  ${fy27[i]/1000:>5,.0f}k  ratio {ratio:.2f}  {bar}")

requests.post(f'{BASE}/softening/set', json={'enabled': True, 'beta': 0.7}, timeout=10)
print('\nDONE')
