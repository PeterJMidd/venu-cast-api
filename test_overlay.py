"""Verify peer seasonality overlay fixes the flat shape for sub-365-day venues."""
import os, time, requests
DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'

for _ in range(15):
    try:
        if requests.get(f'{BASE}/health', timeout=3).status_code == 200:
            break
    except Exception:
        time.sleep(2)

print("[1] Upload + forecast: short-history venues + mature peers...")
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
# Need mature peers in same state for the overlay
venues_to_test = ['Yo-Chi Chippendale', 'Yo-Chi Castle Towers', 'Yo-Chi Bondi Junction',
                  'Yo-Chi Burwood', 'Yo-Chi Indooroopilly', 'Yo-Chi Kawana Waters',
                  'Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Bondi',
                  'Yo-Chi Cronulla', 'Yo-Chi Manly']
requests.post(f'{BASE}/generate', json={'venues': venues_to_test}, timeout=900)

# Disable LFL + softening to see pure model output
requests.post(f'{BASE}/universal-lfl/set', json={'enabled': False, 'target_pct': 0, 'monthly_weights': [0]*12}, timeout=10)
requests.post(f'{BASE}/softening/set', json={'enabled': False, 'beta': 1.0}, timeout=10)

print("\n[2] FY27 monthly shape for sub-365 venues (look for variation):")
r = requests.get(f'{BASE}/venues-detail', timeout=20)
data = r.json()
months = ['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun']
short_history = ['Yo-Chi Chippendale', 'Yo-Chi Castle Towers', 'Yo-Chi Bondi Junction',
                 'Yo-Chi Burwood', 'Yo-Chi Indooroopilly', 'Yo-Chi Kawana Waters']
for vname in short_history:
    v = next((x for x in data['venues'] if x['venue'] == vname), None)
    if not v: continue
    fy27 = v.get('monthly_fy27') or []
    if not fy27: continue
    mn, mx = min(fy27), max(fy27)
    avg = sum(fy27)/len(fy27)
    var_ratio = (mx - mn) / avg * 100 if avg > 0 else 0
    peak_m = months[fy27.index(mx)]
    trough_m = months[fy27.index(mn)]
    print(f"  {vname:<22} model={v.get('model'):<35}  peak={peak_m}@${mx/1000:>4,.0f}k  trough={trough_m}@${mn/1000:>4,.0f}k  variation={var_ratio:.0f}% of avg")

print("\n[3] Mature comparison (network seasonality reference):")
for vname in ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Cronulla']:
    v = next((x for x in data['venues'] if x['venue'] == vname), None)
    if not v: continue
    fy27 = v.get('monthly_fy27') or []
    if not fy27: continue
    mn, mx = min(fy27), max(fy27)
    avg = sum(fy27)/len(fy27)
    var_ratio = (mx - mn) / avg * 100 if avg > 0 else 0
    peak_m = months[fy27.index(mx)]
    trough_m = months[fy27.index(mn)]
    print(f"  {vname:<22} model={v.get('model'):<35}  peak={peak_m}@${mx/1000:>4,.0f}k  trough={trough_m}@${mn/1000:>4,.0f}k  variation={var_ratio:.0f}% of avg")

requests.post(f'{BASE}/softening/set', json={'enabled': True, 'beta': 0.7}, timeout=10)
print('\nDONE')
