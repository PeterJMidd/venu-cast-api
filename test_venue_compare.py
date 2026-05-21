"""Verify /venue-compare returns clean side-by-side venue data."""
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

# Forecast a few venues
venues = ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Carlton', 'Yo-Chi Gouger St',
          'Yo-Chi Barangaroo', 'Yo-Chi Burwood']
requests.post(f'{BASE}/generate', json={'venues': venues}, timeout=900)

print("\n[2] Compare 4 venues from different clusters:")
compare = ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Carlton', 'Yo-Chi Gouger St']
r = requests.get(f'{BASE}/venue-compare?venues=' + '|'.join(compare), timeout=30)
data = r.json()
print(f"    Status: {r.status_code}, count={data['count']}\n")

print(f"    {'Venue':<22} {'State':<5} {'Cluster':<10} {'Cohort':<20} {'FY26 $':>14} {'FY27 $':>14} {'Growth':>8}")
for v in data['venues']:
    if 'error' in v:
        print(f"    {v['venue']:<22} ERR: {v['error']}")
        continue
    g = f"{v['growth_pct']:.1f}%" if v['growth_pct'] is not None else '-'
    print(f"    {v['venue']:<22} {v['state']:<5} {v['cluster']:<10} {v['cohort']:<20} ${v['fy26_total']:>12,.0f} ${v['fy27_total']:>12,.0f} {g:>8}")

print("\n[3] Monthly comparison (Jul-Dec FY27 forecast $) for the 4 venues:")
print(f"    {'Venue':<22} " + ''.join(f"{lbl:>11}" for lbl in ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']))
for v in data['venues']:
    if 'error' in v:
        continue
    fy27 = [m['fy27'] for m in v['monthly'][:6]]
    print(f"    {v['venue']:<22} " + ''.join(f"${x:>10,.0f}" for x in fy27))

print("\n[4] Monthly growth % for the 4 venues:")
print(f"    {'Venue':<22} " + ''.join(f"{lbl:>9}" for lbl in ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']))
for v in data['venues']:
    if 'error' in v:
        continue
    g = [m['growth_pct'] for m in v['monthly']]
    g_str = ''.join(f"{x:>8.1f}%" if x is not None else f"{'-':>9}" for x in g)
    print(f"    {v['venue']:<22} {g_str}")

print("\n[5] Edge cases:")

# Invalid venue
print("\n    Invalid venue:")
r = requests.get(f'{BASE}/venue-compare?venues=Foo|Bar', timeout=10)
print(f"      Response: {r.json()}")

# Just 2 venues
print("\n    Just 2 venues:")
r = requests.get(f'{BASE}/venue-compare?venues=Yo-Chi Albert St|Yo-Chi Balaclava', timeout=10)
d = r.json()
print(f"      Count: {d['count']}")
for v in d['venues']:
    print(f"      {v['venue']}: FY27 ${v['fy27_total']:,.0f}")

# Empty
print("\n    Empty:")
r = requests.get(f'{BASE}/venue-compare?venues=', timeout=10)
print(f"      Status: {r.status_code}, body: {r.json()}")

print("\nDONE")
