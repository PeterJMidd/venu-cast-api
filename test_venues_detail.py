"""Verify /venues-detail endpoint output."""
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

# A mix of cohorts
venues = ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Burwood', 'Yo-Chi Castle Towers',
          'Yo-Chi New Venue 01', 'Yo-Chi Carlton', 'Yo-Chi Gouger St', 'Yo-Chi Barangaroo']
requests.post(f'{BASE}/generate', json={'venues': venues}, timeout=900)

print("\n[2] /venues-detail (no filter):")
r = requests.get(f'{BASE}/venues-detail', timeout=30)
data = r.json()
print(f"    Venues: {data['venue_count']}")

print(f"\n    {'Venue':<24} {'State':<5} {'Cluster':<10} {'Cohort':<20} {'Open Date':<12} {'FY26':>12} {'FY27':>12} {'Growth':>8} {'ATP':>7} {'Guests':>9}")
for v in data['venues']:
    print(f"    {v['venue']:<24} {v['state']:<5} {v['cluster']:<10} {v['cohort']:<20} {v['opening_date'] or '-':<12} ${v['fy26_forecast']:>10,.0f} ${v['fy27_forecast']:>10,.0f}"
          f" {(str(v['growth_pct'])+'%') if v['growth_pct'] is not None else '-':>8} ${v['avg_atp'] or 0:>6.2f} {int(v['transactions_fy27'] or 0):>9,}")

print("\n[3] Auto-generated comments:")
for v in data['venues']:
    print(f"\n    {v['venue']}:")
    print(f"      Comment: {v['comment']}")

print("\n[4] State summary:")
print(f"    {'State':<6} {'Venues':>7} {'FY26 $':>14} {'FY27 $':>14} {'Growth':>8} {'Guests':>10}")
for s in data['state_summary']:
    g = f"{s['growth_pct']:.1f}%" if s['growth_pct'] is not None else '-'
    print(f"    {s['state']:<6} {s['venue_count']:>7} ${s['fy26']:>12,.0f} ${s['fy27']:>12,.0f} {g:>8} {int(s['transactions_fy27']):>10,}")

print("\n[5] Cluster summary:")
print(f"    {'State':<6} {'Cluster':<12} {'Venues':>7} {'FY26 $':>14} {'FY27 $':>14} {'Growth':>8}")
for s in data['cluster_summary']:
    g = f"{s['growth_pct']:.1f}%" if s['growth_pct'] is not None else '-'
    print(f"    {s['state']:<6} {s['cluster']:<12} {s['venue_count']:>7} ${s['fy26']:>12,.0f} ${s['fy27']:>12,.0f} {g:>8}")

print("\n[6] Filter test — state=VIC only:")
r = requests.get(f'{BASE}/venues-detail?state=VIC', timeout=30)
d = r.json()
print(f"    Venue count: {d['venue_count']}")
for v in d['venues']:
    print(f"    - {v['venue']} ({v['state']}, {v['cluster']})")

print("\nDONE")
