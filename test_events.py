"""Verify the events aggregation in /sales-detail."""
import os, time, requests
DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'
for _ in range(15):
    try:
        if requests.get(f'{BASE}/health', timeout=3).status_code == 200:
            break
    except Exception:
        time.sleep(2)

print("[1] Upload + pre-fetch + generate forecast...")
files = {
    'sales_history': open(os.path.join(DATA_DIR, '01_Sales_History_Template.xlsx'), 'rb'),
    'atp_history': open(os.path.join(DATA_DIR, 'Average ticket history.xlsx'), 'rb'),
    'location_bi': open(os.path.join(DATA_DIR, 'D03_Location_BI.xlsx'), 'rb'),
    'venue_details': open(os.path.join(DATA_DIR, 'Venue Details.xlsx'), 'rb'),
    'multipliers': open(os.path.join(DATA_DIR, 'venue_city_mapping_updated_1.csv'), 'rb'),
}
requests.post(f'{BASE}/upload', files=files, timeout=120)
for fh in files.values():
    fh.close()
requests.post(f'{BASE}/data-sources', timeout=300)

venues = ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Barangaroo', 'Yo-Chi Gouger St']
requests.post(f'{BASE}/generate', json={'venues': venues}, timeout=900)

print("\n[2] All venues — events per month:")
r = requests.get(f'{BASE}/sales-detail', timeout=30)
data = r.json()
print(f"    Venues: {data['summary']['venues_in_filter']}")
print(f"\n    {'Month':<5}  {'FY26 Hol':>8} {'FY27 Hol':>8} {'FY26 Wx':>7} {'FY27 Wx':>7}  FY26 Holiday names")
for m in data['monthly']:
    h_names = ', '.join(m['fy26_holidays'][:3]) if m['fy26_holidays'] else '—'
    print(f"    {m['month_label']:<5}  {m['fy26_holiday_count']:>8} {m['fy27_holiday_count']:>8} {m['fy26_extreme_count']:>7} {m['fy27_extreme_count']:>7}  {h_names}")

print("\n[3] State=QLD only — should show Brisbane-specific events:")
r = requests.get(f'{BASE}/sales-detail?state=QLD', timeout=30)
data = r.json()
print(f"    Venues: {data['summary']['venues_in_filter']}")
print(f"\n    {'Month':<5}  {'FY26 Hol':>8} {'FY27 Hol':>8} {'FY26 Wx':>7} {'FY27 Wx':>7}  Sample holiday")
for m in data['monthly']:
    h_names = m['fy26_holidays'][:2] if m['fy26_holidays'] else m['fy27_holidays'][:2]
    h_str = ', '.join(h_names) if h_names else '—'
    print(f"    {m['month_label']:<5}  {m['fy26_holiday_count']:>8} {m['fy27_holiday_count']:>8} {m['fy26_extreme_count']:>7} {m['fy27_extreme_count']:>7}  {h_str}")

print("\n[4] State=VIC only — should show VIC-specific events (Melb Cup, Grand Final):")
r = requests.get(f'{BASE}/sales-detail?state=VIC', timeout=30)
data = r.json()
# Look for Melbourne Cup (Nov) and Grand Final Day (Sep)
nov = next((m for m in data['monthly'] if m['month_label'] == 'Nov'), None)
sep = next((m for m in data['monthly'] if m['month_label'] == 'Sep'), None)
if nov:
    print(f"    Nov — FY26: {nov['fy26_holidays']}")
    print(f"    Nov — FY27: {nov['fy27_holidays']}")
if sep:
    print(f"    Sep — FY26: {sep['fy26_holidays']}")
    print(f"    Sep — FY27: {sep['fy27_holidays']}")

print("\n[5] Extreme weather detail (FY26 sample):")
r = requests.get(f'{BASE}/sales-detail', timeout=30)
data = r.json()
for m in data['monthly']:
    if m['fy26_extreme_count'] > 0:
        print(f"    {m['month_label']} FY26: {m['fy26_extreme_count']} extreme days")
        for ev in m['fy26_extreme_detail'][:3]:
            print(f"        {ev['date']} — {ev['detail']}")
        break
for m in data['monthly']:
    if m['fy27_extreme_count'] > 0:
        print(f"    {m['month_label']} FY27: {m['fy27_extreme_count']} extreme days")
        for ev in m['fy27_extreme_detail'][:3]:
            print(f"        {ev['date']} — {ev['detail']}")
        break

print("\nDONE")
