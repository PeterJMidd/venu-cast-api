"""Verify /venues-detail returns monthly_fy25/26/27 arrays for the sparkline."""
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
requests.post(f'{BASE}/generate', json={'venues': ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Carlton']}, timeout=900)

print("\n[2] /venues-detail monthly arrays:")
r = requests.get(f'{BASE}/venues-detail', timeout=15)
months = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
for v in r.json()['venues']:
    print(f"\n  {v['venue']} ({v['state']}, {v['cluster']}):")
    print(f"    {'':<8} " + ' '.join(f'{m:>9}' for m in months))
    for label, arr in [('FY25', v.get('monthly_fy25')), ('FY26', v.get('monthly_fy26')), ('FY27', v.get('monthly_fy27'))]:
        if not arr:
            print(f"    {label:<8} (no data)")
            continue
        vals = ' '.join(f"${x/1000:>8,.0f}k" if x else f"{'-':>9}" for x in arr)
        print(f"    {label:<8} {vals}")
    fy25_tot = sum(v.get('monthly_fy25') or [])
    fy26_tot = sum(v.get('monthly_fy26') or [])
    fy27_tot = sum(v.get('monthly_fy27') or [])
    print(f"    Totals:  FY25=${fy25_tot:,.0f}   FY26=${fy26_tot:,.0f}   FY27=${fy27_tot:,.0f}")

print("\nDONE")
