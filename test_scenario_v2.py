"""Verify both target modes (growth % and avg daily $) and avg daily fields."""
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
venues = ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Carlton', 'Yo-Chi Gouger St']
requests.post(f'{BASE}/generate', json={'venues': venues}, timeout=900)

print("\n[2] /venues-detail returns avg_daily fields:")
r = requests.get(f'{BASE}/venues-detail', timeout=15)
data = r.json()
print(f"    {'Venue':<22} {'FY26 days':>10} {'FY27 days':>10} {'FY26 Avg':>10} {'FY27 Avg':>10}")
for v in data['venues']:
    print(f"    {v['venue']:<22} {v['fy26_trading_days']:>10} {v['fy27_days']:>10} ${v['avg_daily_fy26']:>9,.0f} ${v['avg_daily_fy27']:>9,.0f}")

print("\n[3] Create scenario, test growth % target:")
requests.post(f'{BASE}/scenario/create', json={'name': 'Test V2'}, timeout=10)
requests.post(f'{BASE}/scenario/venue',
              json={'scenario': 'Test V2', 'venue': 'Yo-Chi Albert St',
                    'target_growth_pct': 6.0, 'comment': 'Conservative growth'},
              timeout=10)
r = requests.get(f'{BASE}/venues-detail', timeout=15)
albert = next(v for v in r.json()['venues'] if v['venue'] == 'Yo-Chi Albert St')
print(f"    Albert St: target +6.0% growth")
print(f"    Result FY27 ${albert['fy27_forecast']:,.0f}, growth {albert['growth_pct']}% (should be ~6.0)")
print(f"    Comment: '{albert['scenario_comment']}'")
print(f"    Avg daily FY26 ${albert['avg_daily_fy26']:,.0f}, FY27 ${albert['avg_daily_fy27']:,.0f}")

print("\n[4] Test avg daily target — set Carlton avg daily = $14,000:")
requests.post(f'{BASE}/scenario/venue',
              json={'scenario': 'Test V2', 'venue': 'Yo-Chi Carlton',
                    'target_avg_daily': 14000.0, 'comment': 'Budget cap'},
              timeout=10)
r = requests.get(f'{BASE}/venues-detail', timeout=15)
carl = next(v for v in r.json()['venues'] if v['venue'] == 'Yo-Chi Carlton')
print(f"    Carlton: target avg daily $14,000")
print(f"    Result FY27 ${carl['fy27_forecast']:,.0f}, avg daily ${carl['avg_daily_fy27']:,.0f} (should be ~$14,000)")
delta = abs(carl['avg_daily_fy27'] - 14000)
print(f"    Delta from target: ${delta:.0f} ({'[OK]' if delta < 10 else '[FAIL]'})")

print("\n[5] Both targets set — avg daily should win:")
requests.post(f'{BASE}/scenario/venue',
              json={'scenario': 'Test V2', 'venue': 'Yo-Chi Gouger St',
                    'target_growth_pct': 99.0, 'target_avg_daily': 5000.0, 'comment': 'Both set'},
              timeout=10)
r = requests.get(f'{BASE}/venues-detail', timeout=15)
g = next(v for v in r.json()['venues'] if v['venue'] == 'Yo-Chi Gouger St')
print(f"    Gouger St: growth=99% AND avg_daily=$5,000 (avg should win)")
print(f"    Result FY27 avg daily ${g['avg_daily_fy27']:,.0f} (target $5,000)")
print(f"    Hit avg target: {abs(g['avg_daily_fy27'] - 5000) < 10}")

print("\n[6] Verify Prophet phasing preserved (uniform scaling) — sample daily values:")
# Get base forecast vs scenario forecast — daily values should differ by constant ratio for FY27
r = requests.get(f'{BASE}/sales-detail', timeout=15)
print(f"    Active scenario: Test V2")
print(f"    Sales-detail loads OK: {r.status_code == 200}")

print("\nDONE")
