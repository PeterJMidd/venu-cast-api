"""Verify /sales-detail handles all 4 sales_type values correctly."""
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

venues = ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Barangaroo', 'Yo-Chi Carlton']
requests.post(f'{BASE}/generate', json={'venues': venues}, timeout=900)

print("\n[2] All four sales types — KPI comparison:")
print(f"\n    {'Sales Type':<20} {'FY26 Comparator':>18} {'FY27 Forecast':>18} {'Growth %':>10}")

values_by_type = {}
for stype in ['servings', 'retail', 'pos_discounts', 'all']:
    r = requests.get(f'{BASE}/sales-detail?sales_type={stype}', timeout=30)
    if r.status_code != 200:
        print(f"    {stype:<20} ERROR {r.status_code}")
        continue
    data = r.json()
    s = data['summary']
    values_by_type[stype] = s
    print(f"    {stype:<20} {s['fy26_comparator']:>18,.0f} {s['fy27_forecast']:>18,.0f} {s['total_growth_pct']:>10.1f}%")

# Verify: All = Servings + Retail + POS Discounts (where POS is negative)
print("\n[3] Reconciliation: All Combined = Servings + Retail + POS Discounts")
serv = values_by_type['servings']
retail = values_by_type['retail']
disc = values_by_type['pos_discounts']
allv = values_by_type['all']

calc_fy26 = serv['fy26_comparator'] + retail['fy26_comparator'] + disc['fy26_comparator']
calc_fy27 = serv['fy27_forecast'] + retail['fy27_forecast'] + disc['fy27_forecast']
fy26_err = abs(calc_fy26 - allv['fy26_comparator']) / abs(allv['fy26_comparator']) * 100 if allv['fy26_comparator'] else 0
fy27_err = abs(calc_fy27 - allv['fy27_forecast']) / abs(allv['fy27_forecast']) * 100 if allv['fy27_forecast'] else 0

print(f"\n    FY26 sum check:")
print(f"      Servings:      ${serv['fy26_comparator']:>14,.0f}")
print(f"      Retail:        ${retail['fy26_comparator']:>14,.0f}")
print(f"      POS Discounts: ${disc['fy26_comparator']:>14,.0f}")
print(f"      Calc total:    ${calc_fy26:>14,.0f}")
print(f"      All Combined:  ${allv['fy26_comparator']:>14,.0f}")
print(f"      Delta: {fy26_err:.4f}%  {'[OK]' if fy26_err < 0.01 else '[FAIL]'}")

print(f"\n    FY27 sum check:")
print(f"      Servings:      ${serv['fy27_forecast']:>14,.0f}")
print(f"      Retail:        ${retail['fy27_forecast']:>14,.0f}")
print(f"      POS Discounts: ${disc['fy27_forecast']:>14,.0f}")
print(f"      Calc total:    ${calc_fy27:>14,.0f}")
print(f"      All Combined:  ${allv['fy27_forecast']:>14,.0f}")
print(f"      Delta: {fy27_err:.4f}%  {'[OK]' if fy27_err < 0.01 else '[FAIL]'}")

# Verify POS discounts are negative
print(f"\n[4] POS Discounts sign check:")
print(f"      FY26 POS Discounts: ${disc['fy26_comparator']:,.0f} {'[OK negative]' if disc['fy26_comparator'] < 0 else '[FAIL - should be negative]'}")
print(f"      FY27 POS Discounts: ${disc['fy27_forecast']:,.0f} {'[OK negative]' if disc['fy27_forecast'] < 0 else '[FAIL - should be negative]'}")

# Verify cohort filter still works on different sales types
print(f"\n[5] Cohort=FY26 Existing + sales_type=all (Total should equal LFL):")
r = requests.get(f'{BASE}/sales-detail?cohort=FY26%20Existing&sales_type=all', timeout=30)
s = r.json()['summary']
print(f"    Total Growth: {s['total_growth_pct']:.1f}%   LFL Growth: {s['lfl_growth_pct']:.1f}%")
print(f"    {'[OK]' if abs(s['total_growth_pct'] - s['lfl_growth_pct']) < 0.01 else '[FAIL]'} (should match)")

print("\nDONE")
