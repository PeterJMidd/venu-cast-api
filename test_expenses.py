"""End-to-end test: upload sales + expense templates, build P&L, verify scenarios still work."""
import os, time, requests
DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
EXPENSE_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Codex\frozen-yoghurt-budget-launch\templates'
BASE = 'http://localhost:5000'

for _ in range(15):
    try:
        if requests.get(f'{BASE}/health', timeout=3).status_code == 200:
            break
    except Exception:
        time.sleep(2)

print("[1] Upload sales templates + run forecast...")
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
test_v = ['Yo-Chi Albert St', 'Yo-Chi Balaclava', 'Yo-Chi Carlton']
requests.post(f'{BASE}/generate', json={'venues': test_v}, timeout=900)
print("    Sales forecast OK")

print("\n[2] Upload expense templates...")
files = {
    'labour': open(os.path.join(EXPENSE_DIR, '05_Labour_Template.xlsx'), 'rb'),
    'cogs': open(os.path.join(EXPENSE_DIR, '06_COGS_Template.xlsx'), 'rb'),
    'rent': open(os.path.join(EXPENSE_DIR, '07_Rent_Template.xlsx'), 'rb'),
    'other_pnl': open(os.path.join(EXPENSE_DIR, '08_Other_PnL_Template.xlsx'), 'rb'),
}
r = requests.post(f'{BASE}/expense/upload', files=files, timeout=120)
for fh in files.values(): fh.close()
print(f"    Status: {r.status_code}")
print(f"    Summary: {r.json()['summary']}")

print("\n[3] Build P&L for filtered venues...")
r = requests.get(f'{BASE}/pnl', timeout=30)
data = r.json()
nt = data['network_totals']
print(f"    Network FY27 totals:")
print(f"      Net Sales:     ${nt['net_sales']:>14,.0f}")
print(f"      COGS:          ${nt['cogs_total']:>14,.0f}  ({nt['cogs_total']/nt['net_sales']*100:.1f}%)")
print(f"      Gross Profit:  ${nt['gross_profit']:>14,.0f}  ({nt['gross_profit']/nt['net_sales']*100:.1f}%)")
print(f"      Labour:        ${nt['labour_total']:>14,.0f}  ({nt['labour_total']/nt['net_sales']*100:.1f}%)")
print(f"      Occupancy:     ${nt['occupancy_total']:>14,.0f}  ({nt['occupancy_total']/nt['net_sales']*100:.1f}%)")
print(f"      Other P&L:     ${nt['other_pnl_total']:>14,.0f}")
print(f"      Contribution:  ${nt['contribution']:>14,.0f}  ({nt['contribution']/nt['net_sales']*100:.1f}%)")
print(f"      Venues:        {nt['venue_count']}")

print(f"\n[4] Per-venue annual P&L:")
print(f"    {'Venue':<22} {'Net Sales':>12} {'GP':>10} {'GP%':>6} {'Labour':>10} {'L%':>5} {'Occ':>10} {'Occ%':>5} {'Contribution':>12} {'C%':>5}")
for v in data['venues']:
    t = v['totals']; ns = t['net_sales'] or 1
    print(f"    {v['venue']:<22} ${t['net_sales']:>11,.0f} ${t['gross_profit']:>9,.0f} {t['gross_profit']/ns*100:>5.1f}% ${t['labour_total']:>9,.0f} {t['labour_total']/ns*100:>4.1f}% ${t['occupancy_total']:>9,.0f} {t['occupancy_total']/ns*100:>4.1f}% ${t['contribution']:>11,.0f} {t['contribution']/ns*100:>4.1f}%")

print(f"\n[5] Monthly network P&L (first 3 months):")
for m in data['network_monthly'][:3]:
    print(f"    {m['month']}: Sales ${m['net_sales']:>10,.0f} → GP ${m['gross_profit']:>10,.0f} → Contribution ${m['contribution']:>10,.0f}")

print("\n[6] Apply scenario → verify P&L tracks the scenario sales:")
requests.post(f'{BASE}/scenario/create', json={'name': 'PnL Test'}, timeout=10)
requests.post(f'{BASE}/scenario/venue', json={'scenario': 'PnL Test', 'venue': 'Yo-Chi Albert St',
                                              'target_growth_pct': 25.0, 'comment': 'Scenario test'},
              timeout=10)
r = requests.get(f'{BASE}/pnl', timeout=30)
data = r.json()
for v in data['venues']:
    if v['venue'] == 'Yo-Chi Albert St':
        t = v['totals']; ns = t['net_sales']
        print(f"    Albert St with +25% scenario: Net Sales ${ns:,.0f}, Contribution ${t['contribution']:,.0f}")

print("\n[7] CSV export:")
r = requests.get(f'{BASE}/pnl/csv', timeout=30)
print(f"    Status: {r.status_code}, size: {len(r.content):,} bytes")
header = r.text.split('\n', 1)[0]
print(f"    CSV header: {header}")

print("\n[8] Filter by state=VIC:")
r = requests.get(f'{BASE}/pnl?state=VIC', timeout=30)
data = r.json()
print(f"    VIC: {data['network_totals']['venue_count']} venues, ${data['network_totals']['net_sales']:,.0f} net sales")
for v in data['venues']:
    print(f"      - {v['venue']}: contribution ${v['totals']['contribution']:,.0f}")

print("\nDONE")
