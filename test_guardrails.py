"""Verify guardrails behave correctly on extreme + normal venues."""
import os, time, requests
DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'
for _ in range(15):
    try:
        if requests.get(f'{BASE}/health', timeout=3).status_code == 200:
            break
    except Exception:
        time.sleep(2)

print("[1] Upload + pre-fetch + generate (subset incl. problematic venues)...")
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

venues = [
    'Yo-Chi Albert St',      # mature, stable
    'Yo-Chi Balaclava',      # mature, healthy
    'Yo-Chi Cronulla',       # the dipping NSW one
    'Yo-Chi Chippendale',    # opened Dec 2025 — should get massive growth without guardrail
    'Yo-Chi Surry Hills',    # another NSW decliner
    'Yo-Chi Burwood',        # opened Oct 2025 — partial FY26
]
requests.post(f'{BASE}/generate', json={'venues': venues}, timeout=900)

def fetch(guards):
    r = requests.get(f'{BASE}/venues-detail?guardrails={guards}', timeout=20)
    return {v['venue']: v for v in r.json()['venues']}, r.json()

print("\n[2] Compare guardrails OFF vs ON:")
off, off_meta = fetch('off')
on, on_meta = fetch('on')
print(f"    Guardrails ON: {on_meta['guardrails_active']}, venues capped/floored: {on_meta['venues_guardrailed']}")
print()
print(f"    {'Venue':<24} {'FY26':>11} {'OFF FY27':>11} {'OFF %':>7} {'ON FY27':>11} {'ON %':>7}  Guardrail note")
for v in venues:
    o = off.get(v); n = on.get(v)
    if not o or not n: continue
    print(f"    {v:<24} ${o['fy26_forecast']:>9,.0f} ${o['fy27_forecast']:>9,.0f} {o['growth_pct']:>6.1f}% ${n['fy27_forecast']:>9,.0f} {n['growth_pct']:>6.1f}%  {n.get('guardrail_note','')[:80]}")

print("\n[3] Verify guardrails leave scenarios untouched:")
requests.post(f'{BASE}/scenario/create', json={'name': 'Guardrail Test'}, timeout=10)
requests.post(f'{BASE}/scenario/venue', json={'scenario': 'Guardrail Test', 'venue': 'Yo-Chi Chippendale',
                                              'target_growth_pct': 250.0, 'comment': 'Aggressive override'},
              timeout=10)
r = requests.get(f'{BASE}/venues-detail?guardrails=on', timeout=20)
for v in r.json()['venues']:
    if v['venue'] == 'Yo-Chi Chippendale':
        print(f"    Chippendale with scenario +250% (guardrails ON):")
        print(f"      FY27 ${v['fy27_forecast']:,.0f}, growth {v['growth_pct']}%")
        print(f"      Guardrail note: '{v.get('guardrail_note','')}' (should be empty — scenario beats guardrail)")

print("\nDONE")
