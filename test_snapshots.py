"""End-to-end test: save snapshot → load → export standalone HTML."""
import os, time, requests, re
DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'

for _ in range(15):
    try:
        if requests.get(f'{BASE}/health', timeout=3).status_code == 200:
            break
    except Exception:
        time.sleep(2)

print("[1] Upload + pre-fetch + generate small forecast...")
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

print("\n[2] Save snapshot 'FY27 Base Case'...")
r = requests.post(f'{BASE}/snapshot/save',
                  json={'name': 'FY27 Base Case', 'note': 'Initial run, no growth assumptions applied'},
                  timeout=30)
print(f"    Status: {r.status_code}, response: {r.json()}")

print("\n[3] List snapshots:")
r = requests.get(f'{BASE}/snapshot/list', timeout=10)
for s in r.json().get('snapshots', []):
    print(f"    - {s.get('name')} | saved {s.get('saved_at')} | {s.get('venue_count')} venues | {s.get('size_kb')} KB")
    print(f"      Note: {s.get('note')}")

# Save a second snapshot for variety
print("\n[4] Save snapshot 'FY27 Aggressive 4pct'...")
r = requests.post(f'{BASE}/snapshot/save',
                  json={'name': 'FY27 Aggressive 4pct', 'note': 'Multipliers set to 1.04 across the board'},
                  timeout=30)
print(f"    Status: {r.status_code}")

print("\n[5] Reset + reload from 'FY27 Base Case':")
requests.post(f'{BASE}/reset', timeout=10)
r = requests.get(f'{BASE}/sales-detail', timeout=10)
print(f"    After reset, /sales-detail: {r.status_code} (should be 400 — no results)")

r = requests.post(f'{BASE}/snapshot/load',
                  json={'filename': 'FY27 Base Case'},
                  timeout=30)
print(f"    Load status: {r.status_code}")
print(f"    Loaded: {r.json()}")

r = requests.get(f'{BASE}/sales-detail', timeout=10)
data = r.json()
print(f"    After load, /sales-detail: {r.status_code}, FY27=${data['summary']['fy27_forecast']:,.0f}")

print("\n[6] Export standalone HTML for 'FY27 Base Case':")
r = requests.post(f'{BASE}/export/standalone',
                  json={'name': 'FY27 Base Case', 'note': 'For Sarah review'},
                  timeout=600)
print(f"    Status: {r.status_code}, size: {len(r.content):,} bytes")

# Save the file and verify it's openable
out_file = os.path.join(r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Playwright\results', 'test_standalone.html')
with open(out_file, 'wb') as f:
    f.write(r.content)
print(f"    Saved to: {out_file}")

# Verify the file contains expected content
content = r.content.decode('utf-8', errors='ignore')
checks = [
    ('Has Chart.js CDN', 'chart.js' in content.lower()),
    ('Has SNAPSHOT data injected', '"meta"' in content and '"sales_detail_by_filter"' in content),
    ('Has filter options', '"states"' in content and '"clusters"' in content),
    ('Embedded data not empty', '__SNAPSHOT_JSON__' not in content),
    ('Read-only viewer marker', 'READ-ONLY SNAPSHOT' in content),
]
print("\n[7] Validation of generated HTML:")
for label, ok in checks:
    print(f"    {'[OK]' if ok else '[FAIL]'} {label}")

# Count precomputed filter slices
m = re.search(r'"sales_detail_by_filter":\s*\{', content)
if m:
    # Count keys
    start = m.end()
    # Simple counter
    depth = 1
    count = 0
    i = start
    while i < len(content) and depth > 0:
        if content[i] == '{': depth += 1
        elif content[i] == '}': depth -= 1
        elif content[i] == '"' and depth == 1:
            count += 1
            # Skip to end of string
            i += 1
            while i < len(content) and content[i] != '"':
                if content[i] == '\\': i += 2
                else: i += 1
        i += 1
    print(f"    [INFO] ~{count // 2} sales-detail filter slices precomputed")

print("\n[8] Delete snapshot 'FY27 Aggressive 4pct':")
r = requests.post(f'{BASE}/snapshot/delete',
                  json={'filename': 'FY27 Aggressive 4pct'}, timeout=10)
print(f"    Status: {r.status_code}, response: {r.json()}")

print("\n[9] Final list:")
r = requests.get(f'{BASE}/snapshot/list', timeout=10)
print(f"    Remaining: {[s['name'] for s in r.json().get('snapshots', [])]}")

print("\nDONE")
