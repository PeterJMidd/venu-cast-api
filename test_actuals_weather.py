"""Verify: actuals included, POS discounts negative, rain/wind/extreme in weather."""
import os
import requests
import csv
import io

DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'
BASE = 'http://localhost:5000'

print("=" * 70)
print("TEST: Actuals (Jul 2025-Apr 2026) + Rain/Wind/Cyclone + Neg Discounts")
print("=" * 70)

# 1) Upload
print("\n[1] Uploading templates...")
files = {
    'sales_history': open(os.path.join(DATA_DIR, '01_Sales_History_Template.xlsx'), 'rb'),
    'atp_history': open(os.path.join(DATA_DIR, 'Average ticket history.xlsx'), 'rb'),
    'location_bi': open(os.path.join(DATA_DIR, 'D03_Location_BI.xlsx'), 'rb'),
    'venue_details': open(os.path.join(DATA_DIR, 'Venue Details.xlsx'), 'rb'),
    'multipliers': open(os.path.join(DATA_DIR, 'venue_city_mapping_updated_1.csv'), 'rb'),
}
r = requests.post(f'{BASE}/upload', files=files, timeout=120)
for fh in files.values():
    fh.close()
print(f"    Status: {r.status_code}")

# 2) Pre-fetch weather (with new fields)
print("\n[2] Pre-fetching weather including rain/wind/extreme events...")
r = requests.post(f'{BASE}/data-sources', timeout=300)
print(f"    Status: {r.status_code}, states: {r.json()['states_loaded']}")

# 3) Inspect weather summary - should include rain/gusts/extreme
print("\n[3] Weather data summary (per state)...")
r = requests.get(f'{BASE}/data-sources', timeout=15)
ds = r.json()
for key, info in list(ds['weather'].items())[:5]:
    state = key.split('_')[0]
    print(f"    {state}: {info['days']} days, temp {info['min_temp']}-{info['max_temp']}C avg {info['avg_temp']}C")
    print(f"        Total rain: {info['total_rain_mm']}mm, max gust: {info['max_gust_kmh']}km/h, extreme days: {info['extreme_weather_days']}")

# 4) Run forecast for 2 venues
print("\n[4] Running forecast for 2 venues...")
r = requests.get(f'{BASE}/venues', timeout=15)
vlist = r.json()['venues']
test_venues = [v['venue'] for v in vlist if v.get('multipliers') and v.get('state') in ('VIC', 'QLD')][:2]
print(f"    Test venues: {test_venues}")

r = requests.post(f'{BASE}/generate', json={'venues': test_venues}, timeout=600)
print(f"    Status: {r.status_code}")
result = r.json()
print(f"    Successful: {result['summary']['successful']}/{result['summary']['total_venues']}")
print(f"    Models used: {result['summary']['models_used']}")

# 5) Verify rain/wind/extreme are in result
print("\n[5] Verifying weather data in forecast result...")
for r_v in result['results']:
    if 'servings' in r_v:
        v = r_v['venue']
        wxT = r_v.get('weather_temp_max', [])
        wxP = r_v.get('weather_precip_mm', [])
        wxG = r_v.get('weather_gusts_kmh', [])
        wxE = r_v.get('weather_extreme', [])
        print(f"\n    {v} ({r_v.get('state','?')}):")
        print(f"      Temperature: {len(wxT)} days, range {min([t for t in wxT if t is not None]):.1f}-{max([t for t in wxT if t is not None]):.1f}C")
        precips = [p for p in wxP if p is not None]
        if precips:
            print(f"      Rain: {len(precips)} days, avg {sum(precips)/len(precips):.2f}mm, max {max(precips):.1f}mm")
        gusts = [g for g in wxG if g is not None]
        if gusts:
            print(f"      Wind gusts: avg {sum(gusts)/len(gusts):.1f}km/h, max {max(gusts):.1f}km/h")
        extreme_count = sum(wxE)
        print(f"      Extreme-weather (cyclone-like) days in forecast: {extreme_count}")

        # Verify POS discounts are negative
        if 'discounts' in r_v:
            disc = r_v['discounts']['sales']
            n_neg = sum(1 for d in disc if d < 0)
            print(f"      POS discounts: {n_neg}/{len(disc)} are negative (expected: all)")

# 6) Download CSV and verify
print("\n[6] Downloading CSV...")
r = requests.get(f'{BASE}/download-csv', timeout=60)
print(f"    Status: {r.status_code}, size: {len(r.content):,} bytes")

# Parse CSV and verify content
csv_text = r.text
reader = csv.DictReader(io.StringIO(csv_text))
rows = list(reader)
print(f"    Total rows: {len(rows)}")

# Check columns
print(f"    Columns: {reader.fieldnames}")
assert 'fy' in reader.fieldnames, "Missing 'fy' column"
assert 'type' in reader.fieldnames, "Missing 'type' column (actual/forecast)"
assert 'weather_precip_mm' in reader.fieldnames, "Missing rain column"
assert 'weather_gusts_kmh' in reader.fieldnames, "Missing gust column"
assert 'weather_extreme_event' in reader.fieldnames, "Missing extreme column"

# Verify actuals vs forecasts split
actuals = [r for r in rows if r['type'] == 'actual']
forecasts = [r for r in rows if r['type'] == 'forecast']
print(f"\n    Actual rows: {len(actuals)} (Jul 2025 - Apr 2026 history)")
print(f"    Forecast rows: {len(forecasts)} (May 2026 - Jun 2027)")

if actuals:
    print(f"    First actual date: {actuals[0]['date']}")
    print(f"    Last actual date: {actuals[-1]['date']}")
if forecasts:
    print(f"    First forecast date: {forecasts[0]['date']}")
    print(f"    Last forecast date: {forecasts[-1]['date']}")

# Verify FY tagging
fy_set = set(r['fy'] for r in rows if r['fy'])
print(f"\n    Financial years tagged: {sorted(fy_set)}")

# Verify discount sign
discount_actuals = [float(r['pos_discounts']) for r in actuals if r['pos_discounts']]
discount_fcs = [float(r['pos_discounts']) for r in forecasts if r['pos_discounts']]
print(f"\n    Actual POS discount sign check:")
if discount_actuals:
    n_neg = sum(1 for d in discount_actuals if d < 0)
    print(f"      Actuals: {n_neg}/{len(discount_actuals)} negative (sample: {discount_actuals[:3]})")
if discount_fcs:
    n_neg = sum(1 for d in discount_fcs if d < 0)
    print(f"      Forecasts: {n_neg}/{len(discount_fcs)} negative (sample: {discount_fcs[:3]})")

# Show sample row from actuals and forecasts
print(f"\n    Sample ACTUAL row (FY 2026):")
sample = [r for r in actuals if r['date'] >= '2026-01-01']
if sample:
    s = sample[0]
    print(f"      Date: {s['date']}, FY: {s['fy']}, Type: {s['type']}")
    print(f"      Servings: {s['servings_sales']}, Retail: {s['retail_sales']}, Discount: {s['pos_discounts']}")
    print(f"      Weather: temp={s['weather_temp_max']}, rain={s['weather_precip_mm']}mm, gusts={s['weather_gusts_kmh']}km/h, extreme={s['weather_extreme_event']}")

print(f"\n    Sample FORECAST row (FY 2027):")
sample_fc = [r for r in forecasts if r['fy'] == 'FY 2027']
if sample_fc:
    s = sample_fc[0]
    print(f"      Date: {s['date']}, FY: {s['fy']}, Type: {s['type']}")
    print(f"      Servings: {s['servings_sales']}, Retail: {s['retail_sales']}, Discount: {s['pos_discounts']}")
    print(f"      Weather: temp={s['weather_temp_max']}, rain={s['weather_precip_mm']}mm, gusts={s['weather_gusts_kmh']}km/h, extreme={s['weather_extreme_event']}")

# Find an extreme weather day if any
extreme_rows = [r for r in rows if r['weather_extreme_event'] == '1']
print(f"\n    Extreme-weather days in dataset: {len(extreme_rows)}")
if extreme_rows:
    s = extreme_rows[0]
    print(f"      Sample: {s['venue']}, {s['date']}, gusts={s['weather_gusts_kmh']}km/h, rain={s['weather_precip_mm']}mm")

print("\n" + "=" * 70)
print("ALL CHECKS COMPLETE")
print("=" * 70)
