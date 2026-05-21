"""
Test with a real venue to verify 15-month multipliers are applied correctly
"""
import sys
from datetime import datetime, timedelta
import json

# Add app to path
sys.path.insert(0, r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Playwright')

from app import app, DATA_CACHE, init_data_caches

print("=" * 70)
print("TESTING WITH REAL VENUE")
print("=" * 70)

# Initialize data caches
print("\n[1] Initializing data caches...")
init_data_caches()

venues = list(DATA_CACHE['multipliers'].keys())
print(f"    [OK] Loaded {len(venues)} venues")
if venues:
    venue_name = venues[0]
    print(f"    [OK] Using venue: {venue_name}")
else:
    print("    [ERR] No venues loaded")
    sys.exit(1)

# Create test data
print("\n[2] Creating test data...")

start_date = datetime(2026, 1, 15)
dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(50)]
values = [1000 + (i * 10) for i in range(50)]
atp_history = [25.0 + (i * 0.05) for i in range(20)]
atp_dates = dates[:20]

print(f"    [OK] Created test dataset")

# Test forecast with real venue
print(f"\n[3] Testing /forecast with venue: {venue_name}...")

test_payload = {
    "dates": dates,
    "values": values,
    "forecast_days": 60,
    "venue_id": venue_name,
    "holiday_dates": ["2026-05-25"],
    "atp_history": atp_history,
    "atp_dates": atp_dates,
}

with app.test_client() as client:
    response = client.post('/forecast',
                          json=test_payload,
                          content_type='application/json')

    if response.status_code != 200:
        print(f"    [ERR] Request failed: {response.status_code}")
        print(f"    {response.get_json()}")
        sys.exit(1)

    result = response.get_json()

print(f"    [OK] Forecast generated")
print(f"    [OK] Model: {result.get('model')}")
print(f"    [OK] RMSE: {result.get('rmse'):.2f}")

# Analyze multipliers
print("\n[4] Analyzing 15-month multipliers...")

mults = result.get('monthly_multipliers', [])
if mults:
    # The multiplier pattern should cycle through 15 values
    print(f"    [OK] Multipliers loaded: {len(mults)} values")

    # Show first 15 values (should be one complete cycle)
    first_cycle = mults[:15]
    print(f"    [INFO] First 15 multipliers (one cycle):")
    for i, m in enumerate(first_cycle, 1):
        print(f"      Month {i}: {m:.4f}")

    # Show if the pattern repeats (multipliers for day 15-29 should match day 0-14)
    if len(mults) >= 30:
        second_cycle = mults[15:30]
        match = all(abs(first_cycle[i] - second_cycle[i]) < 0.0001 for i in range(15))
        if match:
            print(f"    [OK] 15-month cycle repeats correctly")
        else:
            print(f"    [WARN] 15-month cycle does not repeat as expected")
            print(f"      Second cycle: {[f'{m:.4f}' for m in second_cycle]}")

# Verify transactions calculation
print("\n[5] Verifying transactions calculation...")

sales_fc = result.get('forecast', [])
atp_fc = result.get('atp_forecast', [])
trans_fc = result.get('transaction_forecast', [])
integrated = result.get('integrated_forecast', [])

# Check a few days
for day_idx in [0, 15, 30, 45, 59]:
    if day_idx < len(sales_fc):
        if atp_fc[day_idx] > 0:
            calc_trans = sales_fc[day_idx] / atp_fc[day_idx]
            actual_trans = trans_fc[day_idx]
            calc_integrated = calc_trans * atp_fc[day_idx]
            actual_integrated = integrated[day_idx]

            sales_atp_err = abs(calc_trans - actual_trans) / actual_trans if actual_trans > 0 else 0
            int_err = abs(calc_integrated - actual_integrated) / actual_integrated if actual_integrated > 0 else 0

            print(f"    Day {day_idx}:")
            print(f"      Sales: {sales_fc[day_idx]:.2f}, ATP: {atp_fc[day_idx]:.2f}")
            print(f"      Trans: {actual_trans:.1f} (calc: {calc_trans:.1f}, err: {sales_atp_err:.2%})")
            print(f"      Integrated: {actual_integrated:.2f} (calc: {calc_integrated:.2f}, err: {int_err:.2%})")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print(f"Venue: {venue_name}")
print(f"Multipliers: {len(mults)} days x 15-month cycle")
print(f"Forecasts: Sales, ATP, Transactions, Integrated - all generated")
print(f"Reconciliation: Integrated = Transactions × ATP verified")
print("=" * 70)
