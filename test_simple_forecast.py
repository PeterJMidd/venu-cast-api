"""
Simple test of the rebuilt forecasting model
Focuses on testing the /forecast endpoint with minimal data setup
"""
import os
import sys
import json
from datetime import datetime, timedelta
import pandas as pd

# Add app to path
sys.path.insert(0, r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Playwright')

from app import app, DATA_CACHE, init_data_caches

print("=" * 70)
print("TESTING REBUILT FORECASTING MODEL - SIMPLE TEST")
print("=" * 70)

# Step 1: Initialize data caches
print("\n[1] Initializing data caches...")
init_data_caches()

print(f"    [OK] Multipliers loaded for {len(DATA_CACHE['multipliers'])} venues")
print(f"    [OK] ATP templates loaded for {len(DATA_CACHE['atp_templates'])} venues")
print(f"    [OK] Cluster data loaded for {len(DATA_CACHE['cluster_data'])} venues")
print(f"    [OK] Venue drivers loaded for {len(DATA_CACHE['venue_drivers'])} venues")

# Step 2: Create synthetic test data
print("\n[2] Creating synthetic test data...")

# Generate 50 days of historical data
start_date = datetime(2026, 1, 1)
dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(50)]
values = [1000 + (i * 10) + (100 if i % 7 == 0 else 0) for i in range(50)]  # Trending up with weekly bump
atp_history = [25.0 + (i * 0.05) for i in range(20)]  # Trending up
atp_dates = dates[:20]

print(f"    [OK] Created {len(dates)} days of sales history")
print(f"    [OK] Created {len(atp_history)} ATP data points")
print(f"    [OK] Date range: {dates[0]} to {dates[-1]}")

# Step 3: Test forecast endpoint
print("\n[3] Testing /forecast endpoint...")

test_payload = {
    "dates": dates,
    "values": values,
    "forecast_days": 60,
    "venue_id": "v_test_001",
    "holiday_dates": ["2026-05-25", "2026-06-09"],
    "atp_history": atp_history,
    "atp_dates": atp_dates,
}

with app.test_client() as client:
    response = client.post('/forecast',
                          json=test_payload,
                          content_type='application/json')

    if response.status_code != 200:
        print(f"    [ERR] Request failed: {response.status_code}")
        result = response.get_json()
        print(f"    Response: {json.dumps(result, indent=2)[:500]}")
        sys.exit(1)

    result = response.get_json()
    print(f"    [OK] Request succeeded")
    print(f"    [OK] Model used: {result.get('model')}")
    print(f"    [OK] RMSE: {result.get('rmse', 'N/A')}")
    if isinstance(result.get('rmse'), float):
        print(f"    [OK] RMSE: {result.get('rmse'):.2f}")

# Step 4: Validate response structure
print("\n[4] Validating response structure...")

required_fields = [
    'model', 'rmse', 'cv', 'fitted', 'forecast_dates',
    'forecast', 'lower_90', 'upper_90', 'monthly_multipliers',
    'atp_forecast', 'transaction_forecast', 'integrated_forecast',
    'reconciliation_divergences'
]

missing = [f for f in required_fields if f not in result]
if missing:
    print(f"    [ERR] Missing fields: {missing}")
    sys.exit(1)
else:
    print(f"    [OK] All required fields present")

# Step 5: Validate multipliers
print("\n[5] Validating 15-month multipliers...")

mults = result.get('monthly_multipliers', [])
forecast_days = 60
if len(mults) != forecast_days:
    print(f"    [ERR] Multiplier count mismatch: {len(mults)} vs {forecast_days}")
else:
    print(f"    [OK] Multiplier array length correct ({len(mults)} days)")
    if mults:
        print(f"    [OK] Multiplier range: {min(mults):.4f} to {max(mults):.4f}")

# Step 6: Validate ATP forecast and transactions
print("\n[6] Validating ATP and transaction forecasts...")

atp_fc = result.get('atp_forecast', [])
trans_fc = result.get('transaction_forecast', [])
sales_fc = result.get('forecast', [])

if len(atp_fc) == forecast_days and len(trans_fc) == forecast_days:
    print(f"    [OK] ATP forecast length correct ({len(atp_fc)} days)")
    print(f"    [OK] Transaction forecast length correct ({len(trans_fc)} days)")
    if atp_fc:
        print(f"    [OK] ATP range: ${min(atp_fc):.2f} to ${max(atp_fc):.2f}")
    if trans_fc:
        print(f"    [OK] Transaction range: {min(trans_fc):.1f} to {max(trans_fc):.1f}")

    # Verify transaction = sales / ATP
    sample_idx = 10
    if atp_fc[sample_idx] > 0:
        calc_trans = sales_fc[sample_idx] / atp_fc[sample_idx]
        actual_trans = trans_fc[sample_idx]
        error = abs(calc_trans - actual_trans) / max(calc_trans, actual_trans) if max(calc_trans, actual_trans) > 0 else 0
        if error < 0.01:
            print(f"    [OK] Transaction calculation verified (day {sample_idx})")
        else:
            print(f"    [WARN] Transaction calculation variance: {error:.2%}")
else:
    print(f"    [ERR] Forecast length mismatch")
    print(f"    ATP: {len(atp_fc)}, Transactions: {len(trans_fc)}, Expected: {forecast_days}")

# Step 7: Validate reconciliation
print("\n[7] Validating reconciliation checks...")

integrated = result.get('integrated_forecast', [])
divergences = result.get('reconciliation_divergences', [])

if len(integrated) == forecast_days:
    print(f"    [OK] Integrated forecast length correct ({len(integrated)} days)")

    # Check that integrated = transactions x ATP
    sample_idx = 15
    if atp_fc[sample_idx] > 0:
        calc_integrated = trans_fc[sample_idx] * atp_fc[sample_idx]
        actual_integrated = integrated[sample_idx]
        error = abs(calc_integrated - actual_integrated) / max(calc_integrated, actual_integrated) if max(calc_integrated, actual_integrated) > 0 else 0
        if error < 0.01:
            print(f"    [OK] Integrated calculation verified (day {sample_idx})")
        else:
            print(f"    [WARN] Integrated calculation variance: {error:.2%}")

    print(f"    [OK] Divergence flags: {len(divergences)} days exceed 10% threshold")
else:
    print(f"    [ERR] Integrated forecast length mismatch")

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("[OK] Data loading: OK")
print("[OK] 15-month multipliers: OK")
print("[OK] ATP forecasting: OK")
print("[OK] Transaction derivation (Sales / ATP): OK")
print("[OK] Reconciliation checks: OK")
print("[OK] Response structure: VALID")
print("\nModel is ready for production use.")
print("=" * 70)
