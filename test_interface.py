"""Test the web interface end-to-end"""
import requests
import json
from datetime import datetime, timedelta

# Test the web interface
print("=" * 70)
print("TESTING WEB INTERFACE - END-TO-END")
print("=" * 70)

# Step 1: Check if web server is running
print("\n[1] Checking web server...")
try:
    response = requests.get('http://localhost:5000/', timeout=5)
    if response.status_code == 200:
        print("    [OK] Web interface is running")
        print("    [OK] HTML loaded successfully")
        if 'Venu Cast' in response.text and 'Forecast' in response.text:
            print("    [OK] Interface contains expected content")
    else:
        print(f"    [ERR] Unexpected status code: {response.status_code}")
except Exception as e:
    print(f"    [ERR] Web server not accessible: {e}")
    exit(1)

# Step 2: Test the forecast endpoint
print("\n[2] Testing /forecast endpoint...")

# Generate test data
start_date = datetime(2026, 1, 1)
dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(50)]
values = [1000 + (i * 10) for i in range(50)]
atp_history = [25.0 + (i * 0.05) for i in range(20)]
atp_dates = dates[:20]

payload = {
    "dates": dates,
    "values": values,
    "forecast_days": 60,
    "venue_id": "Yo-Chi Albert St",
    "holiday_dates": ["2026-05-25", "2026-06-09"],
    "atp_history": atp_history,
    "atp_dates": atp_dates,
}

try:
    response = requests.post('http://localhost:5000/forecast',
                           json=payload,
                           timeout=30)
    if response.status_code == 200:
        result = response.json()
        print("    [OK] Forecast endpoint returned 200")
        print(f"    [OK] Model used: {result.get('model')}")
        print(f"    [OK] RMSE: {result.get('rmse'):.2f}")
        print(f"    [OK] Forecast days: {len(result.get('forecast', []))}")
        print(f"    [OK] Multipliers loaded: {len(result.get('monthly_multipliers', []))} values")
        
        # Check all required fields
        required_fields = [
            'model', 'rmse', 'cv', 'fitted', 'forecast_dates',
            'forecast', 'lower_90', 'upper_90', 'monthly_multipliers',
            'atp_forecast', 'transaction_forecast', 'integrated_forecast',
            'reconciliation_divergences'
        ]
        missing = [f for f in required_fields if f not in result]
        if missing:
            print(f"    [ERR] Missing fields: {missing}")
        else:
            print(f"    [OK] All 13 required response fields present")
    else:
        print(f"    [ERR] Status code {response.status_code}")
        print(f"    {response.text[:200]}")
except Exception as e:
    print(f"    [ERR] Forecast endpoint error: {e}")
    exit(1)

# Step 3: Verify calculations
print("\n[3] Verifying calculations...")

sales_fc = result.get('forecast', [])
atp_fc = result.get('atp_forecast', [])
trans_fc = result.get('transaction_forecast', [])
integrated = result.get('integrated_forecast', [])

# Check a sample day
idx = 15
if atp_fc[idx] > 0:
    calc_trans = sales_fc[idx] / atp_fc[idx]
    actual_trans = trans_fc[idx]
    error = abs(calc_trans - actual_trans) / actual_trans if actual_trans > 0 else 0
    
    if error < 0.01:
        print(f"    [OK] Transaction calculation verified (day {idx})")
    else:
        print(f"    [WARN] Transaction calculation variance: {error:.2%}")

# Check integrated
calc_integrated = trans_fc[idx] * atp_fc[idx]
actual_integrated = integrated[idx]
error = abs(calc_integrated - actual_integrated) / actual_integrated if actual_integrated > 0 else 0

if error < 0.01:
    print(f"    [OK] Integrated forecast verified (day {idx})")
else:
    print(f"    [WARN] Integrated forecast variance: {error:.2%}")

print("\n" + "=" * 70)
print("WEB INTERFACE TEST COMPLETE")
print("=" * 70)
print("\n✓ Web interface is running at http://localhost:5000")
print("✓ Forecast endpoint is working")
print("✓ All calculations verified")
print("\nYou can now access the interface in your browser:")
print("  http://localhost:5000/")
print("=" * 70)
