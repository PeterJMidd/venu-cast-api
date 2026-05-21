#!/bin/bash
set -e

echo "==> Installing Python packages..."
pip install -r requirements.txt

echo "==> Installing cmdstan..."
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

echo "==> Pre-compiling Prophet Stan model (prevents runtime compilation)..."
python -c "
import pandas as pd
import numpy as np
from prophet import Prophet

# Generate dummy data to force Stan model compilation
dates = pd.date_range('2023-01-01', periods=120, freq='D')
df = pd.DataFrame({'ds': dates, 'y': np.abs(np.random.randn(120)) + 100})
m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
m.fit(df)
print('Prophet Stan model pre-compiled and cached successfully')
"

echo "==> Build complete"
