# Venu Cast Enhancements - Quick Reference Guide

**7 Major Improvements to Forecasting Accuracy**

---

## Enhancement 1: Weather × Day-of-Week Interaction

**What it does**: Combines weather patterns with day-of-week behavior to improve predictions.

**Example**:
- Base forecast for Friday: 1,000 units
- Temperature on Friday: 85°F (warm)
- Warm Fridays typically get 15% boost
- **Adjusted forecast: 1,150 units**

**When it helps**:
- Outdoor/weather-dependent venues (patios, seasonal concepts)
- Weather affects customer behavior differently by day of week
- Summer vs winter seasonal patterns

**Data needed**: `weather_map` (date → temperature)

**Function**: `add_weather_dow_interaction(df, dow_encoded)`

---

## Enhancement 2: 15-Month Monthly Forecast Multipliers

**What it does**: Applies repeating monthly patterns (seasonal cycles) to forecasts.

**Example**:
- Base forecast: 1,000 units/day
- March multiplier: 1.2 (historically 20% higher)
- April multiplier: 0.95 (historically 5% lower)
- **March forecast: 1,200 units | April forecast: 950 units**

**When it helps**:
- Seasonal businesses (ice cream shops higher in summer)
- Holiday effects (December typically higher)
- Annual cycles that repeat year after year

**Data source**: `venue_city_mapping_updated_1.csv`

**Cycle**: 15 months repeating (pattern from past 15 months applied forward)

**Functions**: 
- `load_monthly_multipliers(multiplier_df)`
- `get_monthly_multiplier(venue_id, forecast_date, multiplier_map)`

---

## Enhancement 3: New Venue Ramp-Up and Cannibalization

**What it does**: Forecasts new store openings with gradual ramp-up and accounts for lost sales at nearby stores.

**Example**:
- New venue opens in market
- Parent venue (similar location) had 5,000 units/month
- Ramp-up schedule: [20%, 40%, 70%, 100%] over 4 months
- Month 1 forecast: 5,000 × 20% = **1,000 units**
- Month 4 forecast: 5,000 × 100% = **5,000 units**
- **Nearby stores lose ~30% of sales to new venue**

**When it helps**:
- Expanding to new locations
- Understanding cannibalization between nearby stores
- Planning inventory for new store openings

**Data source**: `Venue Details.xlsx` (parent venue, ramp-up schedule, impacted stores)

**Functions**:
- `load_venue_drivers(venue_details_df)`
- `forecast_new_venue(parent_forecast, ramp_up_schedule, opening_date, ...)`
- `apply_cannibalization(all_forecasts, impact_map, forecast_dates)`

---

## Enhancement 4: Cluster-Based Cannibalization Analysis

**What it does**: Groups venues by location cluster and models competitive overlap effects.

**Example**:
- High-density urban cluster: 5 venues competing closely
- Each venue loses ~15% to cluster competition
- Suburban cluster: 2 venues, ~5% loss
- Rural standalone: 0% cluster loss

**Cluster types**:
- High Density Urban (competitive, 15% loss)
- Urban Strip (moderate, 8% loss)
- Suburban (light, 3% loss)
- Rural (none, 0% loss)

**When it helps**:
- Multi-unit operators in competitive markets
- Planning closures or relocations
- Understanding local competition effects

**Data source**: `D03_Location_BI.xlsx` (Cat_Cluster, Cat_Cluster_Type)

**Functions**:
- `load_cluster_data(cluster_df)`
- `analyze_cluster_history(sales_history_dict, cluster_memberships)`
- `forecast_cluster(cluster_venues, base_forecasts, ...)`

---

## Enhancement 5: Multi-Component Sales Forecasting

**What it does**: Forecasts total, retail, and discount sales separately, then reconciles.

**Example**:
- Historical relationship:
  - Total sales: 1,000 units/day
  - Retail: 700 units/day (70%)
  - Discount: 300 units/day (30%)
- **Forecast total: 1,100**
- **Forecast retail: 770** (70% of 1,100)
- **Forecast discount: 330** (30% of 1,100)
- **Reconciliation check**: 770 + 330 = 1,100 ✓

**When it helps**:
- Accounting for promotional vs. regular sales
- Understanding discount strategy impact
- Separate forecasts for margin analysis

**Data source**: `01_Sales_History_Template.xlsx` (3 sheets: Total, Retail, Discount)

**Tolerance**: Components must match within 5% (configurable)

**Functions**:
- `validate_sales_components(total_sales, retail_sales, pos_discounts)`
- `forecast_sales_components(venue_id, dates, total_vals, retail_vals, ...)`

---

## Enhancement 6: Average Ticket Price (ATP) Forecasting

**What it does**: Forecasts price per transaction independently, with optional growth overrides.

**Example**:
- Historical ATP: $8.50/transaction
- Forecast without changes: $8.50 (stable)
- Growth template: +2% in January (price increase)
- **Forecast ATP: $8.67 in January** ($8.50 × 1.02)

**Growth templates** (user-editable):
- Conservative: +0.5% per month
- Aggressive: +1-2% per month
- Seasonal: Higher in Q4 (holiday pricing)

**When it helps**:
- Planned price increases
- Promotional pricing periods
- Premium vs. standard product mix changes

**Data sources**:
- `average_ticket_history.xlsx` (historical ATP)
- `GL_May to April 26.xlsx` (growth template)

**Functions**:
- `forecast_average_ticket_price(dates, atp_values, forecast_days, ...)`
- `apply_atp_growth_overrides(base_forecast, growth_template, ...)`

---

## Enhancement 7: Transaction Volume Forecasting

**What it does**: Derives transaction count from total sales ÷ ATP, with reconciliation checks.

**Example**:
- Forecast total sales: 10,000 units
- Forecast ATP: $10/transaction
- **Forecast transactions: 10,000 ÷ $10 = 1,000 transactions**

**Reconciliation**:
- Direct method: Forecast transactions separately
- Integrated method: total_sales / ATP
- If divergence > 10%: **WARNING** — investigate data quality

**When it helps**:
- Understanding customer count changes
- Staffing planning (transactions = foot traffic)
- Identifying price vs. volume shifts

**Functions**:
- `compute_transaction_forecast(total_sales_forecast, atp_forecast)`
- `reconcile_forecasts(direct_total, integrated_total, threshold=0.10)`

---

## Three-Tier Forecasting Fallback

**How it works**:

1. **Try Prophet** (primary)
   - Seasonal decomposition
   - Regressors (weather, holidays)
   - Confidence intervals (90%)
   
2. **Fall back to SARIMA** (secondary)
   - If Prophet insufficient data or fails
   - Auto ARIMA configuration
   
3. **Fall back to Holt-Winters** (last resort)
   - Exponential smoothing
   - Always works, always available

**Example**:
```
Venue A (2 years history):    Prophet ✓
Venue B (6 months history):   Prophet → SARIMA ✓
Venue C (2 weeks history):    Prophet → SARIMA → Holt-Winters ✓
Venue D (new venue):          Holt-Winters (parent-based) ✓
```

---

## Response Structure

When you call `/forecast` POST, you get:

```json
{
  "model": "prophet",
  "rmse": 45.2,
  "cv": 0.08,
  "fitted": [950, 1020, 1100, ...],
  "forecast_dates": ["2024-06-01", "2024-06-02", ...],
  "forecast": [980, 1050, 1120, ...],
  "lower_90": [890, 950, ...],
  "upper_90": [1070, 1150, ...],
  
  // Enhancement 2 output
  "monthly_multipliers": {
    "2024-06-01": 1.2,
    "2024-06-02": 1.2,
    ...
  },
  
  // Enhancement 5 output
  "components_forecast": {
    "total": [980, 1050, 1120, ...],
    "retail": [686, 735, 784, ...],
    "discount": [294, 315, 336, ...]
  },
  
  // Enhancement 6-7 output
  "atp_forecast": [9.85, 9.87, 9.89, ...],
  "transaction_forecast": [99.5, 106.3, 113.1, ...],
  "integrated_forecast": [9656, 10424, 11118, ...],
  "reconciliation_divergences": [1.2, 0.8, 0.9, ...]
}
```

---

## Quick Request Examples

### Basic Forecast
```bash
POST /forecast
{
  "dates": ["2024-01-01", "2024-01-02", ...],
  "values": [1000, 950, 1100, ...],
  "forecast_days": 30
}
```

### With Weather & Holidays (E1)
```bash
{
  "dates": [...],
  "values": [...],
  "forecast_days": 30,
  "weather_map": {"2024-01-01": 72, "2024-01-02": 68, ...},
  "holiday_dates": ["2024-12-25", "2024-01-01"]
}
```

### With Monthly Multipliers (E2)
```bash
{
  "dates": [...],
  "values": [...],
  "forecast_days": 30,
  "venue_id": "venue_123",
  "monthly_multipliers": {...}  // Loaded from CSV
}
```

### With Multi-Component (E5)
```bash
{
  "dates": [...],
  "values": [...],  // Total
  "retail_values": [...],
  "discount_values": [...],
  "forecast_days": 30
}
```

### With ATP & Transactions (E6-7)
```bash
{
  "dates": [...],
  "values": [...],
  "atp_history": [9.5, 9.6, 9.7, ...],
  "atp_dates": ["2024-01-01", "2024-01-02", ...],
  "atp_growth_overrides": {
    "2024-06": 1.02,  // +2% in June
    "2024-07": 1.03   // +3% in July
  },
  "forecast_days": 30
}
```

---

## Choosing Which Enhancements to Use

| Enhancement | Use When | Skip If |
|---|---|---|
| **E1: Weather×DOW** | Outdoor/weather-dependent venue | Indoor venue, stable demand |
| **E2: Monthly Multipliers** | Clear seasonal patterns | Year-round steady business |
| **E3: New Venue Ramp-up** | Opening new location | Mature, existing venues |
| **E4: Cluster Analysis** | Competing venues nearby | Standalone/unique location |
| **E5: Multi-Component** | Track retail vs. discount separately | Simple total sales only |
| **E6-7: ATP+Transactions** | Price/volume analysis important | Total revenue only |

**Recommendation**: Enable all enhancements — they complement each other and the system automatically falls back if data is missing.

---

## Troubleshooting Enhancements

| Issue | Likely Cause | Fix |
|---|---|---|
| `missing_field_error` | Enhancement parameter missing | Provide required data (e.g., weather_map for E1) |
| `high_divergence_warning` | Component forecasts not reconciling | Check data quality in E5 input |
| `fallback_chain_triggered` | Prophet failed (low data quality) | Check for gaps, outliers in historical data |
| `atp_forecast_null` | ATP history data incomplete | Provide atp_history & atp_dates for E6 |
| `transaction_large_error` | ATP forecast very different from expected | Verify ATP growth template is reasonable |

---

## Performance Impact

Each enhancement adds minimal latency:

- Base forecast: 100ms
- E1 (Weather×DOW): +5ms
- E2 (Monthly multipliers): +2ms
- E3 (New venue): +10ms
- E4 (Cluster): +15ms
- E5 (Components): +20ms
- E6 (ATP): +10ms
- E7 (Transactions): +5ms

**Total with all enhancements**: ~160ms per forecast

---

## Contact & Questions

See app.py docstrings for function-level details.  
See DEPLOYMENT_CHECKLIST.md for deployment instructions.  
See STATUS.md for current implementation status.
