# Growth Template Format Examples

**Purpose**: Month-by-month ATP (Average Ticket Price) growth multipliers for Enhancement 6

---

## Template 1: Conservative Growth

**Description**: Steady +0.5% per month = 1.005 multiplier

### CSV Format
```
month,growth_multiplier
1,1.005
2,1.005
3,1.005
4,1.005
5,1.005
6,1.005
7,1.005
8,1.005
9,1.005
10,1.005
11,1.005
12,1.005
```

### Excel Format (as columns)
```
Month | Growth Multiplier
------|------------------
  1   | 1.005
  2   | 1.005
  3   | 1.005
  4   | 1.005
  5   | 1.005
  6   | 1.005
  7   | 1.005
  8   | 1.005
  9   | 1.005
 10   | 1.005
 11   | 1.005
 12   | 1.005
```

**Meaning**: 
- Base ATP: $10.00
- January forecast: $10.00 × 1.005 = **$10.05**
- February forecast: $10.00 × 1.005 = **$10.05**
- (Stable slight growth every month)

---

## Template 2: Aggressive Growth

**Description**: +1-2% per month, ramping up

### CSV Format
```
month,growth_multiplier
1,1.01
2,1.015
3,1.015
4,1.02
5,1.02
6,1.02
7,1.01
8,1.01
9,1.01
10,1.02
11,1.02
12,1.02
```

### Excel Format (as columns)
```
Month | Growth Multiplier
------|------------------
  1   | 1.01
  2   | 1.015
  3   | 1.015
  4   | 1.02
  5   | 1.02
  6   | 1.02
  7   | 1.01
  8   | 1.01
  9   | 1.01
 10   | 1.02
 11   | 1.02
 12   | 1.02
```

**Meaning**:
- Base ATP: $10.00
- January: $10.00 × 1.01 = **$10.10** (+1%)
- February: $10.00 × 1.015 = **$10.15** (+1.5%)
- April-June: $10.00 × 1.02 = **$10.20** (+2%)
- (Higher growth in spring/summer, more moderate in fall)

---

## Template 3: Seasonal Growth (with Q4 Premium)

**Description**: Variable growth with holiday premium in Q4

### CSV Format
```
month,growth_multiplier
1,1.005
2,1.005
3,1.01
4,1.01
5,1.01
6,1.01
7,1.005
8,1.005
9,1.005
10,1.03
11,1.04
12,1.05
```

### Excel Format (as columns)
```
Month | Growth Multiplier
------|------------------
  1   | 1.005  (Post-holiday settle)
  2   | 1.005  (Winter)
  3   | 1.01   (Spring prep)
  4   | 1.01   (Spring)
  5   | 1.01   (Spring)
  6   | 1.01   (Summer start)
  7   | 1.005  (Mid-summer)
  8   | 1.005  (Mid-summer)
  9   | 1.005  (Back-to-school)
 10   | 1.03   (Q4 Holiday begins - 3% growth)
 11   | 1.04   (Black Friday prep - 4% growth)
 12   | 1.05   (Holiday peak - 5% growth)
```

**Meaning**:
- Base ATP: $10.00
- January: $10.00 × 1.005 = **$10.05**
- March: $10.00 × 1.01 = **$10.10**
- October: $10.00 × 1.03 = **$10.30** (holiday premium starts)
- November: $10.00 × 1.04 = **$10.40** (Black Friday)
- December: $10.00 × 1.05 = **$10.50** (holiday peak)

---

## File Format Requirements

### Correct Format ✓
- **CSV**: Month column (1-12) + growth_multiplier column
- **Excel**: Month row/column + multiplier values
- **No headers**: Optional but clear
- **Multiplier range**: 0.95 to 1.20 (typical range)

### Common Format Issues ✗
- ❌ Percentage format (5 instead of 1.05)
- ❌ Wrong month range (0-11 instead of 1-12)
- ❌ Missing months
- ❌ Multiple sheets with conflicting data
- ❌ Multipliers > 1.5 (unrealistic growth)
- ❌ Negative multipliers or multipliers < 0.8 (declining prices)

---

## How to Check Your GL_May to April 26.xlsx

1. **Open the file** in Excel
2. **Check columns/rows**:
   - Do you see months 1-12?
   - Do you see multiplier values in the 1.00-1.20 range?
3. **Verify format**:
   - Are values decimals (1.05) or percentages (5)?
   - Are there multiple conflicting sheets?
4. **Compare to examples above**:
   - Which template type does it match?
   - Are any months missing or have zero/null values?

---

## Integration in app.py

The growth template is used in Enhancement 6:

```python
# Load growth template
atp_growth_overrides = load_growth_template(growth_template_path)

# Apply to ATP forecast
atp_forecast = apply_atp_growth_overrides(
    base_atp_forecast,
    atp_growth_overrides,
    forecast_dates
)
```

**Expected behavior**:
- If month 3 has multiplier 1.01: March ATP = base_atp × 1.01
- If month 12 has multiplier 1.05: December ATP = base_atp × 1.05

---

## Debugging Checklist

If the interface is picking up "wrong templates":

- [ ] Is GL_May to April 26.xlsx actually being loaded?
- [ ] Does it have the correct column headers?
- [ ] Are the multiplier values in the 1.00-1.20 range?
- [ ] Are all 12 months present?
- [ ] Is the file being read as the correct sheet?
- [ ] Are there NULL/empty cells causing fallback?

**If NULL/empty cells exist**, the system should either:
1. Fall back to Conservative template (1.005 for all months)
2. Or error with a message about missing growth template data

---

## Next Step

Compare your GL_May to April 26.xlsx to these three templates above.  
Which one does it most closely resemble?  
Are there differences in:
- The month values (should be 1-12)
- The multiplier values (should be 1.00-1.20)
- The number of sheets or data arrangement

Share what you find and I can help diagnose why it's picking up the "wrong" template.
