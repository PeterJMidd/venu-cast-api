# Excel Template Guide

The budget model expects 7 Excel files. Use the **Download All Templates** button in the Export tab to get a starter workbook with the correct structure. Below are the column specifications.

Venue names must match exactly across all 7 files (case-insensitive, whitespace-normalised). The **Venue Details** file is the master list — every other file is validated against it.

---

## 1. Sales History — `01_sales_history.xlsx`

**Sheet "Daily Sales"**: rows are venues, columns are dates.

| venue_name | 2024-05-01 | 2024-05-02 | ... | 2026-04-30 |
|------------|------------|------------|-----|------------|
| Bondi      | 2850       | 3120       | ... | 2950       |
| Chapel St  | 1750       | 1820       | ... | 1900       |

- ~24 months of daily gross sales (AUD), one column per date
- Blank cells = venue not trading that day
- Dates can be Excel dates or ISO strings (YYYY-MM-DD)

---

## 2. Prior P&L — `02_prior_pnl.xlsx`

**One sheet per venue**, sheet name = venue name. Rows are P&L line items, columns are months.

| Line Item              | Jul-25 | Aug-25 | ... | Jun-26 | Total  |
|------------------------|--------|--------|-----|--------|--------|
| Net Sales              | 80000  | 75000  | ... | 95000  | ...    |
| COGS - Food            | 20000  | 18750  | ... | 23750  | ...    |
| COGS - Packaging       | 2400   | 2250   | ... | 2850   | ...    |
| COGS - Retail          | 1600   | 1500   | ... | 1900   | ...    |
| COGS - Discounts       | 4000   | 3750   | ... | 4750   | ...    |
| Total COGS             | 28000  | ...    | ... | ...    | ...    |
| Gross Profit           | 52000  | ...    | ... | ...    | ...    |
| Labour - Crew          | ...    | ...    | ... | ...    | ...    |
| Labour - Crew Oncosts  | ...    | ...    | ... | ...    | ...    |
| Labour - Management    | ...    | ...    | ... | ...    | ...    |
| Labour - Mgmt Oncosts  | ...    | ...    | ... | ...    | ...    |
| Total Labour           | ...    | ...    | ... | ...    | ...    |
| Occupancy - Base Rent  | ...    | ...    | ... | ...    | ...    |
| Occupancy - Outgoings  | ...    | ...    | ... | ...    | ...    |
| Occupancy - % Rent     | ...    | ...    | ... | ...    | ...    |
| Occupancy - Marketing Levy | ...| ...    | ... | ...    | ...    |
| Total Occupancy        | ...    | ...    | ... | ...    | ...    |
| Venue Contribution     | ...    | ...    | ... | ...    | ...    |

Used for Budget vs Prior Year variance analysis.

---

## 3. Venue Details — `03_venue_details.xlsx`

**Sheet "Venues"**:

| venue_name | state | opening_date | is_active |
|------------|-------|--------------|-----------|
| Bondi      | NSW   | 2020-03-15   | Y         |
| Chapel St  | VIC   | 2024-08-01   | Y         |
| Surfers    | QLD   | 2026-09-01   | Y         |

- `state` must be one of: NSW, VIC, QLD, SA, WA, TAS, NT, ACT
- `opening_date` is the first day of trading
- `is_active` = N excludes the venue from the forecast

**Sheet "Ramp Up"**: 18-month multiplier applied to new venues.

| venue_name | month_1 | month_2 | month_3 | ... | month_18 |
|------------|---------|---------|---------|-----|----------|
| Bondi      | 0.40    | 0.50    | 0.55    | ... | 1.00     |
| Chapel St  | 0.40    | 0.50    | 0.55    | ... | 1.00     |

Mature venues (open ≥ 18 months) automatically use multiplier = 1.0 regardless of this sheet.

---

## 4. Average Ticket — `04_average_ticket.xlsx`

Rows = venues, columns = 12 budget months.

| venue_name | 2026-07-01 | 2026-08-01 | ... | 2027-06-01 |
|------------|------------|------------|-----|------------|
| Bondi      | 14.50      | 14.50      | ... | 15.00      |
| Chapel St  | 13.75      | 13.75      | ... | 14.25      |

In AUD per transaction. Used to derive forecast transaction count.

---

## 5. Labour — `05_labour.xlsx`

| venue_name | Sales Per Labour Hour | Avg Hourly Rate | Oncosts % | Mgmt Salary Monthly | Mgmt Oncosts % |
|------------|----------------------|-----------------|-----------|---------------------|----------------|
| Bondi      | 120                  | 28.50           | 12        | 6500                | 12             |
| Chapel St  | 110                  | 27.00           | 12        | 6200                | 12             |

- **SPLH** in AUD (e.g. $120 means 1 labour hour per $120 of sales)
- **Hourly Rate** in AUD
- **Oncosts %** as a percentage (12 = 12% — superannuation, payroll tax, workers' comp, leave loading)
- **Mgmt Salary** monthly base for the venue manager (allocated daily across the month)

---

## 6. COGS — `06_cogs.xlsx`

| venue_name | Food % | Packaging % | Retail % | Discount % |
|------------|--------|-------------|----------|------------|
| Bondi      | 25     | 3           | 2        | 5          |
| Chapel St  | 26     | 3           | 2        | 5          |

- **Food %, Packaging %, Retail %** — applied to net_sales
- **Discount %** — applied to gross_sales (since discounts reduce ticket size)

All values entered as percentages (25 = 25%, not 0.25).

---

## 7. Rent — `07_rent.xlsx`

| venue_name | Base Rent Monthly | Outgoings Monthly | % Rent Threshold | % Rent Rate | Marketing Levy % |
|------------|-------------------|-------------------|------------------|-------------|------------------|
| Bondi      | 8000              | 2500              | 50000            | 8           | 2                |
| Chapel St  | 7500              | 2200              | 45000            | 8           | 2                |

- **Base Rent** + **Outgoings** in AUD/month, allocated daily
- **% Rent Threshold** in AUD/month — when monthly sales exceed this, % rent kicks in
- **% Rent Rate** as percentage on excess (8 = 8% of (monthly_sales − threshold))
- **Marketing Levy %** applied daily to net_sales (centre marketing fund contribution)

Set threshold to 0 to disable percentage rent.

---

## Validation Errors

The Upload tab shows warnings for:
- Venues in Sales History / Avg Ticket / Labour / COGS / Rent that don't exist in Venue Details
- Invalid state codes
- Missing opening dates
- Non-numeric sales values
- Date columns in Sales History that aren't parseable

Fix the source spreadsheet and re-upload — only the affected file needs to be re-loaded.
