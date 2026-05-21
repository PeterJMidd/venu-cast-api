"""Inspect all template files to understand exact structure"""
import os
import pandas as pd

DATA_DIR = r'C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May'

def inspect_excel(path, label):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    try:
        xls = pd.ExcelFile(path)
        print(f"File: {os.path.basename(path)}")
        print(f"Sheets: {xls.sheet_names}")
        for sheet in xls.sheet_names:
            print(f"\n--- Sheet: '{sheet}' ---")
            df = pd.read_excel(path, sheet_name=sheet, nrows=5)
            print(f"Shape: {df.shape}")
            print(f"Columns ({len(df.columns)}):")
            for i, col in enumerate(df.columns[:30]):
                print(f"  [{i}] {repr(col)}")
            if len(df.columns) > 30:
                print(f"  ... and {len(df.columns) - 30} more columns")
            print(f"First 3 rows:")
            print(df.head(3).to_string(max_cols=8))
    except Exception as e:
        print(f"ERROR: {e}")

def inspect_csv(path, label):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    try:
        df = pd.read_csv(path, nrows=5)
        print(f"File: {os.path.basename(path)}")
        print(f"Shape (first 5 rows): {df.shape}")
        full_df = pd.read_csv(path)
        print(f"Full row count: {len(full_df)}")
        print(f"Columns ({len(df.columns)}):")
        for i, col in enumerate(df.columns):
            print(f"  [{i}] {repr(col)}")
        print(f"First 3 rows:")
        print(df.head(3).to_string(max_cols=10))
    except Exception as e:
        print(f"ERROR: {e}")

inspect_excel(os.path.join(DATA_DIR, '01_Sales_History_Template.xlsx'), '01 SALES HISTORY')
inspect_excel(os.path.join(DATA_DIR, 'Average ticket history.xlsx'), 'AVERAGE TICKET HISTORY')
inspect_excel(os.path.join(DATA_DIR, 'D03_Location_BI.xlsx'), 'D03 LOCATION BI')
inspect_excel(os.path.join(DATA_DIR, 'Venue Details.xlsx'), 'VENUE DETAILS')
inspect_csv(os.path.join(DATA_DIR, 'venue_city_mapping_updated_1.csv'), 'VENUE CITY MAPPING (15M MULTIPLIERS)')
