"""
Load and validate data files from the Forecast directory.
"""

import os
import sys

# Data directory
DATA_DIR = r"C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May"

# Expected files
REQUIRED_FILES = {
    'venue_city_mapping_updated_1.csv': 'Monthly multipliers (15-month cycle)',
    'Venue Details.xlsx': 'New venue drivers, ramp-up schedules, cannibalization',
    'D03_Location_BI.xlsx': 'Cluster memberships (Cat_Cluster, Cat_Cluster_Type)',
    '01_Sales_History_Template.xlsx': 'Sales history (total, retail, discount)',
    'Average ticket history.xlsx': 'ATP historical values',
    'GL_May to April 26.xlsx': 'Growth template (ATP growth overrides)'
}

def check_files():
    """Verify all data files exist."""
    print("\n" + "="*70)
    print("DATA FILE VALIDATION")
    print("="*70)
    print(f"\nChecking directory: {DATA_DIR}\n")

    found_count = 0
    missing = []

    for filename, description in REQUIRED_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"[OK] {filename}")
            print(f"     {description}")
            print(f"     Size: {file_size:,} bytes\n")
            found_count += 1
        else:
            print(f"[MISSING] {filename}\n")
            missing.append(filename)

    print("="*70)
    print(f"RESULT: Found {found_count}/{len(REQUIRED_FILES)} files")
    print("="*70)

    if missing:
        print("\nMissing files:")
        for f in missing:
            print(f"  - {f}")
        return False

    return True

def load_with_openpyxl():
    """Try loading Excel files with openpyxl."""
    print("\n" + "="*70)
    print("EXCEL FILE STRUCTURE CHECK")
    print("="*70)

    excel_files = [
        ('Venue Details.xlsx', ['venue_id', 'parent_venue_id', 'opening_date']),
        ('D03_Location_BI.xlsx', ['venue_id', 'Cat_Cluster', 'Cat_Cluster_Type']),
        ('01_Sales_History_Template.xlsx', ['date', 'venue_id']),
        ('Average ticket history.xlsx', ['venue_id', 'date', 'average_ticket_price']),
    ]

    try:
        import openpyxl
        print("\n[OK] openpyxl available\n")

        for filename, expected_columns in excel_files:
            filepath = os.path.join(DATA_DIR, filename)
            if os.path.exists(filepath):
                try:
                    wb = openpyxl.load_workbook(filepath, read_only=True)
                    print(f"[OK] {filename}")
                    print(f"     Sheets: {wb.sheetnames}")

                    for sheet in wb.sheetnames[:1]:  # Check first sheet
                        ws = wb[sheet]
                        headers = [cell.value for cell in ws[1]]
                        print(f"     Sheet '{sheet}' headers: {headers[:5]}{'...' if len(headers) > 5 else ''}")
                    print()
                    wb.close()
                except Exception as e:
                    print(f"[ERROR] {filename}: {e}\n")
        return True
    except ImportError:
        print("[INFO] openpyxl not available - will be loaded on Render")
        print("       Continuing with file existence check only...\n")
        return True

def summarize():
    """Print summary of validation."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\nAll required data files are present and accessible.")
    print("\nNext steps:")
    print("  1. Deploy enhanced app.py to Render")
    print("  2. Mount data directory in Render container")
    print("  3. Run integration tests on Render")
    print("  4. Test /forecast endpoint with real data")
    print("\nFiles available for app.py to load:")
    print("  - Monthly multipliers (E2)")
    print("  - Venue drivers & ramp-up (E3)")
    print("  - Cluster data (E4)")
    print("  - Sales history components (E5)")
    print("  - ATP history & growth template (E6-7)")

    print("\n" + "="*70)

if __name__ == "__main__":
    if check_files():
        load_with_openpyxl()
        summarize()
    else:
        print("\nERROR: Not all required files found.")
        print("Please ensure all files are in:")
        print(f"  {DATA_DIR}")
        sys.exit(1)
