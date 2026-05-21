"""
Check the actual structure and contents of the GL_May to April 26.xlsx file
to diagnose what 'wrong templates' are being picked up.
"""

import os
import sys

# Try to find the file
possible_locations = [
    r"C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\V_14 May\GL_May to April 26.xlsx",
    r"C:\Users\PeterMiddleton\OneDrive - Yochi\AI\Forecast\GL_May to April 26.xlsx",
    r"C:\Users\PeterMiddleton\Downloads\GL_May to April 26.xlsx",
]

found_file = None
for location in possible_locations:
    if os.path.exists(location):
        found_file = location
        print(f"✓ Found file at: {location}")
        break

if not found_file:
    print("✗ File not found in any expected location:")
    for loc in possible_locations:
        print(f"  - {loc}")
    print("\nSearching entire OneDrive - Yochi folder...")

    # Search recursively
    search_dir = r"C:\Users\PeterMiddleton\OneDrive - Yochi"
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if "GL_May" in file and file.endswith(".xlsx"):
                found_path = os.path.join(root, file)
                print(f"Found: {found_path}")
                file_size = os.path.getsize(found_path)
                print(f"Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    sys.exit(1)

# File found, now examine it
file_size = os.path.getsize(found_file)
print(f"\nFile size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")

try:
    import openpyxl
    print("\n" + "="*70)
    print("CHECKING SHEET STRUCTURE")
    print("="*70)

    wb = openpyxl.load_workbook(found_file, read_only=True)
    print(f"\nSheets found: {wb.sheetnames}")
    print(f"Total sheets: {len(wb.sheetnames)}\n")

    for sheet_name in wb.sheetnames:
        print(f"\n--- Sheet: '{sheet_name}' ---")
        ws = wb[sheet_name]

        # Get dimensions
        max_row = ws.max_row
        max_col = ws.max_column
        print(f"Dimensions: {max_row} rows × {max_col} columns")

        # Get first 15 rows to understand structure
        print(f"\nFirst rows (up to 15):")
        print("-" * 70)

        for i, row in enumerate(ws.iter_rows(min_row=1, max_row=min(15, max_row), values_only=True), 1):
            # Show first 6 columns
            display_row = [str(val)[:15] if val is not None else "NULL" for val in row[:6]]
            print(f"Row {i:2d}: {display_row}")

        if max_row > 15:
            print(f"... ({max_row - 15} more rows)")

        # Check for month columns
        print(f"\nAnalysis:")
        first_row = list(ws.iter_rows(min_row=1, max_row=1, values_only=True))[0]

        has_months = False
        has_multipliers = False

        for cell in first_row:
            if cell and isinstance(cell, (int, str)):
                cell_str = str(cell).lower()
                if "month" in cell_str or "m" in cell_str:
                    has_months = True
                if "multiplier" in cell_str or "growth" in cell_str or "factor" in cell_str:
                    has_multipliers = True

        if has_months:
            print("  ✓ Contains 'Month' column")
        if has_multipliers:
            print("  ✓ Contains growth/multiplier column")

        # Check if values look like multipliers (1.00 - 1.20 range)
        multiplier_count = 0
        for row in ws.iter_rows(min_row=2, max_row=min(50, max_row), values_only=True):
            for cell in row:
                if cell and isinstance(cell, (int, float)):
                    if 0.8 < cell < 1.3:  # Typical multiplier range
                        multiplier_count += 1

        if multiplier_count > 0:
            print(f"  ✓ Found {multiplier_count} values in multiplier range (0.8-1.3)")

        print()

    wb.close()

except ImportError:
    print("\n⚠ openpyxl not installed - cannot read Excel file")
    print("Install with: pip install openpyxl")
except Exception as e:
    print(f"\n✗ Error reading file: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print("""
Expected Growth Template Format:
- Column 1: Month (1-12) or Month header
- Column 2: Growth multiplier (1.005-1.05 typical range)

For Conservative: All multipliers = 1.005
For Aggressive:   Multipliers vary 1.01-1.02
For Seasonal:     Q4 higher (1.03-1.05), rest lower

If the file doesn't match this format, that could be why
the interface is picking up "wrong templates".

Compare the actual structure above to the expected format
in GROWTH_TEMPLATE_EXAMPLES.md (just created).
""")
