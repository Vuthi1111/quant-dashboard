import pandas as pd
import os
import glob
from pathlib import Path

COT_DIR = "/Users/macos/Documents/ALGO/03_Data/COT"
OUTPUT_FILE = "/Users/macos/Documents/ALGO/03_Data/raw/MACRO/gold_cot.csv"
MARKET_NAME = 'GOLD - COMMODITY EXCHANGE INC.'

def parse_cot_data():
    print(f"Searching for COT data in {COT_DIR}...")
    xls_files = glob.glob(os.path.join(COT_DIR, "*.xls"))
    
    if not xls_files:
        print("No .xls files found!")
        return
        
    print(f"Found {len(xls_files)} files. Parsing...")
    
    dfs = []
    
    for file in xls_files:
        print(f"  Reading {os.path.basename(file)}...")
        try:
            df = pd.read_excel(file)
            # Filter for Gold
            gold_df = df[df['Market_and_Exchange_Names'] == MARKET_NAME].copy()
            if not gold_df.empty:
                dfs.append(gold_df)
        except Exception as e:
            print(f"  Error reading {file}: {e}")
            
    if not dfs:
        print("No Gold data found in any of the files.")
        return
        
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Clean up dates
    final_df['Date'] = pd.to_datetime(final_df['Report_Date_as_MM_DD_YYYY'])
    final_df = final_df.sort_values('Date')
    
    # Calculate Net Managed Money
    final_df['MM_Net_Long'] = final_df['M_Money_Positions_Long_ALL'] - final_df['M_Money_Positions_Short_ALL']
    
    # Calculate Managed Money as a % of Total Open Interest (to normalize)
    final_df['MM_Net_Pct_OI'] = final_df['MM_Net_Long'] / final_df['Open_Interest_All']
    
    # Select only the columns we need
    cols_to_keep = [
        'Date', 
        'Open_Interest_All', 
        'M_Money_Positions_Long_ALL', 
        'M_Money_Positions_Short_ALL', 
        'MM_Net_Long',
        'MM_Net_Pct_OI'
    ]
    
    clean_df = final_df[cols_to_keep].set_index('Date')
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    clean_df.to_csv(OUTPUT_FILE)
    print(f"\nSaved {len(clean_df)} weeks of COT data to {OUTPUT_FILE}")
    print(clean_df.head())
    print(clean_df.tail())

if __name__ == "__main__":
    parse_cot_data()
