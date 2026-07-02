import pandas as pd
from pathlib import Path
import os

def resample_m5_to_1h(input_path: str, output_path: str):
    print(f"Loading M5 data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Parse datetime
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Resampling {len(df):,} M5 bars to 1H...")
    
    # Resample rules for OHLCV
    resample_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum"
    }
    
    # Using '1h' instead of '1H' as newer pandas versions prefer lowercase
    df_1h = df.resample("1h").agg(resample_rules)
    
    # Drop rows where there was no trading (weekends, etc.)
    df_1h.dropna(inplace=True)
    
    print(f"Saving {len(df_1h):,} 1H bars to {output_path}...")
    df_1h.to_csv(output_path)
    print("Done!")

if __name__ == "__main__":
    INPUT_FILE = "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/XAUUSD_M5.csv"
    OUTPUT_FILE = "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/XAUUSD_1H.csv"
    
    resample_m5_to_1h(INPUT_FILE, OUTPUT_FILE)
