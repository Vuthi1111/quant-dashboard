import pandas as pd
import numpy as np
from pathlib import Path
import sys

def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    df_4h = df.resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'TickVolume': 'sum'
    }).dropna()
    return df_4h

def audit_dataset(asset: str, csv_path: str):
    print(f"\n--- Data Scarcity Audit: {asset} ---")
    
    path = Path(csv_path)
    if not path.exists():
        print(f"File not found: {path}")
        return
        
    if "NAS100" in asset:
        df = pd.read_csv(path, sep='\t')
        df.rename(columns={'DateTime': 'Time'}, inplace=True)
    else:
        df = pd.read_csv(path, sep=',')
        df.columns = df.columns.str.capitalize()
        df.rename(columns={'Tick_volume': 'TickVolume'}, inplace=True)
        
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)
        
    total_1h = len(df)
    
    # Resample to 4H
    df_4h = resample_to_4h(df)
    total_4h = len(df_4h)
    
    # Apply EWMA Volatility Warmup
    # To get a stable EWMA(span=20), we typically drop the first ~20 bars.
    warmup_bars = 20
    usable_4h = total_4h - warmup_bars
    
    # Embargo Simulation
    # Assume 5 folds, horizon=1 (4H). Embargo gap = horizon + 1 = 2 bars per fold.
    # Total Embargo loss = 5 folds * 2 = 10 bars (negligible overall, but good to note).
    embargo_loss = 10
    final_trainable_bars = usable_4h - embargo_loss
    
    years = (df_4h.index[-1] - df_4h.index[0]).days / 365.25
    
    print(f"Total 1H Bars:        {total_1h:,}")
    print(f"Total 4H Bars:        {total_4h:,}  (over {years:.1f} years)")
    print(f"Usable 4H Bars:       {usable_4h:,}  (after volatility warmup)")
    print(f"Final Trainable:      {final_trainable_bars:,}  (after embargo purging)")
    
    # Capacity Recommendation
    # Rule of thumb for LightGBM: 
    # Minimum samples per leaf = total_samples / (num_leaves * 20) roughly to avoid overfitting
    # If we want at least 200 samples per leaf on average:
    rec_leaves = max(2, int(final_trainable_bars / 200))
    rec_leaves = min(rec_leaves, 15) # Cap it strictly at 15
    rec_depth = min(5, int(np.log2(rec_leaves)))
    
    print(f"[{asset}] Recommended Max Depth:  {rec_depth}")
    print(f"[{asset}] Recommended Max Leaves: {rec_leaves}")


if __name__ == "__main__":
    nas_path = "/Users/macos/Documents/ALGO/03_Data/raw/NAS100/1h_data.csv"
    gold_path = "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/XAUUSD_1H.csv"
    
    audit_dataset("NAS100", nas_path)
    audit_dataset("GOLD", gold_path)
