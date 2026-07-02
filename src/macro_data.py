import yfinance as yf
import pandas as pd
from pathlib import Path
import os
import time

MACRO_CSV_PATH = Path("/Users/macos/Documents/ALGO/03_Data/raw/macro_1h.csv")

def download_historical_macro_1h():
    """
    Downloads 1-hour data for VIX, DXY, and US10Y for the max allowed period (730 days).
    Localizes timezones to UTC for perfect alignment with MT5 broker time.
    """
    tickers = ["^VIX", "DX-Y.NYB", "^TNX"]
    
    print("Fetching 1H Macro Data from yfinance (this may take a moment)...")
    try:
        data = yf.download(tickers, period="730d", interval="1h", progress=False)
        
        if data.empty:
            print("ERROR: Failed to download yfinance data.")
            return None
            
        # Extract only the Close prices
        close_df = data['Close'].copy()
        
        # yfinance returns timezone-aware datetimes (America/New_York)
        # We must convert this to UTC to align with our MT5 CSV (which is UTC time)
        close_df.index = close_df.index.tz_convert('UTC').tz_localize(None)
        
        # We also need to map the tickers to their column names explicitly 
        # because the order of columns in yfinance might vary
        cols = []
        for col in close_df.columns:
            if col == "DX-Y.NYB": cols.append("DXY_Close")
            elif col == "^TNX": cols.append("US10Y_Close")
            elif col == "^VIX": cols.append("VIX_Close")
            else: cols.append(col)
        close_df.columns = cols
        
        # Forward fill any NaNs (e.g. if VIX opens later than DXY)
        close_df = close_df.ffill()
        
        # Save to CSV for caching
        MACRO_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        close_df.to_csv(MACRO_CSV_PATH)
        print(f"Successfully cached {len(close_df)} hourly macro bars to {MACRO_CSV_PATH}")
        
        return close_df
    except Exception as e:
        print(f"Error fetching macro data: {e}")
        return None

def load_cached_macro_1h():
    """Loads the cached macro data, or downloads it if it doesn't exist/is too old."""
    if not MACRO_CSV_PATH.exists():
        return download_historical_macro_1h()
    
    # If the file is older than 24 hours, re-download
    file_age_seconds = time.time() - os.path.getmtime(MACRO_CSV_PATH)
    if file_age_seconds > 86400:
        return download_historical_macro_1h()
        
    df = pd.read_csv(MACRO_CSV_PATH, index_col=0, parse_dates=True)
    return df

def get_live_macro_snapshot():
    """Fetches the latest live values for the dashboard (very fast)."""
    try:
        # We fetch 1d period, 1m interval just to get the absolute latest tick
        data = yf.download(["^VIX", "DX-Y.NYB", "^TNX"], period="1d", interval="1m", progress=False)
        if data.empty:
            return None
            
        close_df = data['Close'].ffill().iloc[-1]
        
        return {
            "DXY": close_df["DX-Y.NYB"],
            "US10Y": close_df["^TNX"],
            "VIX": close_df["^VIX"]
        }
    except Exception:
        return None

if __name__ == "__main__":
    df = download_historical_macro_1h()
    if df is not None:
        print(df.tail())
