import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import pandas_datareader.data as web
import os

MACRO_TICKERS = ["^VIX", "DX-Y.NYB", "^TNX", "HYG"]
COT_FILE = "/Users/macos/Documents/ALGO/03_Data/raw/MACRO/gold_cot.csv"

def fetch_macro_data(start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
    """
    Fetches daily macro data from Yahoo Finance (DXY, VIX, TNX, GLD) and FRED (TIPS).
    Merges with COT data.
    """
    if end_date is None:
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')
        
    print(f"    [Macro] Fetching VIX, DXY, TNX, GLD, TIPS, and COT...")
    
    # 1. Fetch YFinance Data
    data = yf.download(MACRO_TICKERS, start=start_date, end=end_date, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close_df = data['Close'].copy()
    else:
        close_df = data.copy()
        
    close_df = close_df.rename(columns={
        "^VIX": "macro_vix",
        "DX-Y.NYB": "macro_dxy",
        "^TNX": "macro_tnx",
        "HYG": "macro_hyg"
    })
    
    # 2. Fetch GLD Volume
    gld_data = yf.download("GLD", start=start_date, end=end_date, progress=False)
    if isinstance(gld_data.columns, pd.MultiIndex):
        close_df["gld_volume"] = gld_data['Volume']["GLD"]
    else:
        close_df["gld_volume"] = gld_data['Volume']
        
    # 3. Fetch TIPS (Real Yields) from FRED
    try:
        tips_df = web.DataReader("DFII10", "fred", start_date, end_date)
        close_df["macro_tips"] = tips_df["DFII10"]
    except Exception as e:
        print(f"    [Warning] Could not fetch TIPS from FRED: {e}")
        close_df["macro_tips"] = np.nan
        
    # 4. Load COT Data
    if os.path.exists(COT_FILE):
        cot_df = pd.read_csv(COT_FILE, parse_dates=['Date']).set_index('Date')
        # Reindex to daily to merge with close_df
        cot_daily = cot_df.reindex(close_df.index, method='ffill')
        close_df["cot_mm_net_long"] = cot_daily["MM_Net_Long"]
        close_df["cot_mm_pct_oi"] = cot_daily["MM_Net_Pct_OI"]
    else:
        close_df["cot_mm_net_long"] = np.nan
        close_df["cot_mm_pct_oi"] = np.nan
    
    # Forward fill any missing daily data (holidays, weekly reports)
    close_df = close_df.ffill()
    
    # Compute daily percentage changes
    for col in ["macro_vix", "macro_dxy", "macro_tnx", "macro_tips", "gld_volume"]:
        if col in close_df.columns:
            close_df[f"{col}_pct"] = close_df[col].pct_change()
            
    # Shift by 1 day! (Avoid Lookahead bias)
    close_df = close_df.shift(1).dropna(subset=["macro_dxy", "macro_tips", "cot_mm_pct_oi"])
    
    close_df.index = close_df.index.tz_localize(None)
    return close_df

def merge_macro_features(df_mt5: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the MT5 dataframe (which has a Datetime index) and merges the macro data
    using forward filling.
    """
    # 1. Ensure df_mt5 index is datetime and tz-naive
    if not isinstance(df_mt5.index, pd.DatetimeIndex):
        df_mt5.index = pd.to_datetime(df_mt5.index)
    
    df_mt5.index = df_mt5.index.tz_localize(None)
    
    # 2. Get the min and max dates from the MT5 data to fetch exactly what we need
    start_date = (df_mt5.index.min() - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    end_date = (df_mt5.index.max() + pd.Timedelta(days=2)).strftime('%Y-%m-%d')
    
    # 3. Fetch macro data
    macro_df = fetch_macro_data(start_date, end_date)
    
    # 4. Merge onto MT5 data. Since MT5 is hourly/4H and macro is daily, 
    # we use 'merge_asof' to perform an exact forward-fill alignment.
    # df_mt5 needs to be sorted by index
    df_mt5 = df_mt5.sort_index()
    macro_df = macro_df.sort_index()
    
    # Extract index to columns for merge_asof
    df_mt5_merged = pd.merge_asof(
        df_mt5,
        macro_df,
        left_index=True,
        right_index=True,
        direction='backward'  # Always take the most recent past value
    )
    
    return df_mt5_merged
