import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from feature_engineering import build_features, build_vol_regime_labels, load_mt5_csv

DATA_DIR = "/Users/macos/Documents/ALGO/03_Data/raw"
ASSETS = ["NAS100", "GOLD"]

def analyze_asset(asset_name):
    csv_path = os.path.join(DATA_DIR, asset_name, "1h_data.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping {asset_name}, data not found.")
        return
        
    print(f"\n=========================================================")
    print(f"[{asset_name}] Loading 1H Data and Computing ADR...")
    df = load_mt5_csv(csv_path)
    
    # Resample to daily to get true daily High/Low
    df_daily = df.resample('D').agg({'High': 'max', 'Low': 'min'}).dropna()
    df_daily['DailyRange'] = df_daily['High'] - df_daily['Low']
    df_daily['ADR_20'] = df_daily['DailyRange'].rolling(20).mean().shift(1)
    
    # Map back to intraday
    df['Date'] = pd.to_datetime(df.index.date)
    df = df.join(df_daily[['ADR_20']], on='Date')
    df = df.dropna(subset=['ADR_20'])
    
    # Calculate rolling Intraday Range
    df['DayHigh'] = df.groupby('Date')['High'].cummax()
    df['DayLow']  = df.groupby('Date')['Low'].cummin()
    df['IntradayRange'] = df['DayHigh'] - df['DayLow']
    df['ADR_Exhaustion'] = df['IntradayRange'] / df['ADR_20']
    
    print(f"[{asset_name}] Building Regime Model...")
    feat_df = build_features(df)
    label_df = build_vol_regime_labels(df, forward_bars=4, bar_offset=4, regime_pct_high=0.70, regime_pct_low=0.30)
    
    joined = pd.concat([feat_df, label_df, df[['ADR_Exhaustion', 'Close', 'Open']]], axis=1).dropna(subset=['vol_regime'])
    X = joined[feat_df.columns].values.astype(np.float32)
    y = joined['vol_regime'].values.astype(np.int32)
    
    split = int(len(X) * 0.75)
    sc = StandardScaler().fit(X[:split])
    model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, class_weight='balanced', random_state=42, verbose=-1)
    model.fit(sc.transform(X[:split]), y[:split])
    
    probs = model.predict_proba(sc.transform(X))[:, 1]
    joined['regime_prob'] = probs
    
    oos = joined.iloc[split:].copy()
    oos['is_high_vol'] = oos['regime_prob'] > 0.70
    
    # More robust Trend Definition: Are we above or below the 20-period moving average?
    oos['SMA_20'] = oos['Close'].rolling(20).mean()
    oos['Trend_Dir'] = np.where(oos['Close'] > oos['SMA_20'], 1, -1)
    
    # Multiple Horizons
    for h in [1, 4, 8]:
        oos[f'Fwd_{h}H_Ret'] = (oos['Close'].shift(-h) / oos['Close']) - 1
        oos[f'Trend_Cont_{h}H'] = oos[f'Fwd_{h}H_Ret'] * oos['Trend_Dir']
        
    oos = oos.dropna(subset=['Trend_Cont_8H'])
    high_vol = oos[oos['is_high_vol'] == True].copy()
    
    buckets = [0, 0.4, 0.8, 1.2, 5.0]
    labels = ["< 40%", "40% - 80%", "80% - 120%", "> 120%"]
    high_vol.loc[:, 'ADR_Bucket'] = pd.cut(high_vol['ADR_Exhaustion'], bins=buckets, labels=labels)
    
    print(f"\n--- {asset_name} HIGH VOLATILITY REGIME: TREND CONTINUATION BY ADR EXHAUSTION ---")
    print(f"{'ADR Exhaustion':<15} | {'1H Cont (BPS)':<15} | {'4H Cont (BPS)':<15} | {'8H Cont (BPS)':<15} | {'Count'}")
    print("-" * 80)
    
    results_1h = high_vol.groupby('ADR_Bucket')['Trend_Cont_1H'].mean() * 10000
    results_4h = high_vol.groupby('ADR_Bucket')['Trend_Cont_4H'].mean() * 10000
    results_8h = high_vol.groupby('ADR_Bucket')['Trend_Cont_8H'].mean() * 10000
    counts = high_vol.groupby('ADR_Bucket')['Trend_Cont_1H'].count()
    
    for label, r1, r4, r8, c in zip(labels, results_1h, results_4h, results_8h, counts):
        if not np.isnan(r1):
            print(f"{label:<15} | {r1:<15.1f} | {r4:<15.1f} | {r8:<15.1f} | {c}")

def main():
    for asset in ASSETS:
        analyze_asset(asset)

if __name__ == "__main__":
    main()
