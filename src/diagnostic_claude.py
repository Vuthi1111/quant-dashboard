import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PREDICTIONS_FILE = "/Users/macos/Documents/ALGO/04_Models/combined_core/oos_predictions.csv"
HORIZON_BARS = 4

def run_diagnostic():
    print("Loading OOS Predictions...")
    try:
        df = pd.read_csv(PREDICTIONS_FILE)
    except FileNotFoundError:
        print(f"File not found: {PREDICTIONS_FILE}. Please run run_combined_pipeline.py first.")
        return

    # To calculate forward realized volatility, we need to know the forward price range.
    # The oos_predictions.csv only has Next_High, Next_Low, Next_Close (which is a 1-bar horizon).
    # Since our breakout simulator might hold for a few bars (e.g. 4), we should load the primary data.
    
    print("Loading Primary Data for True Forward Range...")
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from feature_engineering import load_mt5_csv
    from run_combined_pipeline import resample_to_4h

    # Load and resample
    raw_path = "/Users/macos/Documents/ALGO/03_Data/raw/NAS100/1h_data.csv"
    df_raw = load_mt5_csv(raw_path)
    df_4h = resample_to_4h(df_raw)
    
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    
    # Calculate rolling 4-bar High and Low
    df_4h['Fwd_High'] = df_4h['High'].rolling(HORIZON_BARS).max().shift(-HORIZON_BARS)
    df_4h['Fwd_Low']  = df_4h['Low'].rolling(HORIZON_BARS).min().shift(-HORIZON_BARS)
    
    # Join this back to the predictions
    df = df.join(df_4h[['Fwd_High', 'Fwd_Low']], how='inner')
    
    # Forward Realized Volatility: High - Low of the next 4 bars, expressed as a % of Entry Price
    df['Fwd_Range_Pct'] = (df['Fwd_High'] - df['Fwd_Low']) / df['Entry_Price']
    
    # Bin by Regime Probability
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['Very Low (0.0-0.2)', 'Low (0.2-0.4)', 'Neutral (0.4-0.6)', 'High (0.6-0.8)', 'Extreme (0.8-1.0)']
    df['Regime_Bucket'] = pd.cut(df['Regime_Prob'], bins=bins, labels=labels, include_lowest=True)
    
    print("\n--- CLAUDE'S DIAGNOSTIC: Forward Realized Volatility by Regime Probability ---")
    summary = df.groupby('Regime_Bucket')['Fwd_Range_Pct'].agg(['count', 'mean', 'std']).reset_index()
    
    # Format the percentages for readability
    summary['mean_pct'] = summary['mean'] * 100
    summary['std_pct'] = summary['std'] * 100
    
    for _, row in summary.iterrows():
        print(f"Bucket: {row['Regime_Bucket']:<20} | Bars: {row['count']:>5} | Avg Fwd Range: {row['mean_pct']:.2f}% | Std Dev: {row['std_pct']:.2f}%")
        
    # Let's also look at tail events (e.g. what percentage of bars in each bucket result in a > 1% move)
    df['Big_Move'] = (df['Fwd_Range_Pct'] > 0.01).astype(int)
    prob_big_move = df.groupby('Regime_Bucket')['Big_Move'].mean() * 100
    
    print("\nProbability of a >1% Range Expansion (next 4 bars):")
    for bucket, prob in prob_big_move.items():
        print(f"Bucket: {bucket:<20} | Prob: {prob:.1f}%")

if __name__ == "__main__":
    run_diagnostic()
