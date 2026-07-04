import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import load_mt5_csv
from run_combined_pipeline import resample_to_4h, compute_sharpe

PREDICTIONS_FILE = "/Users/macos/Documents/ALGO/04_Models/combined_core/oos_predictions.csv"
HORIZON_BARS = 4
SLIPPAGE_PCT = 0.00015 # 1.5 bps slippage (typical for NAS100 breakouts)

def run_breakout_sim():
    print("Loading Data...")
    df = pd.read_csv(PREDICTIONS_FILE)
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    
    raw_path = "/Users/macos/Documents/ALGO/03_Data/raw/NAS100/1h_data.csv"
    df_raw = load_mt5_csv(raw_path)
    df_4h = resample_to_4h(df_raw)
    
    # We need the High/Low of the signal bar (to set our pending stops)
    df_4h['Signal_High'] = df_4h['High']
    df_4h['Signal_Low'] = df_4h['Low']
    
    # We need the Close of the 4th bar for our time-based exit
    df_4h['Exit_Close'] = df_4h['Close'].shift(-HORIZON_BARS)
    
    # We need to know the max High and min Low over the next 4 bars to see if our stops were hit
    df_4h['Fwd_Max_High'] = df_4h['High'].shift(-1).rolling(HORIZON_BARS).max()
    df_4h['Fwd_Min_Low'] = df_4h['Low'].shift(-1).rolling(HORIZON_BARS).min()
    
    df = df.join(df_4h[['Signal_High', 'Signal_Low', 'Exit_Close', 'Fwd_Max_High', 'Fwd_Min_Low']], how='inner')
    df.dropna(inplace=True)
    
    print("\n--- DEEPSEEK'S REGIME-GATED BREAKOUT SIMULATOR ---")
    print(f"Holding Period: {HORIZON_BARS} Bars (16 Hours)")
    print("Threshold | Trades | Longs | Shorts | Win Rate | Honest Sharpe | Avg Trade %")
    print("-" * 80)
    
    thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    
    for thresh in thresholds:
        pnls = np.zeros(len(df))
        trades_taken = 0
        wins = 0
        longs = 0
        shorts = 0
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # THE GATE: Only trade if ML predicts Volatility Expansion
            if row['Regime_Prob'] < thresh:
                continue
                
            # THE TRIGGER: Place stops at the High and Low of the signal bar
            buy_trigger = row['Signal_High']
            sell_trigger = row['Signal_Low']
            
            fwd_high = row['Fwd_Max_High']
            fwd_low = row['Fwd_Min_Low']
            exit_price = row['Exit_Close']
            
            long_triggered = fwd_high > buy_trigger
            short_triggered = fwd_low < sell_trigger
            
            pnl = 0.0
            
            if long_triggered and short_triggered:
                # Whipsaw: Both stops hit during the 4 bars. 
                # Pessimistic Tie-Breaker: Assume we got stopped out on both for a loss equal to the range.
                # Or just assume the worst-case exit. We will assume we took the worst trade.
                # Actually, a simpler assumption: it's a loss equal to the signal bar's range.
                pnl = - (buy_trigger - sell_trigger) / row['Entry_Price']
                trades_taken += 1
            elif long_triggered:
                pnl = (exit_price - buy_trigger) / buy_trigger
                longs += 1
                trades_taken += 1
            elif short_triggered:
                pnl = (sell_trigger - exit_price) / sell_trigger
                shorts += 1
                trades_taken += 1
                
            if pnl != 0.0:
                pnl -= SLIPPAGE_PCT
                pnls[i] = pnl
                if pnl > 0:
                    wins += 1
                    
        sharpe = compute_sharpe(pnls)
        win_rate = (wins / trades_taken) if trades_taken > 0 else 0.0
        avg_trade = (np.sum(pnls) / trades_taken) if trades_taken > 0 else 0.0
        
        print(f"{thresh:>9.2f} | {trades_taken:>6} | {longs:>5} | {shorts:>6} | {win_rate:>7.1%} | {sharpe:>13.2f} | {avg_trade*100:>10.2f}%")

if __name__ == "__main__":
    run_breakout_sim()
