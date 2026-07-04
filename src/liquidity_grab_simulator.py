import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import load_mt5_csv
from run_combined_pipeline import resample_to_4h, compute_sharpe

PREDICTIONS_FILE = "/Users/macos/Documents/ALGO/04_Models/combined_core/oos_predictions.csv"
SLIPPAGE_PCT = 0.00015 # 1.5 bps slippage

def run_liquidity_grab_sim():
    print("Loading Data...")
    df_preds = pd.read_csv(PREDICTIONS_FILE)
    df_preds['Time'] = pd.to_datetime(df_preds['Time'])
    df_preds.set_index('Time', inplace=True)
    
    raw_path = "/Users/macos/Documents/ALGO/03_Data/raw/NAS100/1h_data.csv"
    df_raw = load_mt5_csv(raw_path)
    df_4h = resample_to_4h(df_raw)
    
    # Let's align the data
    # df_preds represents the "Signal Bar"
    # We need to look at the next bar (Trigger Bar) and the bar after that (Trade Bar)
    
    df_4h['Signal_High'] = df_4h['High']
    df_4h['Signal_Low'] = df_4h['Low']
    df_4h['Signal_Mid'] = (df_4h['High'] + df_4h['Low']) / 2.0
    
    # Trigger Bar (i+1)
    df_4h['Trigger_Open'] = df_4h['Open'].shift(-1)
    df_4h['Trigger_High'] = df_4h['High'].shift(-1)
    df_4h['Trigger_Low'] = df_4h['Low'].shift(-1)
    df_4h['Trigger_Close'] = df_4h['Close'].shift(-1)
    
    # Trade Bar (i+2)
    df_4h['Trade_High'] = df_4h['High'].shift(-2)
    df_4h['Trade_Low'] = df_4h['Low'].shift(-2)
    df_4h['Trade_Close'] = df_4h['Close'].shift(-2)
    
    df = df_preds.join(df_4h[['Signal_High', 'Signal_Low', 'Signal_Mid', 
                              'Trigger_Open', 'Trigger_High', 'Trigger_Low', 'Trigger_Close',
                              'Trade_High', 'Trade_Low', 'Trade_Close']], how='inner')
    df.dropna(inplace=True)
    
    print("\n--- GLM'S LIQUIDITY GRAB FADE SIMULATOR ---")
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
            
            if row['Regime_Prob'] < thresh:
                continue
                
            sig_h, sig_l, sig_m = row['Signal_High'], row['Signal_Low'], row['Signal_Mid']
            t_o, t_h, t_l, t_c = row['Trigger_Open'], row['Trigger_High'], row['Trigger_Low'], row['Trigger_Close']
            
            is_up_sweep = (t_h > sig_h) and (t_c < t_o) # Broke high, closed red
            is_dn_sweep = (t_l < sig_l) and (t_c > t_o) # Broke low, closed green
            
            if is_up_sweep and is_dn_sweep:
                continue # Too messy, swept both in one bar, pass
                
            entry_price = t_c
            pnl = 0.0
            
            # The trade happens in the Trade Bar
            trade_h = row['Trade_High']
            trade_l = row['Trade_Low']
            trade_c = row['Trade_Close']
            
            if is_up_sweep:
                # Enter SHORT at trigger close
                stop_loss = t_h # Stop above the sweep wick
                target = sig_m # Target signal midpoint
                
                # Assume pessimistic tie-breaker
                if trade_h >= stop_loss and trade_l <= target:
                    pnl = (entry_price - stop_loss) / entry_price
                elif trade_h >= stop_loss:
                    pnl = (entry_price - stop_loss) / entry_price
                elif trade_l <= target:
                    pnl = (entry_price - target) / entry_price
                else:
                    pnl = (entry_price - trade_c) / entry_price
                
                longs_taken = False
                shorts += 1
                
            elif is_dn_sweep:
                # Enter LONG at trigger close
                stop_loss = t_l # Stop below the sweep wick
                target = sig_m
                
                if trade_l <= stop_loss and trade_h >= target:
                    pnl = (stop_loss - entry_price) / entry_price
                elif trade_l <= stop_loss:
                    pnl = (stop_loss - entry_price) / entry_price
                elif trade_h >= target:
                    pnl = (target - entry_price) / entry_price
                else:
                    pnl = (trade_c - entry_price) / entry_price
                    
                longs_taken = True
                longs += 1
                
            else:
                continue # No sweep setup
                
            if pnl != 0.0:
                pnl -= SLIPPAGE_PCT
                pnls[i] = pnl
                trades_taken += 1
                if pnl > 0:
                    wins += 1
                    
        sharpe = compute_sharpe(pnls)
        win_rate = (wins / trades_taken) if trades_taken > 0 else 0.0
        avg_trade = (np.sum(pnls) / trades_taken) if trades_taken > 0 else 0.0
        
        print(f"{thresh:>9.2f} | {trades_taken:>6} | {longs:>5} | {shorts:>6} | {win_rate:>7.1%} | {sharpe:>13.2f} | {avg_trade*100:>10.2f}%")

if __name__ == "__main__":
    run_liquidity_grab_sim()
