import pandas as pd
import numpy as np
import os

PREDICTIONS_FILE = "/Users/macos/Documents/ALGO/04_Models/combined_core/oos_predictions.csv"
SLIPPAGE = 0.0001
RISK_MULTIPLIER = 1.0  # 1:1 R/R
MIN_PIPS_PCT = 0.0005  # minimum target distance to avoid intra-bar suicide

def compute_sharpe(returns: np.ndarray, rf: float = 0.0) -> float:
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return (np.mean(returns) - rf) / np.std(returns) * np.sqrt(1512)

def simulate_exits(df: pd.DataFrame, threshold: float):
    # Determine which bars triggered a trade
    mask = np.abs(df['Integrated_Score']) > threshold
    
    # We create a full array of all bars to compute the honest Sharpe
    pnls = np.zeros(len(df))
    trades_taken = 0
    wins = 0
    
    for i in range(len(df)):
        if not mask.iloc[i]:
            continue
            
        row = df.iloc[i]
        entry = row['Entry_Price']
        vol = row['EWMA_Vol']
        z = row['Clipped_Z']
        
        direction = 1 if z > 0 else -1
        
        # Restore Z-Score magnitude targeting!
        tp_dist = abs(z) * vol
        
        # Min Pips filter (don't take trades where targets are inside the spread noise)
        if tp_dist < MIN_PIPS_PCT:
            continue
            
        sl_dist = tp_dist * RISK_MULTIPLIER # Symmetrical risk
        
        tp_level = entry * (1 + (tp_dist * direction))
        sl_level = entry * (1 - (sl_dist * direction))
        
        high = row['Next_High']
        low = row['Next_Low']
        next_close = row['Next_Close']
        
        pnl = 0.0
        
        if direction == 1:
            if high >= tp_level and low <= sl_level: pnl = -sl_dist
            elif low <= sl_level: pnl = -sl_dist
            elif high >= tp_level: pnl = tp_dist
            else: pnl = (next_close - entry) / entry
        else:
            if low <= tp_level and high >= sl_level: pnl = -sl_dist
            elif high >= sl_level: pnl = -sl_dist
            elif low <= tp_level: pnl = tp_dist
            else: pnl = (entry - next_close) / entry
                
        pnl -= SLIPPAGE
        
        # GLM Execution Micro-detail: Assign PnL to the EXACT bar the trade exited.
        # Since this is a 1-bar simulation, it always exits on bar i.
        pnls[i] = pnl
        
        trades_taken += 1
        if pnl > 0:
            wins += 1
            
    sharpe = compute_sharpe(pnls)
    win_rate = (wins / trades_taken) if trades_taken > 0 else 0.0
    return sharpe, trades_taken, win_rate

def run():
    print("Loading OOS Predictions...")
    try:
        df = pd.read_csv(PREDICTIONS_FILE)
    except FileNotFoundError:
        print(f"File not found: {PREDICTIONS_FILE}. Please run run_combined_pipeline.py first.")
        return
        
    print(f"Total OOS bars: {len(df)}")
    
    # 1. Pre-registered threshold evaluation (e.g. 0.20)
    pre_reg = 0.20
    print(f"\n--- PRE-REGISTERED OOS RESULT (Threshold: {pre_reg}) ---")
    sharpe, n_trades, wr = simulate_exits(df, pre_reg)
    print(f"OOS Trades: {n_trades}")
    print(f"OOS Win Rate: {wr:.1%}")
    print(f"OOS Sharpe (Honest Annualized): {sharpe:.2f}")
    
    # Per-fold breakdown
    print("\n  Per-Fold Breakdown:")
    for fold in sorted(df['Fold'].unique()):
        fold_df = df[df['Fold'] == fold]
        f_sharpe, f_trades, f_wr = simulate_exits(fold_df, pre_reg)
        print(f"    Fold {int(fold)}: {f_trades:>4} trades | WR {f_wr:>5.1%} | Sharpe {f_sharpe:>5.2f}")
    
    print("\n--- OOS THRESHOLD SWEEP (Diagnostic) ---")
    print("Threshold | Trades | Win Rate | Honest Sharpe")
    print("-" * 50)
    
    for thresh in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        sharpe, n_trades, wr = simulate_exits(df, thresh)
        print(f"{thresh:>9.2f} | {n_trades:>6} | {wr:>7.1%} | {sharpe:>13.2f}")

if __name__ == "__main__":
    run()
