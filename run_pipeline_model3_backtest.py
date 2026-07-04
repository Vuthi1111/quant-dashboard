"""
run_pipeline_model3_backtest.py
═══════════════════════════════════════════════════════════════════════════════
ML-Driven Intraday VWAP Scalping Backtest (15M)

Walk-Forward Design:
  - Train on expanding window, predict on next unseen chunk
  - Each bar's prediction comes ONLY from a model trained on strictly prior data
  - Purge gap of 16 bars between train and test to prevent label leakage
  - Predictions are stitched together to form one continuous equity curve

Trade Logic:
  - Entry: When |VWAP Z-Score| > 2.0 AND model probability > threshold
  - Direction: Long if Z < -2 (price below VWAP), Short if Z > +2 (price above)
  - Target: Price returns to VWAP (dynamic, bar-by-bar)
  - Stop: Price hits 3σ deviation from VWAP (momentum loss)
  - Time Stop: 16 bars (4 hours on 15M)
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/macos/Documents/ALGO/projects/volatility_regime_model/src")
from feature_engineering import (
    load_mt5_csv, resample_to_15m, create_vwap_scalp_labels,
    garman_klass, parkinson, rogers_satchell,
    range_ratio, tick_vol_acceleration
)

ARTIFACT_DIR = "/Users/macos/.gemini/antigravity/brain/a79aad02-781a-4a03-8ce4-594372872646"
SPREAD_PIPS = 3.0  # 3 pips spread cost per round trip on Gold


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES (same as refined pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def hurst_rs(series, min_n=8):
    n = len(series)
    if n < min_n * 2:
        return np.nan
    max_k = int(np.log2(n))
    if max_k < 2:
        return np.nan
    ns, rs_vals = [], []
    for k in range(1, max_k + 1):
        chunk_size = n // (2 ** k)
        if chunk_size < min_n:
            break
        rs_list = []
        for start in range(0, n - chunk_size + 1, chunk_size):
            chunk = series[start:start + chunk_size]
            mean_c = np.mean(chunk)
            dev = np.cumsum(chunk - mean_c)
            r = np.max(dev) - np.min(dev)
            s = np.std(chunk, ddof=1)
            if s > 1e-12:
                rs_list.append(r / s)
        if rs_list:
            ns.append(chunk_size)
            rs_vals.append(np.mean(rs_list))
    if len(ns) < 2:
        return np.nan
    log_n = np.log(ns)
    log_rs = np.log(rs_vals)
    slope = np.polyfit(log_n, log_rs, 1)[0]
    return np.clip(slope, 0, 1)


def build_full_intraday_features(df):
    feat = pd.DataFrame(index=df.index)
    
    hours = df.index.hour + df.index.minute / 60.0
    feat['hour_sin'] = pd.Series(np.sin(2 * np.pi * hours / 24.0), index=df.index).shift(1)
    feat['hour_cos'] = pd.Series(np.cos(2 * np.pi * hours / 24.0), index=df.index).shift(1)
    dow = df.index.dayofweek
    feat['dow_sin'] = pd.Series(np.sin(2 * np.pi * dow / 5.0), index=df.index).shift(1)
    feat['dow_cos'] = pd.Series(np.cos(2 * np.pi * dow / 5.0), index=df.index).shift(1)
    
    df_temp = df.copy()
    df_temp['date'] = df_temp.index.date
    df_temp['typ_price'] = (df_temp['High'] + df_temp['Low'] + df_temp['Close']) / 3.0
    df_temp['tv'] = df_temp['typ_price'] * df_temp['Tick_Volume']
    cum_vol = df_temp.groupby('date')['Tick_Volume'].cumsum()
    cum_tv = df_temp.groupby('date')['tv'].cumsum()
    vwap = cum_tv / cum_vol.replace(0, np.nan)
    df_temp['tv2'] = (df_temp['typ_price'] ** 2) * df_temp['Tick_Volume']
    cum_tv2 = df_temp.groupby('date')['tv2'].cumsum()
    vwap_var = (cum_tv2 / cum_vol.replace(0, np.nan)) - (vwap ** 2)
    vwap_std = np.sqrt(vwap_var.clip(lower=1e-9))
    
    feat['vwap'] = vwap.shift(1)
    feat['vwap_std'] = vwap_std.shift(1)
    feat['vwap_zscore'] = ((df['Close'] - vwap) / vwap_std.replace(0, np.nan)).shift(1)
    
    lr = np.log(df['Close'] / df['Close'].shift(1))
    for lag in [1, 2, 4, 8, 16]:
        feat[f'ret_lag{lag}'] = lr.shift(lag)
    
    for w in [4, 8, 16]:
        feat[f'GK_{w}'] = garman_klass(df, w).shift(1)
        feat[f'PK_{w}'] = parkinson(df, w).shift(1)
        feat[f'RS_{w}'] = rogers_satchell(df, w).shift(1)
    
    feat['HV_16'] = lr.rolling(16).std().shift(1)
    feat['HV_96'] = lr.rolling(96).std().shift(1)
    feat['vol_ratio'] = feat['HV_16'] / (feat['HV_96'] + 1e-9)
    
    feat['roc_4'] = (df['Close'] / df['Close'].shift(4) - 1).shift(1)
    feat['roc_16'] = (df['Close'] / df['Close'].shift(16) - 1).shift(1)
    feat['roc_96'] = (df['Close'] / df['Close'].shift(96) - 1).shift(1)
    
    delta = lr.copy()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    feat['rsi_14'] = (100 - 100 / (1 + rs)).shift(1)
    
    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    feat['bb_pos'] = ((df['Close'] - ma20) / (2 * std20 + 1e-9)).shift(1)
    
    feat['range_ratio'] = range_ratio(df).shift(1)
    feat['tickvol_accel'] = tick_vol_acceleration(df).shift(1)
    
    vol_4 = df['Tick_Volume'].rolling(4).sum()
    vol_96_avg = df['Tick_Volume'].rolling(96).mean() * 4
    feat['rtv'] = (vol_4 / vol_96_avg.replace(0, np.nan)).shift(1)
    
    print("    [Features] Computing rolling Hurst (32-bar)...")
    closes = df['Close'].values
    hurst_vals = np.full(len(closes), np.nan)
    for i in range(32, len(closes)):
        hurst_vals[i] = hurst_rs(closes[i-32:i])
    feat['hurst_32'] = pd.Series(hurst_vals, index=df.index).shift(1)
    
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_predict(df_clean, feature_cols, n_folds=5, purge_bars=16):
    """
    Generate out-of-sample predictions for every labeled event using
    expanding-window walk-forward. Returns df_clean with 'ml_prob' column.
    """
    n = len(df_clean)
    fold_size = n // (n_folds + 1)
    
    all_preds = pd.Series(np.nan, index=df_clean.index)
    fold_aucs = []
    
    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_start = train_end + purge_bars
        test_end = min(test_start + fold_size, n)
        
        if test_end <= test_start:
            break
        
        train = df_clean.iloc[:train_end]
        test = df_clean.iloc[test_start:test_end]
        
        X_train = train[feature_cols].values
        y_train = train['vwap_label'].values
        w_train = train['uniqueness_weight'].values
        
        X_test = test[feature_cols].values
        y_test = test['vwap_label'].values
        
        model = lgb.LGBMClassifier(
            objective='binary', metric='auc', boosting_type='gbdt',
            learning_rate=0.05, num_leaves=16, max_depth=5,
            feature_fraction=0.8, verbose=-1, n_estimators=200
        )
        model.fit(X_train, y_train, sample_weight=w_train)
        
        preds = model.predict_proba(X_test)[:, 1]
        all_preds.iloc[test_start:test_end] = preds
        
        auc = roc_auc_score(y_test, preds)
        fold_aucs.append(auc)
        print(f"    Fold {fold+1}: Train={len(X_train):,} → Test={len(X_test):,} | AUC={auc:.4f}")
    
    df_clean['ml_prob'] = all_preds
    return df_clean, fold_aucs


# ─────────────────────────────────────────────────────────────────────────────
# TRADE SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

def simulate_trades(df_full, df_preds, prob_threshold=0.55, max_horizon=16):
    """
    Simulate trades on the full 15M bar series using ML predictions.
    
    For each prediction event:
      - If ml_prob >= threshold AND |vwap_zscore| > 2: ENTER
      - Direction: Long if z < -2, Short if z > +2
      - Exit: VWAP touch (target), 3σ (stop), or max_horizon (time stop)
    
    Returns a DataFrame of trades with entry/exit/pnl.
    """
    prices = df_full['Close'].values
    vwap_vals = df_full['vwap'].values if 'vwap' in df_full.columns else None
    vwap_std_vals = df_full['vwap_std'].values if 'vwap_std' in df_full.columns else None
    
    # Map prediction index to full index
    full_idx = df_full.index
    pred_times = df_preds[df_preds['ml_prob'].notna()].index
    
    trades = []
    in_trade = False
    trade_exit_bar = 0
    
    for t in pred_times:
        prob = df_preds.loc[t, 'ml_prob']
        if prob < prob_threshold:
            continue
        
        # Find this bar in the full dataframe
        try:
            i = full_idx.get_loc(t)
        except KeyError:
            continue
        
        if i >= len(prices) - max_horizon:
            continue
        
        # Skip if we're still in a previous trade
        if in_trade and i < trade_exit_bar:
            continue
        
        z = df_preds.loc[t, 'vwap_zscore'] if 'vwap_zscore' in df_preds.columns else None
        if z is None or pd.isna(z):
            continue
        
        is_long = z <= -2.0
        entry_price = prices[i]
        
        # Simulate forward
        exit_price = entry_price
        exit_reason = 'time_stop'
        bars_held = max_horizon
        
        for j in range(1, max_horizon + 1):
            idx = i + j
            if idx >= len(prices):
                break
            
            p = prices[idx]
            v = vwap_vals[idx] if vwap_vals is not None and idx < len(vwap_vals) else None
            vs = vwap_std_vals[idx] if vwap_std_vals is not None and idx < len(vwap_std_vals) else None
            
            if v is None or vs is None or pd.isna(v) or pd.isna(vs):
                continue
            
            if is_long:
                if p >= v:  # Target hit
                    exit_price = p
                    exit_reason = 'target'
                    bars_held = j
                    break
                elif p <= v - 3.0 * vs:  # Stop hit
                    exit_price = p
                    exit_reason = 'stop'
                    bars_held = j
                    break
            else:
                if p <= v:  # Target hit
                    exit_price = p
                    exit_reason = 'target'
                    bars_held = j
                    break
                elif p >= v + 3.0 * vs:  # Stop hit
                    exit_price = p
                    exit_reason = 'stop'
                    bars_held = j
                    break
        else:
            exit_price = prices[min(i + max_horizon, len(prices) - 1)]
        
        # PnL
        if is_long:
            pnl_raw = exit_price - entry_price
        else:
            pnl_raw = entry_price - exit_price
        
        pnl_net = pnl_raw - (SPREAD_PIPS * 0.01)  # Spread cost in price terms
        
        trade_exit_bar = i + bars_held
        in_trade = True
        
        trades.append({
            'entry_time': t,
            'exit_time': full_idx[min(i + bars_held, len(full_idx) - 1)],
            'direction': 'LONG' if is_long else 'SHORT',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_raw': pnl_raw,
            'pnl_net': pnl_net,
            'bars_held': bars_held,
            'exit_reason': exit_reason,
            'ml_prob': prob,
            'vwap_zscore': z
        })
    
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(trades_df, thresholds_df, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Model 3: ML-Driven VWAP Scalping Backtest (15M Gold, 2004–2021)', 
                 fontsize=14, fontweight='bold')
    
    # 1. Equity Curve
    ax = axes[0, 0]
    cumulative = trades_df['pnl_net'].cumsum()
    ax.plot(trades_df['entry_time'], cumulative, color='#2196F3', linewidth=1.2)
    ax.fill_between(trades_df['entry_time'], 0, cumulative, alpha=0.15, color='#2196F3')
    ax.set_title('Cumulative PnL (Net of Spread)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative $ per 1 oz')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax = axes[0, 1]
    peak = cumulative.cummax()
    dd = cumulative - peak
    ax.fill_between(trades_df['entry_time'], dd, 0, color='#F44336', alpha=0.4)
    ax.set_title('Drawdown')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown ($)')
    ax.grid(True, alpha=0.3)
    
    # 3. Win Rate by Exit Reason
    ax = axes[1, 0]
    exit_stats = trades_df.groupby('exit_reason').agg(
        count=('pnl_net', 'count'),
        win_rate=('pnl_net', lambda x: (x > 0).mean()),
        avg_pnl=('pnl_net', 'mean')
    )
    colors = ['#4CAF50', '#F44336', '#FF9800']
    bars = ax.bar(exit_stats.index, exit_stats['win_rate'], color=colors[:len(exit_stats)])
    for bar, count in zip(bars, exit_stats['count']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'n={count}', ha='center', fontsize=10)
    ax.set_title('Win Rate by Exit Type')
    ax.set_ylabel('Win Rate')
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Threshold Sensitivity
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(thresholds_df['threshold'], thresholds_df['sharpe'], 
            color='#2196F3', linewidth=2, marker='o', label='Sharpe')
    ax2.plot(thresholds_df['threshold'], thresholds_df['n_trades'], 
             color='#FF9800', linewidth=2, marker='s', linestyle='--', label='# Trades')
    ax.set_title('Threshold Sensitivity')
    ax.set_xlabel('Probability Threshold')
    ax.set_ylabel('Sharpe Ratio', color='#2196F3')
    ax2.set_ylabel('# Trades', color='#FF9800')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = f"{save_dir}/model3_backtest.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  MODEL 3: ML-DRIVEN VWAP SCALPING BACKTEST")
    print("=" * 70)
    
    # ── 1. LOAD & PREPARE ──
    data_path = "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/XAUUSD_M5.csv"
    print(f"\n[1] Loading data...")
    df_raw = load_mt5_csv(data_path)
    df_15m = resample_to_15m(df_raw)
    print(f"    {len(df_15m):,} 15M bars ({df_15m.index[0].date()} → {df_15m.index[-1].date()})")
    
    # ── 2. BUILD FEATURES ──
    print("\n[2] Building features...")
    features_df = build_full_intraday_features(df_15m)
    df_full = pd.concat([df_15m, features_df], axis=1)
    
    # ── 3. CREATE LABELS ──
    print("\n[3] Creating labels...")
    df_labeled = create_vwap_scalp_labels(df_full, max_horizon=16)
    
    # ── 4. CLEAN ──
    meta_cols = [
        "Open", "High", "Low", "Close", "Tick_Volume", "Volume", "Spread",
        "spread", "vwap_label", "fpt_horizon", "uniqueness_weight",
        "date", "typ_price", "tv", "tv2"
    ]
    feature_cols = [c for c in df_labeled.columns if c not in meta_cols]
    
    valid_mask = (~df_labeled[feature_cols].isna().any(axis=1) 
                  & ~df_labeled['vwap_label'].isna())
    df_clean = df_labeled[valid_mask].copy()
    
    print(f"    {len(df_clean):,} valid events | Base Rate: {df_clean['vwap_label'].mean():.2%}")
    
    # ── 5. WALK-FORWARD PREDICTIONS ──
    print("\n[4] Generating Walk-Forward Predictions...")
    df_pred, fold_aucs = walk_forward_predict(df_clean, feature_cols, n_folds=5, purge_bars=16)
    
    mean_auc = np.mean(fold_aucs)
    print(f"\n    Mean WF-AUC: {mean_auc:.4f}")
    
    # ── 6. THRESHOLD SWEEP ──
    print("\n[5] Running Threshold Sweep...")
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    threshold_results = []
    
    for thresh in thresholds:
        trades = simulate_trades(df_full, df_pred, prob_threshold=thresh, max_horizon=16)
        if len(trades) < 10:
            continue
        
        total_pnl = trades['pnl_net'].sum()
        win_rate = (trades['pnl_net'] > 0).mean()
        avg_pnl = trades['pnl_net'].mean()
        std_pnl = trades['pnl_net'].std()
        sharpe = (avg_pnl / std_pnl) * np.sqrt(252 * 4) if std_pnl > 0 else 0  # ~4 trades/day annualized
        max_dd = (trades['pnl_net'].cumsum() - trades['pnl_net'].cumsum().cummax()).min()
        n_targets = (trades['exit_reason'] == 'target').sum()
        n_stops = (trades['exit_reason'] == 'stop').sum()
        n_time = (trades['exit_reason'] == 'time_stop').sum()
        
        threshold_results.append({
            'threshold': thresh,
            'n_trades': len(trades),
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'targets': n_targets,
            'stops': n_stops,
            'time_stops': n_time
        })
        
        print(f"    Thresh={thresh:.2f} | Trades={len(trades):,} | WinRate={win_rate:.2%} | "
              f"Sharpe={sharpe:.2f} | Total PnL=${total_pnl:.2f} | MaxDD=${max_dd:.2f}")
    
    thresholds_df = pd.DataFrame(threshold_results)
    
    # ── 7. OPTIMAL THRESHOLD BACKTEST ──
    if len(thresholds_df) > 0:
        best_row = thresholds_df.loc[thresholds_df['sharpe'].idxmax()]
        best_thresh = best_row['threshold']
        print(f"\n[6] Best Threshold: {best_thresh:.2f} (Sharpe={best_row['sharpe']:.2f})")
        
        trades_best = simulate_trades(df_full, df_pred, prob_threshold=best_thresh, max_horizon=16)
        
        print(f"\n{'='*50}")
        print(f"  FINAL BACKTEST RESULTS (Threshold={best_thresh:.2f})")
        print(f"{'='*50}")
        print(f"  Total Trades      : {len(trades_best):,}")
        print(f"  Win Rate           : {(trades_best['pnl_net'] > 0).mean():.2%}")
        print(f"  Avg PnL/Trade      : ${trades_best['pnl_net'].mean():.4f}")
        print(f"  Total PnL (1 oz)   : ${trades_best['pnl_net'].sum():.2f}")
        print(f"  Sharpe Ratio       : {best_row['sharpe']:.2f}")
        print(f"  Max Drawdown       : ${best_row['max_dd']:.2f}")
        print(f"  Avg Bars Held      : {trades_best['bars_held'].mean():.1f}")
        print(f"  Exit Breakdown:")
        print(f"    Target Hits      : {(trades_best['exit_reason']=='target').sum():,}")
        print(f"    Stop Hits        : {(trades_best['exit_reason']=='stop').sum():,}")
        print(f"    Time Stops       : {(trades_best['exit_reason']=='time_stop').sum():,}")
        
        # Direction breakdown
        longs = trades_best[trades_best['direction'] == 'LONG']
        shorts = trades_best[trades_best['direction'] == 'SHORT']
        print(f"\n  Direction Breakdown:")
        print(f"    LONG  : {len(longs):,} trades | WR={((longs['pnl_net']>0).mean() if len(longs) > 0 else 0):.2%} | Avg=${longs['pnl_net'].mean():.4f}")
        print(f"    SHORT : {len(shorts):,} trades | WR={((shorts['pnl_net']>0).mean() if len(shorts) > 0 else 0):.2%} | Avg=${shorts['pnl_net'].mean():.4f}")
        
        # ── 8. PLOT ──
        print(f"\n[7] Generating plots...")
        plot_results(trades_best, thresholds_df, ARTIFACT_DIR)
    
    print(f"\n{'='*70}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
