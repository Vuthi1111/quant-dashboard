"""
blind_holdout_test.py
═══════════════════════════════════════════════════════════════════════════════
TRUE BLIND HOLDOUT TEST — Model 3 (VWAP Scalping)

The model was trained ONLY on data from 2004–2021 (XAUUSD_M5.csv).
This script loads fresh M1 data from 2021–2025 (completely unseen),
resamples to 15M, and runs the full pipeline to see if the edge survives.

This is the single most important test in the entire project.
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
SPREAD_PIPS = 3.0


# ── FEATURES (identical to production pipeline) ──

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


# ── DATA LOADERS ──

def load_dat_mt_csv(path):
    """Load DAT_MT format: no header, comma-separated, YYYY.MM.DD,HH:MM,O,H,L,C,V"""
    df = pd.read_csv(path, header=None, 
                     names=['date', 'time', 'Open', 'High', 'Low', 'Close', 'Tick_Volume'])
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M')
    df.set_index('datetime', inplace=True)
    df.drop(columns=['date', 'time'], inplace=True)
    df.sort_index(inplace=True)
    return df


def load_and_combine_fresh_data():
    """Load all fresh M1 files from 2021-2025, combine, and resample to 15M."""
    files = [
        "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/DAT_MT_XAUUSD_M1_2021.csv",
        "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/DAT_MT_XAUUSD_M1_2022.csv",
        "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/DAT_MT_XAUUSD_M1_2023.csv",
        "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/DAT_MT_XAUUSD_M1_2024.csv",
        "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/DAT_MT_XAUUSD_M1_2025.csv",
    ]
    
    dfs = []
    for f in files:
        print(f"    Loading {f.split('/')[-1]}...")
        dfs.append(load_dat_mt_csv(f))
    
    df_m1 = pd.concat(dfs)
    df_m1.sort_index(inplace=True)
    
    # Only use data AFTER Aug 2021 (the training data cutoff)
    cutoff = '2021-08-06'
    df_fresh = df_m1[df_m1.index >= cutoff].copy()
    print(f"    Fresh data after {cutoff}: {len(df_fresh):,} M1 bars")
    print(f"    Date range: {df_fresh.index[0]} → {df_fresh.index[-1]}")
    
    # Resample to 15M
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Tick_Volume': 'sum'}
    df_15m = df_fresh.resample('15Min').agg(agg_dict)
    df_15m.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    
    # Handle zero tick volume (this dataset may have 0s)
    df_15m['Tick_Volume'] = df_15m['Tick_Volume'].replace(0, 1)
    
    print(f"    Resampled to {len(df_15m):,} 15M bars")
    return df_15m


# ── TRADE SIMULATOR ──

def simulate_trades(df_full, probs, vwap_zscores, prob_threshold=0.70, max_horizon=16):
    prices = df_full['Close'].values
    vwap_vals = df_full['vwap'].values if 'vwap' in df_full.columns else None
    vwap_std_vals = df_full['vwap_std'].values if 'vwap_std' in df_full.columns else None
    
    trades = []
    trade_exit_bar = 0
    
    for i in range(len(probs)):
        if np.isnan(probs[i]) or probs[i] < prob_threshold:
            continue
        if np.isnan(vwap_zscores[i]) or abs(vwap_zscores[i]) < 2.0:
            continue
        if i < trade_exit_bar:
            continue
        if i >= len(prices) - max_horizon:
            continue
        
        is_long = vwap_zscores[i] <= -2.0
        entry_price = prices[i]
        
        exit_price = entry_price
        exit_reason = 'time_stop'
        bars_held = max_horizon
        
        for j in range(1, max_horizon + 1):
            idx = i + j
            if idx >= len(prices):
                break
            p = prices[idx]
            v = vwap_vals[idx] if vwap_vals is not None else None
            vs = vwap_std_vals[idx] if vwap_std_vals is not None else None
            if v is None or vs is None or np.isnan(v) or np.isnan(vs):
                continue
            
            if is_long:
                if p >= v:
                    exit_price, exit_reason, bars_held = p, 'target', j
                    break
                elif p <= v - 3.0 * vs:
                    exit_price, exit_reason, bars_held = p, 'stop', j
                    break
            else:
                if p <= v:
                    exit_price, exit_reason, bars_held = p, 'target', j
                    break
                elif p >= v + 3.0 * vs:
                    exit_price, exit_reason, bars_held = p, 'stop', j
                    break
        else:
            exit_price = prices[min(i + max_horizon, len(prices) - 1)]
        
        pnl_raw = (exit_price - entry_price) if is_long else (entry_price - exit_price)
        pnl_net = pnl_raw - (SPREAD_PIPS * 0.01)
        
        trade_exit_bar = i + bars_held
        
        trades.append({
            'entry_time': df_full.index[i],
            'direction': 'LONG' if is_long else 'SHORT',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_raw': pnl_raw,
            'pnl_net': pnl_net,
            'bars_held': bars_held,
            'exit_reason': exit_reason,
            'ml_prob': probs[i],
        })
    
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  BLIND HOLDOUT TEST: Model 3 on Fresh 2021–2025 Data")
    print("=" * 70)
    
    # ── 1. LOAD TRAINING DATA (2004–2021) ──
    print("\n[1] Loading TRAINING data (2004–2021)...")
    df_train_raw = load_mt5_csv("/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/XAUUSD_M5.csv")
    df_train_15m = resample_to_15m(df_train_raw)
    print(f"    Training: {len(df_train_15m):,} bars ({df_train_15m.index[0].date()} → {df_train_15m.index[-1].date()})")
    
    # ── 2. LOAD FRESH DATA (Aug 2021–2025) ──
    print("\n[2] Loading HOLDOUT data (Aug 2021–2025)...")
    df_holdout_15m = load_and_combine_fresh_data()
    
    # ── 3. BUILD FEATURES ON TRAINING DATA ──
    print("\n[3] Building features on TRAINING data...")
    train_feat = build_full_intraday_features(df_train_15m)
    df_train_full = pd.concat([df_train_15m, train_feat], axis=1)
    df_train_labeled = create_vwap_scalp_labels(df_train_full, max_horizon=16)
    
    meta_cols = [
        "Open", "High", "Low", "Close", "Tick_Volume", "Volume", "Spread",
        "spread", "vwap_label", "fpt_horizon", "uniqueness_weight",
        "date", "typ_price", "tv", "tv2"
    ]
    feature_cols = [c for c in df_train_labeled.columns if c not in meta_cols]
    
    valid_train = (~df_train_labeled[feature_cols].isna().any(axis=1) 
                   & ~df_train_labeled['vwap_label'].isna())
    df_train_clean = df_train_labeled[valid_train].copy()
    
    print(f"    Training events: {len(df_train_clean):,}")
    
    # ── 4. TRAIN FINAL MODEL ON ALL TRAINING DATA ──
    print("\n[4] Training LightGBM on ALL training data (2004–2021)...")
    X_train = df_train_clean[feature_cols].values
    y_train = df_train_clean['vwap_label'].values
    w_train = df_train_clean['uniqueness_weight'].values
    
    model = lgb.LGBMClassifier(
        objective='binary', metric='auc', boosting_type='gbdt',
        learning_rate=0.05, num_leaves=16, max_depth=5,
        feature_fraction=0.8, verbose=-1, n_estimators=200
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    print(f"    Model trained on {len(X_train):,} samples.")
    
    # ── 5. BUILD FEATURES ON HOLDOUT DATA ──
    print("\n[5] Building features on HOLDOUT data (Aug 2021–2025)...")
    holdout_feat = build_full_intraday_features(df_holdout_15m)
    df_holdout_full = pd.concat([df_holdout_15m, holdout_feat], axis=1)
    df_holdout_labeled = create_vwap_scalp_labels(df_holdout_full, max_horizon=16)
    
    valid_holdout = (~df_holdout_labeled[feature_cols].isna().any(axis=1) 
                     & ~df_holdout_labeled['vwap_label'].isna())
    df_holdout_clean = df_holdout_labeled[valid_holdout].copy()
    
    print(f"    Holdout events: {len(df_holdout_clean):,}")
    print(f"    Holdout Base Rate: {df_holdout_clean['vwap_label'].mean():.2%}")
    
    # ── 6. PREDICT ON HOLDOUT ──
    print("\n[6] Running inference on HOLDOUT (model has NEVER seen this data)...")
    X_holdout = df_holdout_clean[feature_cols].values
    y_holdout = df_holdout_clean['vwap_label'].values
    
    preds = model.predict_proba(X_holdout)[:, 1]
    
    auc = roc_auc_score(y_holdout, preds)
    
    print(f"\n{'='*60}")
    print(f"  ★ BLIND HOLDOUT AUC: {auc:.4f} ★")
    print(f"{'='*60}")
    
    if auc > 0.65:
        print("  ✅ EDGE SURVIVES — Model generalizes to unseen data!")
    elif auc > 0.55:
        print("  ⚠️  MARGINAL — Some signal but degraded.")
    else:
        print("  ❌ EDGE DOES NOT SURVIVE — Model has overfit to training period.")
    
    # ── 7. SIMULATE TRADES ON HOLDOUT ──
    print(f"\n[7] Simulating trades on holdout (threshold=0.70)...")
    
    # Add predictions back
    df_holdout_clean['ml_prob'] = preds
    
    # Get vwap_zscore values aligned with predictions
    probs_full = np.full(len(df_holdout_full), np.nan)
    zscore_full = df_holdout_full['vwap_zscore'].values if 'vwap_zscore' in df_holdout_full.columns else np.full(len(df_holdout_full), np.nan)
    
    # Map predictions back to full dataframe indices
    for i, idx in enumerate(df_holdout_clean.index):
        loc = df_holdout_full.index.get_loc(idx)
        probs_full[loc] = preds[i]
    
    trades = simulate_trades(df_holdout_full, probs_full, zscore_full, prob_threshold=0.70, max_horizon=16)
    
    if len(trades) > 0:
        total_pnl = trades['pnl_net'].sum()
        win_rate = (trades['pnl_net'] > 0).mean()
        avg_pnl = trades['pnl_net'].mean()
        std_pnl = trades['pnl_net'].std()
        sharpe = (avg_pnl / std_pnl) * np.sqrt(252 * 4) if std_pnl > 0 else 0
        max_dd = (trades['pnl_net'].cumsum() - trades['pnl_net'].cumsum().cummax()).min()
        
        n_targets = (trades['exit_reason'] == 'target').sum()
        n_stops = (trades['exit_reason'] == 'stop').sum()
        n_time = (trades['exit_reason'] == 'time_stop').sum()
        
        longs = trades[trades['direction'] == 'LONG']
        shorts = trades[trades['direction'] == 'SHORT']
        
        print(f"\n{'='*60}")
        print(f"  BLIND HOLDOUT BACKTEST (2021–2025)")
        print(f"{'='*60}")
        print(f"  Total Trades       : {len(trades):,}")
        print(f"  Win Rate           : {win_rate:.2%}")
        print(f"  Avg PnL/Trade      : ${avg_pnl:.4f}")
        print(f"  Total PnL (1 oz)   : ${total_pnl:.2f}")
        print(f"  Sharpe Ratio       : {sharpe:.2f}")
        print(f"  Max Drawdown       : ${max_dd:.2f}")
        print(f"  Avg Bars Held      : {trades['bars_held'].mean():.1f}")
        print(f"  Exit Breakdown:")
        print(f"    Target Hits      : {n_targets:,}")
        print(f"    Stop Hits        : {n_stops:,}")
        print(f"    Time Stops       : {n_time:,}")
        print(f"\n  Direction Breakdown:")
        print(f"    LONG  : {len(longs):,} trades | WR={(longs['pnl_net']>0).mean() if len(longs)>0 else 0:.2%} | Avg=${longs['pnl_net'].mean() if len(longs)>0 else 0:.4f}")
        print(f"    SHORT : {len(shorts):,} trades | WR={(shorts['pnl_net']>0).mean() if len(shorts)>0 else 0:.2%} | Avg=${shorts['pnl_net'].mean() if len(shorts)>0 else 0:.4f}")
        
        # ── 8. PLOT ──
        print(f"\n[8] Generating holdout equity curve...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('BLIND HOLDOUT: Model 3 on Unseen 2021–2025 Gold Data', fontsize=14, fontweight='bold')
        
        ax = axes[0]
        cumulative = trades['pnl_net'].cumsum()
        ax.plot(trades['entry_time'], cumulative, color='#2196F3', linewidth=1.5)
        ax.fill_between(trades['entry_time'], 0, cumulative, alpha=0.15, color='#2196F3')
        ax.set_title(f'Cumulative PnL (Net) — Sharpe={sharpe:.2f}')
        ax.set_ylabel('$ per 1 oz')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        peak = cumulative.cummax()
        dd = cumulative - peak
        ax.fill_between(trades['entry_time'], dd, 0, color='#F44336', alpha=0.4)
        ax.set_title('Drawdown')
        ax.set_ylabel('$')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = f"{ARTIFACT_DIR}/blind_holdout_2021_2025.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {path}")
    else:
        print("    No trades generated at threshold=0.70")
    
    print(f"\n{'='*70}")
    print(f"  BLIND HOLDOUT TEST COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
