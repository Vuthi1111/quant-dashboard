"""
run_pipeline_model3.py — REFINED
═══════════════════════════════════════════════════════════════════════════════
Intraday VWAP & Volatility Scalping Pipeline (15M)

Refinements over v1:
  1. Expanded feature set: adds volatility estimators, momentum, RSI, 
     Bollinger Band position, range ratio, tick vol acceleration
  2. Fixes Hurst silently failing — uses a simpler R/S estimator inline
  3. Purged Walk-Forward validation (5 folds) instead of single train/test
  4. Permutation null test to confirm signal isn't structural leakage
  5. Proper feature exclusion list
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss
import optuna

sys.path.insert(0, "/Users/macos/Documents/ALGO/projects/volatility_regime_model/src")
from feature_engineering import (
    load_mt5_csv, resample_to_15m, 
    build_intraday_features, create_vwap_scalp_labels,
    garman_klass, parkinson, rogers_satchell,
    range_ratio, tick_vol_acceleration
)

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED INTRADAY FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def hurst_rs(series, min_n=8):
    """Simple R/S Hurst exponent — no external dependency."""
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


def build_full_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full intraday feature set combining VWAP microstructure with 
    volatility estimators and momentum.
    """
    feat = pd.DataFrame(index=df.index)
    
    # ── 1. Time of Day (cyclical) ──
    hours = df.index.hour + df.index.minute / 60.0
    feat['hour_sin'] = pd.Series(np.sin(2 * np.pi * hours / 24.0), index=df.index).shift(1)
    feat['hour_cos'] = pd.Series(np.cos(2 * np.pi * hours / 24.0), index=df.index).shift(1)
    
    # Day of week
    dow = df.index.dayofweek
    feat['dow_sin'] = pd.Series(np.sin(2 * np.pi * dow / 5.0), index=df.index).shift(1)
    feat['dow_cos'] = pd.Series(np.cos(2 * np.pi * dow / 5.0), index=df.index).shift(1)
    
    # ── 2. Daily Anchored VWAP ──
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
    
    # ── 3. Log Returns & Lags ──
    lr = np.log(df['Close'] / df['Close'].shift(1))
    for lag in [1, 2, 4, 8, 16]:
        feat[f'ret_lag{lag}'] = lr.shift(lag)
    
    # ── 4. Volatility Estimators ──
    for w in [4, 8, 16]:
        feat[f'GK_{w}'] = garman_klass(df, w).shift(1)
        feat[f'PK_{w}'] = parkinson(df, w).shift(1)
        feat[f'RS_{w}'] = rogers_satchell(df, w).shift(1)
    
    # Historical vol
    feat['HV_16'] = lr.rolling(16).std().shift(1)
    feat['HV_96'] = lr.rolling(96).std().shift(1)
    feat['vol_ratio'] = feat['HV_16'] / (feat['HV_96'] + 1e-9)
    
    # ── 5. Momentum ──
    feat['roc_4'] = (df['Close'] / df['Close'].shift(4) - 1).shift(1)
    feat['roc_16'] = (df['Close'] / df['Close'].shift(16) - 1).shift(1)
    feat['roc_96'] = (df['Close'] / df['Close'].shift(96) - 1).shift(1)
    
    # ── 6. RSI (14-bar) ──
    delta = lr.copy()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    feat['rsi_14'] = (100 - 100 / (1 + rs)).shift(1)
    
    # ── 7. Bollinger Band Position ──
    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    feat['bb_pos'] = ((df['Close'] - ma20) / (2 * std20 + 1e-9)).shift(1)
    
    # ── 8. Microstructure ──
    feat['range_ratio'] = range_ratio(df).shift(1)
    feat['tickvol_accel'] = tick_vol_acceleration(df).shift(1)
    
    # ── 9. Relative Tick Volume (RTV) ──
    vol_4 = df['Tick_Volume'].rolling(4).sum()
    vol_96_avg = df['Tick_Volume'].rolling(96).mean() * 4
    feat['rtv'] = (vol_4 / vol_96_avg.replace(0, np.nan)).shift(1)
    
    # ── 10. Hurst Exponent (inline R/S, rolling 32 bars) ──
    print("    [Features] Computing rolling Hurst (32-bar)...")
    closes = df['Close'].values
    hurst_vals = np.full(len(closes), np.nan)
    for i in range(32, len(closes)):
        hurst_vals[i] = hurst_rs(closes[i-32:i])
    feat['hurst_32'] = pd.Series(hurst_vals, index=df.index).shift(1)
    
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# PURGED WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def purged_walk_forward(X, y, w, feature_cols, n_folds=5, purge_bars=16):
    """
    Time-series walk-forward with purge gap between train and test.
    """
    n = len(X)
    fold_size = n // (n_folds + 1)  # Reserve first fold for initial training
    
    results = []
    
    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_start = train_end + purge_bars  # Purge gap
        test_end = min(test_start + fold_size, n)
        
        if test_end <= test_start:
            break
            
        X_train = X[:train_end]
        y_train = y[:train_end]
        w_train = w[:train_end] if w is not None else None
        
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        if len(np.unique(y_test)) < 2:
            continue
            
        model = lgb.LGBMClassifier(
            objective='binary', metric='auc', boosting_type='gbdt',
            learning_rate=0.05, num_leaves=16, max_depth=5,
            feature_fraction=0.8, verbose=-1, n_estimators=200
        )
        model.fit(X_train, y_train, sample_weight=w_train)
        
        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        brier = brier_score_loss(y_test, preds)
        pos_rate = y_test.mean()
        
        results.append({
            'fold': fold + 1,
            'train_n': len(X_train),
            'test_n': len(X_test),
            'auc': auc,
            'brier': brier,
            'pos_rate': pos_rate,
            'model': model
        })
        
        print(f"  Fold {fold+1}: Train={len(X_train):,} | Test={len(X_test):,} | AUC={auc:.4f} | Brier={brier:.4f} | PosRate={pos_rate:.2%}")
    
    return results


def permutation_null_test(X, y, w, n_shuffles=5):
    """Shuffle labels and measure AUC to get the null distribution."""
    null_aucs = []
    n = len(X)
    split = int(n * 0.8)
    
    for i in range(n_shuffles):
        y_shuf = y.copy()
        np.random.seed(i + 42)
        np.random.shuffle(y_shuf)
        
        model = lgb.LGBMClassifier(
            objective='binary', metric='auc', boosting_type='gbdt',
            learning_rate=0.05, num_leaves=16, max_depth=5,
            verbose=-1, n_estimators=100
        )
        model.fit(X[:split], y_shuf[:split], sample_weight=w[:split] if w is not None else None)
        preds = model.predict_proba(X[split:])[:, 1]
        
        try:
            auc = roc_auc_score(y_shuf[split:], preds)
        except:
            auc = 0.5
        null_aucs.append(auc)
    
    return null_aucs


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  MODEL 3: INTRADAY VWAP SCALPING PIPELINE (15M) — REFINED")
    print("=" * 70)
    
    # ── 1. LOAD DATA ──
    data_path = "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/XAUUSD_M5.csv"
    print(f"\n[1] Loading {data_path}...")
    df_raw = load_mt5_csv(data_path)
    
    print("    Resampling to 15M...")
    df_15m = resample_to_15m(df_raw)
    print(f"    Loaded {len(df_15m):,} 15M bars.")
    
    # ── 2. BUILD FEATURES ──
    print("\n[2] Building Full Intraday Features...")
    features_df = build_full_intraday_features(df_15m)
    
    df_full = pd.concat([df_15m, features_df], axis=1)
    
    # ── 3. CREATE LABELS ──
    print("\n[3] Generating VWAP Scalp Labels (Triple Barrier)...")
    df_labeled = create_vwap_scalp_labels(df_full, max_horizon=16)
    
    # ── 4. CLEAN DATA ──
    meta_cols = [
        "Open", "High", "Low", "Close", "Tick_Volume", "Volume", "Spread",
        "spread", "vwap_label", "fpt_horizon", "uniqueness_weight",
        "date", "typ_price", "tv", "tv2"
    ]
    feature_cols = [c for c in df_labeled.columns if c not in meta_cols]
    
    valid_mask = (~df_labeled[feature_cols].isna().any(axis=1) 
                  & ~df_labeled['vwap_label'].isna())
    df_clean = df_labeled[valid_mask].copy()
    
    positives = df_clean['vwap_label'].sum()
    negatives = len(df_clean) - positives
    
    print(f"\n[4] Data Summary:")
    print(f"    Valid labeled events : {len(df_clean):,}")
    print(f"    Base Rate (Positive) : {positives / len(df_clean):.2%}")
    print(f"    Positives            : {int(positives):,}")
    print(f"    Negatives            : {int(negatives):,}")
    print(f"    Features             : {len(feature_cols)}")
    print(f"    Feature list         : {feature_cols}")
    
    if len(df_clean) < 2000:
        print("\n⚠ Not enough data. Exiting.")
        return
    
    X = df_clean[feature_cols].values
    y = df_clean['vwap_label'].values
    w = df_clean['uniqueness_weight'].values if 'uniqueness_weight' in df_clean.columns else None
    
    # ── 5. PURGED WALK-FORWARD VALIDATION ──
    print("\n[5] Running Purged Walk-Forward Validation (5 folds, purge=16 bars)...")
    wf_results = purged_walk_forward(X, y, w, feature_cols, n_folds=5, purge_bars=16)
    
    if wf_results:
        aucs = [r['auc'] for r in wf_results]
        briers = [r['brier'] for r in wf_results]
        print(f"\n  ── Walk-Forward Summary ──")
        print(f"  Mean AUC   : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"  Min  AUC   : {np.min(aucs):.4f}")
        print(f"  Max  AUC   : {np.max(aucs):.4f}")
        print(f"  Mean Brier : {np.mean(briers):.4f}")
    
    # ── 6. PERMUTATION NULL TEST ──
    print("\n[6] Running Permutation Null Test (5 shuffles)...")
    null_aucs = permutation_null_test(X, y, w, n_shuffles=5)
    print(f"  Null AUCs: {[f'{a:.4f}' for a in null_aucs]}")
    print(f"  Null Mean: {np.mean(null_aucs):.4f}")
    
    if wf_results:
        real_mean = np.mean(aucs)
        null_mean = np.mean(null_aucs)
        lift = real_mean - null_mean
        print(f"\n  Signal Lift (Real - Null): {lift:.4f}")
        if lift > 0.05:
            print("  ✅ SIGNAL IS REAL (lift > 0.05)")
        elif lift > 0.02:
            print("  ⚠️  MARGINAL SIGNAL (0.02 < lift < 0.05)")
        else:
            print("  ❌ NO REAL SIGNAL (lift < 0.02)")
    
    # ── 7. FEATURE IMPORTANCE (last fold) ──
    if wf_results:
        last_model = wf_results[-1]['model']
        importances = last_model.feature_importances_
        idx = np.argsort(importances)[::-1]
        print(f"\n[7] Feature Importance (Last Fold):")
        for i in range(min(10, len(feature_cols))):
            print(f"    {feature_cols[idx[i]]:20s} : {importances[idx[i]]:.0f}")
    
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
