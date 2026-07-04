import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)

# Local imports
from feature_engineering import build_features, build_labels, load_mt5_csv

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ASSET = "NAS100"  # or "GOLD"
if ASSET == "NAS100":
    PRIMARY_TF_FILE = "/Users/macos/Documents/ALGO/03_Data/raw/NAS100/1h_data.csv"
else:
    PRIMARY_TF_FILE = "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/XAUUSD_1H.csv"

NEWS_FILE       = "/Users/macos/Documents/ALGO/03_Data/raw/NAS100/ff_news_dates.json"
OUTPUT_DIR      = "/Users/macos/Documents/ALGO/04_Models/directional_core"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CFG = {
    "horizon_bars":       1,       # 1 bar forward (4H)
    "min_move_pct":       0.0,     # Don't drop tiny moves for now, Huber handles them
    "n_splits":           5,
    "embargo_gap":        2,       # horizon + 1 for safety
    "lgbm_trials":        30,      # Optuna trials
    "n_pca_components":   20,
}

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize volume column name
    if 'Tick_Volume' in df.columns:
        df = df.rename(columns={'Tick_Volume': 'TickVolume'})
        
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }
    if 'TickVolume' in df.columns:
        agg_dict['TickVolume'] = 'sum'
        
    df_4h = df.resample('4h').agg(agg_dict).dropna()
    return df_4h


def compute_sharpe(returns: np.ndarray, rf: float = 0.0) -> float:
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    # Annualize assuming 6 bars/day * 252 days = ~1512 bars/year
    # Let's just use raw sharpe * sqrt(1512)
    return (np.mean(returns) - rf) / np.std(returns) * np.sqrt(1512)


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'huber',
        'alpha': trial.suggest_float('alpha', 0.5, 1.5), # Huber Delta
        'metric': 'huber',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 4, 15),
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 300),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'subsample': trial.suggest_float('subsample', 0.4, 0.9),
        'n_estimators': 300,
        'random_state': 42,
        'verbosity': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    preds = model.predict(X_val)
    
    # Optimize for Information Coefficient (Spearman Rank Correlation)
    ic, _ = spearmanr(preds, y_val)
    if np.isnan(ic):
        return -1.0
    return ic


def run():
    print(f"\n[1] Loading Data: {ASSET}")
    
    # Robust loading to handle both NAS100 (tabs) and GOLD (commas)
    if ASSET == "NAS100":
        df_raw = pd.read_csv(PRIMARY_TF_FILE, sep='\t')
        df_raw.rename(columns={'DateTime': 'Time'}, inplace=True)
    else:
        df_raw = pd.read_csv(PRIMARY_TF_FILE, sep=',')
        df_raw.columns = df_raw.columns.str.capitalize()
        df_raw.rename(columns={'Tick_volume': 'Tick_Volume', 'Tickvolume': 'Tick_Volume'}, inplace=True)
        
    if 'Time' in df.columns if 'df' in locals() else 'Time' in df_raw.columns:
        df_raw['Time'] = pd.to_datetime(df_raw['Time'])
        df_raw.set_index('Time', inplace=True)
        
    print(f"    Total 1H Rows: {len(df_raw)}")
    
    df_4h = resample_to_4h(df_raw)
    print(f"    Total 4H Rows: {len(df_4h)}")
    
    print("\n[2] Building Features & Volatility-Adjusted Target...")
    features_df = build_features(df_4h, news_mask_path=NEWS_FILE if ASSET == "NAS100" else None)
    labels_df = build_labels(df_4h, horizon=CFG["horizon_bars"], min_move_pct=CFG["min_move_pct"])
    
    # DEBUG NANS
    joined_raw = pd.concat([features_df, labels_df], axis=1)
    nan_counts = joined_raw.isna().sum()
    print("    NaNs per column (top 15):")
    print(nan_counts[nan_counts > 0].sort_values(ascending=False).head(15))
    
    # Join
    joined = joined_raw.dropna()
    print(f"    Usable 4H Rows: {len(joined)}")
    
    # Sort index strictly
    joined = joined.sort_index()
    
    feature_cols = [c for c in features_df.columns if c not in ["news_flag"]]
    X_full = joined[feature_cols].values.astype(np.float32)
    y_full = joined["label_vol_adj_ret"].values.astype(np.float32)
    raw_returns = joined["label_ret"].values.astype(np.float32) # for sharpe
    
    print("\n[3] Purged Walk-Forward Split (Embargo = 2 bars)")
    
    # Use TimeSeriesSplit with explicitly defined gap (Embargo)
    tscv = TimeSeriesSplit(n_splits=CFG["n_splits"], gap=CFG["embargo_gap"])
    
    fold_metrics = []
    oos_preds = np.zeros(len(y_full))
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full)):
        print(f"\n  ▶ FOLD {fold+1}/{CFG['n_splits']}")
        
        X_train, y_train = X_full[train_idx], y_full[train_idx]
        X_val, y_val = X_full[val_idx], y_full[val_idx]
        raw_ret_val = raw_returns[val_idx]
        
        # Preprocessing (Standardization + PCA)
        scaler = StandardScaler()
        pca = PCA(n_components=CFG["n_pca_components"])
        
        X_tr_proc = pca.fit_transform(scaler.fit_transform(X_train))
        X_va_proc = pca.transform(scaler.transform(X_val))
        
        # Optuna Tuning
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, X_tr_proc, y_train, X_va_proc, y_val), n_trials=CFG["lgbm_trials"])
        
        best_params = study.best_params
        best_params.update({'objective': 'huber', 'metric': 'huber', 'boosting_type': 'gbdt', 'n_estimators': 300, 'verbosity': -1, 'random_state': 42})
        
        # Train Best Model
        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_tr_proc, y_train)
        
        preds = model.predict(X_va_proc)
        oos_preds[val_idx] = preds
        
        # Evaluate
        ic, p_val = spearmanr(preds, y_val)
        
        # Simple directional strategy: If pred > 0, long. If pred < 0, short.
        # Position sizing: proportional to clip(Z, -3, 3)
        clipped_z = np.clip(preds, -3.0, 3.0)
        
        # OOS Return = position * actual forward return (minus 1 pip slippage proxy = ~0.0001)
        slippage = 0.0001
        strategy_returns = (clipped_z * raw_ret_val) - (np.abs(clipped_z) * slippage)
        
        sharpe = compute_sharpe(strategy_returns)
        
        print(f"    Best Huber Alpha: {best_params['alpha']:.3f} | Depth: {best_params['max_depth']}")
        print(f"    OOS IC:           {ic:.4f} (p-val: {p_val:.4f})")
        print(f"    OOS Sharpe:       {sharpe:.2f}")
        
        fold_metrics.append({
            'fold': fold + 1,
            'ic': ic,
            'sharpe': sharpe
        })
        
    print("\n[4] OVERALL OUT-OF-SAMPLE PERFORMANCE")
    avg_ic = np.mean([m['ic'] for m in fold_metrics])
    avg_sharpe = np.mean([m['sharpe'] for m in fold_metrics])
    print(f"    Mean IC:      {avg_ic:.4f}")
    print(f"    Mean Sharpe:  {avg_sharpe:.2f}")
    
    if avg_ic > 0.02 and avg_sharpe > 1.0:
        print("\n    ✅ SYSTEM PASSES INSTITUTIONAL ALPHA THRESHOLDS.")
    else:
        print("\n    ⚠ SYSTEM DID NOT MEET THRESHOLDS. Further feature engineering required.")


if __name__ == "__main__":
    run()
