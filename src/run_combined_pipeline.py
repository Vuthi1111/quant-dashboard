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
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Local imports
from feature_engineering import build_features, build_labels, build_vol_regime_labels

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ASSET = "NAS100"  # or "GOLD"
if ASSET == "NAS100":
    PRIMARY_TF_FILE = "/Users/macos/Documents/ALGO/03_Data/raw/NAS100/1h_data.csv"
else:
    PRIMARY_TF_FILE = "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/XAUUSD_1H.csv"

NEWS_FILE       = "/Users/macos/Documents/ALGO/03_Data/raw/NAS100/ff_news_dates.json"
OUTPUT_DIR      = "/Users/macos/Documents/ALGO/04_Models/combined_core"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CFG = {
    "horizon_bars":       1,       # 1 bar forward (4H)
    "n_splits":           5,
    "embargo_gap":        2,       # horizon + 1 for safety
    "lgbm_trials":        20,      # Reduced trials for speed
    "n_pca_components":   20,
    "slippage":           0.0001,  # 1 pip approx
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    if 'Tick_Volume' in df.columns:
        df = df.rename(columns={'Tick_Volume': 'TickVolume'})
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    if 'TickVolume' in df.columns:
        agg_dict['TickVolume'] = 'sum'
    return df.resample('4h').agg(agg_dict).dropna()


def compute_sharpe(returns: np.ndarray, rf: float = 0.0) -> float:
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return (np.mean(returns) - rf) / np.std(returns) * np.sqrt(1512)

# ─────────────────────────────────────────────────────────────────────────────
# OPTUNA OBJECTIVES (NESTED CV to prevent leakage)
# ─────────────────────────────────────────────────────────────────────────────
def objective_m1(trial, X_tr_full, y_tr_full):
    # Inner split: take last 20% of training data as inner validation
    split_idx = int(len(X_tr_full) * 0.8)
    X_tr_inner, y_tr_inner = X_tr_full[:split_idx], y_tr_full[:split_idx]
    X_va_inner, y_va_inner = X_tr_full[split_idx:], y_tr_full[split_idx:]
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 4, 15),
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 300),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'subsample': trial.suggest_float('subsample', 0.4, 0.9),
        'n_estimators': 200,
        'random_state': 42,
        'verbosity': -1
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr_inner, y_tr_inner, eval_set=[(X_va_inner, y_va_inner)], callbacks=[lgb.early_stopping(30, verbose=False)])
    preds = model.predict_proba(X_va_inner)[:, 1]
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_va_inner, preds)
    except:
        return 0.5


def objective_m2(trial, X_tr_full, y_tr_full):
    # Inner split: take last 20% of training data as inner validation
    split_idx = int(len(X_tr_full) * 0.8)
    X_tr_inner, y_tr_inner = X_tr_full[:split_idx], y_tr_full[:split_idx]
    X_va_inner, y_va_inner = X_tr_full[split_idx:], y_tr_full[split_idx:]
    
    params = {
        'objective': 'huber',
        'alpha': trial.suggest_float('alpha', 0.5, 1.5),
        'metric': 'huber',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 4, 15),
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 300),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'subsample': trial.suggest_float('subsample', 0.4, 0.9),
        'n_estimators': 200,
        'random_state': 42,
        'verbosity': -1
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr_inner, y_tr_inner, eval_set=[(X_va_inner, y_va_inner)], callbacks=[lgb.early_stopping(30, verbose=False)])
    preds = model.predict(X_va_inner)
    ic, _ = spearmanr(preds, y_va_inner)
    return ic if not np.isnan(ic) else -1.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def run():
    print(f"\n[1] Loading Data: {ASSET}")
    
    if ASSET == "NAS100":
        df_raw = pd.read_csv(PRIMARY_TF_FILE, sep='\t')
        df_raw.rename(columns={'DateTime': 'Time'}, inplace=True)
    else:
        df_raw = pd.read_csv(PRIMARY_TF_FILE, sep=',')
        df_raw.columns = df_raw.columns.str.capitalize()
        df_raw.rename(columns={'Tick_volume': 'TickVolume', 'Tickvolume': 'TickVolume'}, inplace=True)
        
    df_raw['Time'] = pd.to_datetime(df_raw['Time'])
    df_raw.set_index('Time', inplace=True)
    df_4h = resample_to_4h(df_raw)
    
    print("\n[2] Building Features & Targets...")
    features_df = build_features(df_4h, news_mask_path=NEWS_FILE if ASSET == "NAS100" else None)
    
    # Model 1 Target: Volatility Regime
    m1_labels = build_vol_regime_labels(df_4h, forward_bars=1, bar_offset=1)
    
    # Model 2 Target: Volatility-Adjusted Z-Score Return
    m2_labels = build_labels(df_4h, horizon=CFG["horizon_bars"])
    
    # Join everything
    joined = pd.concat([features_df, m1_labels, m2_labels], axis=1).dropna()
    joined = joined.sort_index()
    print(f"    Usable 4H Rows: {len(joined)}")
    
    feature_cols = [c for c in features_df.columns if c != "news_flag"]
    X_full = joined[feature_cols].values.astype(np.float32)
    
    # Extract labels
    y_m1 = joined["vol_regime"].values.astype(np.int32)
    y_m2 = joined["label_vol_adj_ret"].values.astype(np.float32)
    raw_returns = joined["label_ret"].values.astype(np.float32)
    
    # To compute SL/TP, we need current EWMA volatility. It's recoverable as fwd_ret / vol_adj_ret
    ewma_vol = raw_returns / (y_m2 + 1e-9) 
    
    print("\n[3] Integrated Purged Walk-Forward Evaluator")
    tscv = TimeSeriesSplit(n_splits=CFG["n_splits"], gap=CFG["embargo_gap"])
    
    all_oos_returns_naked = []
    all_oos_returns_integrated = []
    all_oos_preds = []
    
    from sklearn.metrics import roc_auc_score
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full)):
        print(f"\n  ▶ FOLD {fold+1}/{CFG['n_splits']}")
        
        X_tr, X_va = X_full[train_idx], X_full[val_idx]
        y1_tr, y1_va = y_m1[train_idx], y_m1[val_idx]
        y2_tr, y2_va = y_m2[train_idx], y_m2[val_idx]
        ret_va = raw_returns[val_idx]
        vol_va = ewma_vol[val_idx]
        
        # PCA
        scaler = StandardScaler()
        pca = PCA(n_components=CFG["n_pca_components"])
        X_tr_p = pca.fit_transform(scaler.fit_transform(X_tr))
        X_va_p = pca.transform(scaler.transform(X_va))
        
        # --- TRAIN MODEL 1 (Classifier) ---
        study1 = optuna.create_study(direction='maximize')
        study1.optimize(lambda t: objective_m1(t, X_tr_p, y1_tr), n_trials=CFG["lgbm_trials"])
        params1 = study1.best_params
        params1.update({'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'n_estimators': 200, 'verbosity': -1, 'random_state': 42})
        m1 = lgb.LGBMClassifier(**params1).fit(X_tr_p, y1_tr)
        
        # --- TRAIN MODEL 2 (Regressor) ---
        study2 = optuna.create_study(direction='maximize')
        study2.optimize(lambda t: objective_m2(t, X_tr_p, y2_tr), n_trials=CFG["lgbm_trials"])
        params2 = study2.best_params
        params2.update({'objective': 'huber', 'metric': 'huber', 'boosting_type': 'gbdt', 'n_estimators': 200, 'verbosity': -1, 'random_state': 42})
        m2 = lgb.LGBMRegressor(**params2).fit(X_tr_p, y2_tr)
        
        # --- PREDICT ON OOS VALIDATION SET ---
        preds_m1 = m1.predict_proba(X_va_p)[:, 1] # Probability of High Volatility
        preds_m2 = m2.predict(X_va_p) # Z-Score
        
        # --- INTEGRATION (BAYESIAN MULTIPLICATION) ---
        clipped_z = np.clip(preds_m2, -3.0, 3.0)
        integrated_pos = preds_m1 * clipped_z
        
        # 3. Simulate Returns (Naked vs Integrated) - Naive 1-Bar Hold
        naked_ret = (clipped_z * ret_va) - (np.abs(clipped_z) * CFG["slippage"])
        integ_ret = (integrated_pos * ret_va) - (np.abs(integrated_pos) * CFG["slippage"])
        
        all_oos_returns_naked.extend(naked_ret)
        all_oos_returns_integrated.extend(integ_ret)
        
        # --- SAVE PREDICTIONS FOR EXIT SIMULATOR ---
        # Get the original index for the validation set
        val_times = joined.index[val_idx]
        fold_preds = pd.DataFrame({
            'Time': val_times,
            'Fold': fold + 1,
            'Regime_Prob': preds_m1,
            'Predicted_Z': preds_m2,
            'Clipped_Z': clipped_z,
            'Integrated_Score': integrated_pos,
            'EWMA_Vol': vol_va,
            'Entry_Price': df_4h.loc[val_times, 'Close'].values,
            'Next_Open': df_4h['Open'].shift(-1).loc[val_times].values,
            'Next_High': df_4h['High'].shift(-1).loc[val_times].values,
            'Next_Low': df_4h['Low'].shift(-1).loc[val_times].values,
            'Next_Close': df_4h['Close'].shift(-1).loc[val_times].values
        })
        all_oos_preds.append(fold_preds)
        
        print(f"    M1 (Regime) AUC: {roc_auc_score(y1_va, preds_m1) if len(np.unique(y1_va))>1 else 0:.3f}")
        ic, _ = spearmanr(preds_m2, y2_va)
        print(f"    M2 (Direct) IC:  {ic:.4f}")
        print(f"    Naked Sharpe:    {compute_sharpe(naked_ret):.2f}")
        print(f"    Integ Sharpe:    {compute_sharpe(integ_ret):.2f}")
        
        # 4. Independent Risk Output (Just for logging)
        # Take Profit is Z * current volatility
        tp_distance = np.abs(clipped_z) * vol_va
        # Stop loss is fixed 1.0 ATR
        sl_distance = vol_va * 1.0 
    
    # Plotting Final Equity Curve
    print("\n[4] OVERALL OUT-OF-SAMPLE PERFORMANCE")
    final_naked = np.array(all_oos_returns_naked)
    final_integ = np.array(all_oos_returns_integrated)
    
    print(f"    Total Naked Sharpe:      {compute_sharpe(final_naked):.2f}")
    print(f"    Total Integrated Sharpe: {compute_sharpe(final_integ):.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(final_naked), label="Naked (M2 Only) Equity", alpha=0.6, color="red")
    plt.plot(np.cumsum(final_integ), label="Integrated (M1 * M2) Equity", linewidth=2, color="green")
    plt.title(f"Walk-Forward OOS Equity Curve ({ASSET})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/equity_comparison.png")
    print(f"    Saved equity comparison chart to {OUTPUT_DIR}/equity_comparison.png")
    
    # Save predictions
    final_preds_df = pd.concat(all_oos_preds)
    final_preds_df.to_csv(f"{OUTPUT_DIR}/oos_predictions.csv", index=False)
    print(f"    Saved OOS predictions to {OUTPUT_DIR}/oos_predictions.csv")

if __name__ == "__main__":
    run()
