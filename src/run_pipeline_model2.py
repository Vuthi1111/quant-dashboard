"""
run_pipeline_4h.py
═══════════════════════════════════════════════════════════════════════════════
MAIN ORCHESTRATOR — Adaptive Walk-Forward Supervised Learning Pipeline
NAS100 — MT5 Data (Resampled to 4H)
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

warnings.filterwarnings("ignore")

# ── Local modules
from feature_engineering import build_features, create_reversion_labels, load_mt5_csv, resample_to_4h
from walk_forward         import PurgedWalkForwardSplit
from model_stack          import run_fold, build_preprocessor

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR        = "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD"
PRIMARY_TF_FILE = os.path.join(DATA_DIR, "XAUUSD_1H.csv")
NEWS_FILE       = os.path.join(DATA_DIR, "ff_news_dates.json")
OUTPUT_DIR      = "/Users/macos/Documents/ALGO/04_Models/walk_forward_ml/output_4h"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CFG = {
    "min_train_months":       12,      # Minimum months of data before first fold
    "embargo_bars":           10,      # ~1.5 weeks of 4H bars between train / val / test
    "rolling_window_months":  24,      # Rolling window = 24 months train (≈ 2 years)
    "holdout_months":         15,      # Final 15 months locked as terminal test
    "step_months":             2,      # Advance 2 month per fold to speed up tuning
    "n_pca_components":       25,      # PCA dimensions post-preprocessing
    "lgbm_trials":            25,      # Optuna trials per fold
    "lstm_seq_len":           10,      # LSTM lookback window (4H)
    "lstm_epochs":            20,      # LSTM max epochs per fold
    # ── Pivot 1: Volatility Regime Classification ──
    "vol_forward_bars":        1,      # 4H forward window (1 x 4H bar)
    "vol_bar_offset":          1,      # Skip 1 bar before window starts
    "vol_pct_high":           0.70,    # Top 30% → HIGH vol regime  (label=1)
    "vol_pct_low":            0.30,    # Bottom 30% → LOW vol regime (label=0)
    "vol_rolling_baseline":  120,      # Rolling 20-day baseline for percentile calc (20 * 6 = 120 bars)
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 80)
print("  ADAPTIVE WALK-FORWARD SUPERVISED LEARNING PIPELINE — GOLD (4H)")
print("═" * 80)
print(f"\n[1] Loading {PRIMARY_TF_FILE} and resampling to 4H...")
df_raw = load_mt5_csv(PRIMARY_TF_FILE)
df_raw = resample_to_4h(df_raw)
print(f"    Loaded {len(df_raw):,} 4H bars | {df_raw.index[0].date()} → {df_raw.index[-1].date()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Building feature matrix (Pivot 2: Probability of Mean Reversion)...")
features_df = build_features(df_raw, news_mask_path=NEWS_FILE)

# Add FPT Mean-Reversion Labels
joined = pd.concat([df_raw, features_df], axis=1)
joined = create_reversion_labels(joined, k=1.5)

# Drop rows where target is NaN (i.e. not an extreme setup)
joined = joined.dropna(subset=["reversion_label"])

# Drop columns that are labels/metadata from X
drop_cols = ["reversion_label", "fpt_horizon", "uniqueness_weight"]
feature_cols = [c for c in joined.columns if c not in drop_cols]

X_full = joined[feature_cols].values.astype(np.float32)
X_full = np.where(np.isinf(X_full), np.nan, X_full)
y_full = joined["reversion_label"].values.astype(np.int32)
w_full = joined["uniqueness_weight"].values.astype(np.float32)
news_mask = joined.get("news_flag", pd.Series(0, index=joined.index)).values.astype(np.float32)
full_index = joined.index

print(f"    Features : {X_full.shape[1]}")
print(f"    Samples  : {X_full.shape[0]:,} (after dropping non-extremes)")
print(f"    Reverts  : {y_full.mean():.3f} | Fails: {1-y_full.mean():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — WALK-FORWARD SPLIT SETUP
# ─────────────────────────────────────────────────────────────────────────────
wfv = PurgedWalkForwardSplit(
    index                 = full_index,
    min_train_months      = CFG["min_train_months"],
    embargo_bars          = CFG["embargo_bars"],
    rolling_window_months = CFG["rolling_window_months"],
    step_months           = CFG["step_months"],
    holdout_months        = CFG["holdout_months"],
)

holdout_idx = wfv.holdout_idx()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — WALK-FORWARD TRAINING
# ─────────────────────────────────────────────────────────────────────────────
import pickle
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "wfv_checkpoint_4h_model2.pkl")
results_expanding = []

if os.path.exists(CHECKPOINT_FILE):
    print(f"\n[!] Found checkpoint {CHECKPOINT_FILE}. Resuming...")
    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            ckpt = pickle.load(f)
            results_expanding = ckpt.get("expanding", [])
    except Exception as e:
        print(f"    Failed to load checkpoint: {e}")

print("\n[4] Running Expanding Walk-Forward Training...\n")

window_type = "expanding"
completed_folds = {r.fold_id for r in results_expanding}

for fold in wfv.generate_folds(window_type):
    if fold.fold_id in completed_folds:
        print(f"  ✓ Skipping Fold {fold.fold_id} (Loaded from checkpoint)")
        continue
        
    print(f"  ▶ Fold {fold.fold_id} | Train: {len(fold.train_idx)} | Val: {len(fold.val_idx)}")
    fold_res = run_fold(
        fold             = fold,
        X_full           = X_full,
        y_full           = y_full,
        w_full           = w_full,
        feature_names    = features_df.columns.tolist(),
        news_mask_full   = news_mask,
        n_pca_components = CFG["n_pca_components"],
        lgbm_trials      = CFG["lgbm_trials"],
        lstm_seq_len     = CFG["lstm_seq_len"],
        lstm_epochs      = CFG["lstm_epochs"],
    )
    results_expanding.append(fold_res)
    
    # Save checkpoint
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump({"expanding": results_expanding}, f)

# Evaluate on Holdout and get best params
print("\n[5] Extracting optimal 4H parameters on Holdout Set...")
X_train = X_full[:holdout_idx[0] - CFG["embargo_bars"]]
y_train = y_full[:holdout_idx[0] - CFG["embargo_bars"]]
X_test  = X_full[holdout_idx]
y_test  = y_full[holdout_idx]

from model_stack import train_lightgbm, build_preprocessor

preprocessor = build_preprocessor(n_components=min(CFG["n_pca_components"], X_train.shape[1] - 1))
X_train_t = preprocessor.fit_transform(X_train)
X_test_t  = preprocessor.transform(X_test)

best_lgbm = train_lightgbm(X_train_t, y_train, X_test_t, y_test, w_train=w_full[:holdout_idx[0] - CFG["embargo_bars"]], n_trials=CFG["lgbm_trials"])

print("\n" + "="*80)
print(f"BEST LIGHTGBM PARAMS FOR 4H FROM HOLDOUT TUNING:")
print(f"  params: {best_lgbm.params}")
print(f"  best_iteration: {best_lgbm.best_iteration}")
print("="*80)

preds = best_lgbm.predict(X_test_t)
auc = roc_auc_score(y_test, preds)

print(f"\n[✓] 4H HOLDOUT AUC ROC: {auc:.4f}")
