"""
run_pipeline.py
═══════════════════════════════════════════════════════════════════════════════
MAIN ORCHESTRATOR — Adaptive Walk-Forward Supervised Learning Pipeline
NAS100 — MT5 Data

Usage:
    python run_pipeline.py

Output directory: /Users/macos/Documents/ALGO/04_Models/walk_forward_ml/output/
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
from feature_engineering import build_features, build_labels, build_vol_regime_labels, load_mt5_csv
from walk_forward         import PurgedWalkForwardSplit
from model_stack          import run_fold, build_preprocessor
from visualization        import (
    plot_equity_curves,
    plot_auc_heatmap,
    plot_calibration,
    plot_drawdown,
    plot_exp_vs_rolling,
    plot_holdout_summary,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR        = "/Users/macos/Documents/ALGO/03_Data/raw/NAS100"
PRIMARY_TF_FILE = os.path.join(DATA_DIR, "1h_data.csv")
NEWS_FILE       = os.path.join(DATA_DIR, "ff_news_dates.json")
OUTPUT_DIR      = "/Users/macos/Documents/ALGO/04_Models/walk_forward_ml/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CFG = {
    "min_train_months":       12,      # Minimum months of data before first fold
    "embargo_bars":           40,      # ~1 week of 1H bars between train / val / test
    "rolling_window_months":  24,      # Rolling window = 24 months train (≈ 2 years)
    "holdout_months":         15,      # Final 15 months locked as terminal test
    "step_months":             1,      # Advance 1 month per fold
    "n_pca_components":       25,      # PCA dimensions post-preprocessing
    "lgbm_trials":            20,      # Optuna trials per fold
    "lstm_seq_len":           20,      # LSTM lookback window
    "lstm_epochs":            20,      # LSTM max epochs per fold
    # ── Pivot 1: Volatility Regime Classification ──
    "vol_forward_bars":        4,      # 4H forward window (4 x 1H bars)
    "vol_bar_offset":          4,      # Skip 4 bars before window starts (eliminates same-bar GK artifact)
    "vol_pct_high":           0.70,    # Top 30% → HIGH vol regime  (label=1)
    "vol_pct_low":            0.30,    # Bottom 30% → LOW vol regime (label=0)
    "vol_rolling_baseline":  480,      # Rolling 20-day baseline for percentile calc
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 80)
print("  ADAPTIVE WALK-FORWARD SUPERVISED LEARNING PIPELINE — NAS100")
print("═" * 80)
print(f"\n[1] Loading {PRIMARY_TF_FILE}...")
df_raw = load_mt5_csv(PRIMARY_TF_FILE)
print(f"    Loaded {len(df_raw):,} bars | {df_raw.index[0].date()} → {df_raw.index[-1].date()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Building feature matrix (Pivot 1: Volatility Regime Classification)...")
features_df  = build_features(df_raw, news_mask_path=NEWS_FILE)
regime_df    = build_vol_regime_labels(
    df_raw,
    forward_bars      = CFG["vol_forward_bars"],
    bar_offset        = CFG["vol_bar_offset"],
    regime_pct_high   = CFG["vol_pct_high"],
    regime_pct_low    = CFG["vol_pct_low"],
    rolling_baseline  = CFG["vol_rolling_baseline"],
)

# Align: drop NaN labels (middle 40% discarded by design)
joined      = pd.concat([features_df, regime_df], axis=1).dropna(subset=["vol_regime"])
X_full      = joined[features_df.columns].values.astype(np.float32)
y_full      = joined["vol_regime"].values.astype(np.int32)
news_mask   = joined.get("news_flag", pd.Series(0, index=joined.index)).values.astype(np.float32)
full_index  = joined.index

print(f"    Features : {X_full.shape[1]}")
print(f"    Samples  : {X_full.shape[0]:,} (after discarding middle 40%)")
print(f"    High Vol : {y_full.mean():.3f} | Low Vol: {1-y_full.mean():.3f}")
print(f"    Date range: {full_index[0].date()} → {full_index[-1].date()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — WALK-FORWARD SPLIT SETUP
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Initialising Walk-Forward Splitter...")
wfv = PurgedWalkForwardSplit(
    index                 = full_index,
    min_train_months      = CFG["min_train_months"],
    embargo_bars          = CFG["embargo_bars"],
    rolling_window_months = CFG["rolling_window_months"],
    step_months           = CFG["step_months"],
    holdout_months        = CFG["holdout_months"],
)

wfv.print_fold_summary("expanding")
wfv.print_fold_summary("rolling")

holdout_idx = wfv.holdout_idx()
print(f"\n    Locked holdout: {full_index[holdout_idx[0]].date()} → "
      f"{full_index[holdout_idx[-1]].date()} ({len(holdout_idx)} bars)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — WALK-FORWARD TRAINING
# ─────────────────────────────────────────────────────────────────────────────
import pickle
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "wfv_checkpoint.pkl")
results_expanding = []
results_rolling   = []

if os.path.exists(CHECKPOINT_FILE):
    print(f"\n[!] Found checkpoint {CHECKPOINT_FILE}. Resuming...")
    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            ckpt = pickle.load(f)
            results_expanding = ckpt.get("expanding", [])
            results_rolling   = ckpt.get("rolling", [])
    except Exception as e:
        print(f"    Failed to load checkpoint: {e}")

print("\n[4] Running Walk-Forward Training...\n")

for window_type, results_list in [("expanding", results_expanding),
                                   ("rolling",   results_rolling)]:
    print(f"\n{'─'*60}")
    print(f"  Window Type: {window_type.upper()}")
    print(f"{'─'*60}")
    
    completed_folds = {r.fold_id for r in results_list}

    for fold in wfv.generate_folds(window_type):
        if fold.fold_id in completed_folds:
            print(f"  ✓ Skipping Fold {fold.fold_id} (Loaded from checkpoint)")
            continue
            
        print(f"\n{fold.summary()}")
        try:
            result = run_fold(
                fold           = fold,
                X_full         = X_full.copy(),
                y_full         = y_full.copy(),
                feature_names  = list(features_df.columns),
                n_pca_components = CFG["n_pca_components"],
                lgbm_trials    = CFG["lgbm_trials"],
                lstm_seq_len   = CFG["lstm_seq_len"],
                lstm_epochs    = CFG["lstm_epochs"],
                news_mask_full = news_mask.copy(),
            )
            results_list.append(result)
            
            # Auto-Checkpoint after every successful fold
            with open(CHECKPOINT_FILE, "wb") as f:
                pickle.dump({"expanding": results_expanding, "rolling": results_rolling}, f)
                
        except Exception as e:
            print(f"  ✗ Fold {fold.fold_id} failed: {e}")
            continue

print(f"\n  Completed: {len(results_expanding)} expanding folds, "
      f"{len(results_rolling)} rolling folds")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — FINAL HOLDOUT EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Final Holdout Evaluation (locked test zone)...")

X_ho = X_full[holdout_idx]
y_ho = y_full[holdout_idx]
nm_ho = news_mask[holdout_idx]

# Retrain on ALL WFV data (excluding holdout) for final model
train_up_to_holdout = np.array([i for i in range(len(X_full)) if i not in set(holdout_idx)])
X_tr_final = X_full[train_up_to_holdout]
y_tr_final = y_full[train_up_to_holdout]

# Final preprocessing on full WFV training set
prep_final = build_preprocessor(n_components=CFG["n_pca_components"])
X_tr_f_p   = prep_final.fit_transform(X_tr_final)
X_ho_p     = prep_final.transform(X_ho)
X_ho_p[nm_ho.astype(bool)] = 0

holdout_preds   = {}
holdout_metrics = {}

# Use last expanding fold's trained models for holdout inference
# (or retrain — here we use a quick LightGBM retrained on all WFV data)
import lightgbm as lgb
dtrain_final = lgb.Dataset(X_tr_f_p, label=y_tr_final)
params_final = {"objective": "binary", "metric": "auc",
                "verbosity": -1, "learning_rate": 0.02,
                "num_leaves": 64, "n_estimators": 300}
n_est = params_final.pop("n_estimators")
lgbm_final = lgb.train(params_final, dtrain_final, num_boost_round=n_est)
holdout_preds["LGBM"] = lgbm_final.predict(X_ho_p)

from sklearn.linear_model import LogisticRegression
lr_final = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")
lr_final.fit(X_tr_f_p, y_tr_final)
holdout_preds["LR"] = lr_final.predict_proba(X_ho_p)[:, 1]

# Simple LGBM+LR average as Meta proxy on holdout
holdout_preds["Meta"] = (holdout_preds["LGBM"] + holdout_preds["LR"]) / 2

for name, pred in holdout_preds.items():
    valid = ~np.isnan(pred)
    try:
        holdout_metrics[name] = {
            "auc":      roc_auc_score(y_ho[valid], pred[valid]),
            "accuracy": accuracy_score(y_ho[valid], (pred[valid] > 0.5).astype(int)),
            "logloss":  log_loss(y_ho[valid], pred[valid]),
        }
        print(f"  {name:6s} — AUC: {holdout_metrics[name]['auc']:.4f} | "
              f"Acc: {holdout_metrics[name]['accuracy']:.4f} | "
              f"LogLoss: {holdout_metrics[name]['logloss']:.4f}")
    except Exception as e:
        print(f"  {name} metrics failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — GENERATE ALL VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Generating Visualizations...")

if results_expanding and results_rolling:
    plot_equity_curves(results_expanding, results_rolling, full_index,
                       os.path.join(OUTPUT_DIR, "01_equity_curves.png"))

    plot_auc_heatmap(results_expanding, results_rolling,
                     os.path.join(OUTPUT_DIR, "02_auc_heatmap.png"))

    plot_calibration(results_expanding,
                     os.path.join(OUTPUT_DIR, "03_calibration_expanding.png"),
                     title_prefix="Expanding | ")

    plot_calibration(results_rolling,
                     os.path.join(OUTPUT_DIR, "03_calibration_rolling.png"),
                     title_prefix="Rolling | ")

    plot_drawdown(results_expanding, results_rolling,
                  os.path.join(OUTPUT_DIR, "04_drawdown.png"))

    plot_exp_vs_rolling(results_expanding, results_rolling,
                        os.path.join(OUTPUT_DIR, "05_exp_vs_rolling_auc.png"))

if holdout_preds:
    plot_holdout_summary(holdout_preds, y_ho, holdout_metrics,
                         os.path.join(OUTPUT_DIR, "06_holdout_summary.png"))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — SAVE RESULTS CSV
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Saving per-fold metrics CSV...")
rows = []
for results, wtype in [(results_expanding, "expanding"),
                        (results_rolling,   "rolling")]:
    for r in results:
        for model_name, m in r.metrics.items():
            rows.append({
                "window_type": wtype,
                "fold_id":     r.fold_id,
                "model":       model_name,
                "auc":         m.get("auc",      np.nan),
                "accuracy":    m.get("accuracy",  np.nan),
                "logloss":     m.get("logloss",   np.nan),
                "val_start":   str(r.test_start if hasattr(r, "test_start") else ""),
            })

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "fold_metrics.csv"), index=False)
print(f"    Saved fold_metrics.csv ({len(metrics_df)} rows)")

print("\n" + "═" * 80)
print("  PIPELINE COMPLETE")
print(f"  All outputs saved to: {OUTPUT_DIR}")
print("═" * 80 + "\n")
