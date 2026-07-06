"""
validate_vol_regime.py
═══════════════════════════════════════════════════════════════════════════════
Volatility Regime Indicator — Dual-Asset Validation
Assets  : NAS100 (NQ) and GOLD (XAUUSD)
Timeframe: 1H bars

Pipeline per asset
  1. Load & merge raw data into clean 1H OHLCV
     - NQ   : 03_Data/raw/NAS100/1h_data.csv  (MT5 tab-delimited)
     - GOLD : 03_Data/raw/GOLD_XAUUSD/DAT_MT_XAUUSD_M1_20xx.csv (10 years)
              concat all years → resample M1 → 1H on the fly

  2. Engineer features  (build_features from feature_engineering.py)
     Macro features (VIX/DXY/TNX/TIPS/COT) are attempted; if the network
     fetch fails the pipeline falls back to pure volatility math features.
     The script reports clearly which mode it ran in.

  3. Build vol-regime labels  (build_vol_regime_labels)
     top-30% future RV → HIGH (1),  bottom-30% → LOW (0), middle discarded.

  4. Purged Walk-Forward Validation  (PurgedWalkForwardSplit)
     Expanding AND rolling windows, 1-month step, 15-month holdout.
     Both LightGBM (Optuna HPO, 20 trials) and a Logistic Regression baseline.

  5. Permutation null test  (5 shuffles, same model) to confirm signal is real.

  6. Outputs
     - Rich terminal summary table  (per-fold + aggregate)
     - results/validation/  directory
         fold_metrics_NQ.csv
         fold_metrics_GOLD.csv
         summary_comparison.csv
         01_auc_heatmap_NQ.png
         02_auc_heatmap_GOLD.png
         03_calibration_NQ.png
         04_calibration_GOLD.png
         05_exp_vs_rolling_NQ.png
         06_exp_vs_rolling_GOLD.png
         07_null_test_NQ.png
         08_null_test_GOLD.png
         09_holdout_NQ.png
         10_holdout_GOLD.png
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, glob, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Local src modules ──────────────────────────────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)

from feature_engineering import (
    load_mt5_csv,
    build_features,
    build_vol_regime_labels,
)
from walk_forward import PurgedWalkForwardSplit

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT   = "/Users/macos/Documents/ALGO/03_Data/raw"
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "results", "validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NQ_1H_FILE    = os.path.join(DATA_ROOT, "NAS100",      "1h_data.csv")
GOLD_M1_GLOB  = os.path.join(DATA_ROOT, "GOLD_XAUUSD", "DAT_MT_XAUUSD_M1_*.csv")
NEWS_FILE     = os.path.join(DATA_ROOT, "NAS100",       "ff_news_dates.json")

CFG = {
    # Walk-Forward
    "min_train_months":      18,
    "embargo_bars":          40,       # ~1 week of 1H bars
    "rolling_window_months": 36,       # 3-year rolling window
    "holdout_months":        15,       # final 15 months locked
    "step_months":            1,
    # Labels
    "vol_forward_bars":       4,       # 4 × 1H = 4H forward window
    "vol_bar_offset":         4,
    "vol_pct_high":          0.70,
    "vol_pct_low":           0.30,
    "vol_rolling_baseline":  480,
    # Model
    "lgbm_trials":           20,
    "n_pca_components":      20,
    # Null test
    "null_shuffles":          5,
}

# Dark-theme palette (matches existing visualization.py)
DARK_BG  = "#0d0d0d"
PANEL_BG = "#141414"
GOLD_CLR = "#FFD700"
TEAL_CLR = "#00FFCC"
PINK_CLR = "#FF6699"
BLUE_CLR = "#00BFFF"
ORNG_CLR = "#FF8C00"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_nq_1h() -> pd.DataFrame:
    """Load NAS100 1H data from MT5 tab-delimited file."""
    print(f"  [NQ] Loading {NQ_1H_FILE} ...")
    df = load_mt5_csv(NQ_1H_FILE, sep="\t")
    df.sort_index(inplace=True)
    df = df[df["Close"] > 0].copy()
    print(f"  [NQ] {len(df):,} bars | {df.index[0].date()} → {df.index[-1].date()}")
    return df


def _parse_gold_m1_file(path: str) -> pd.DataFrame:
    """
    Parse a single DAT_MT_XAUUSD_M1_YYYY.csv file.
    Format: 2024.01.01,18:00,open,high,low,close,volume  (no header)
    """
    df = pd.read_csv(
        path,
        header=None,
        names=["date", "time", "Open", "High", "Low", "Close", "Tick_Volume"],
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"],
                                    format="%Y.%m.%d %H:%M")
    df.set_index("datetime", inplace=True)
    df.drop(columns=["date", "time"], inplace=True)
    df = df[df["Close"] > 0].copy()
    return df


def load_gold_1h() -> pd.DataFrame:
    """
    Merge all per-year GOLD M1 files (2016-2025), then resample to 1H.
    This replaces the existing XAUUSD_1H.csv which was derived only from M5.
    """
    files = sorted(glob.glob(GOLD_M1_GLOB))
    if not files:
        raise FileNotFoundError(f"No GOLD M1 files found at: {GOLD_M1_GLOB}")

    print(f"  [GOLD] Merging {len(files)} M1 files ({os.path.basename(files[0][:4])} "
          f"→ {os.path.basename(files[-1])[:4]}) ...")

    chunks = []
    for f in files:
        try:
            chunks.append(_parse_gold_m1_file(f))
        except Exception as e:
            print(f"    [WARN] Skipping {os.path.basename(f)}: {e}")

    df_m1 = pd.concat(chunks).sort_index()
    df_m1 = df_m1[~df_m1.index.duplicated(keep="first")]
    print(f"  [GOLD] M1 total: {len(df_m1):,} bars")

    # Resample M1 → 1H
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Tick_Volume": "sum"}
    df_1h = df_m1.resample("1h").agg(agg).dropna(subset=["Open", "Close"])
    df_1h = df_1h[df_1h["Close"] > 0]
    print(f"  [GOLD] Resampled to 1H: {len(df_1h):,} bars "
          f"| {df_1h.index[0].date()} → {df_1h.index[-1].date()}")
    return df_1h


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE + LABEL BUILDER (per asset)
# ═════════════════════════════════════════════════════════════════════════════

def build_dataset(df_raw: pd.DataFrame,
                  asset_name: str,
                  news_path: str = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list, bool]:
    """
    Build (features, labels) for one asset.
    Returns: X_full, y_full, feature_names, macro_ok
    """
    print(f"\n  [{asset_name}] Building features ...")
    t0 = time.time()

    # Features
    try:
        feat = build_features(df_raw, news_mask_path=news_path)
        # Check if macro columns actually have data
        macro_cols = ["macro_vix", "macro_dxy", "macro_tnx", "macro_tips"]
        macro_present = all(c in feat.columns for c in macro_cols)
        macro_ok = macro_present and feat["macro_vix"].notna().sum() > 100
    except Exception as e:
        print(f"  [{asset_name}] Feature build error: {e}")
        raise

    print(f"  [{asset_name}] Features: {feat.shape[1]} columns "
          f"| Macro: {'YES' if macro_ok else 'NO (pure vol math fallback)'} "
          f"| Elapsed: {time.time()-t0:.1f}s")

    # Labels
    labels = build_vol_regime_labels(
        df_raw,
        forward_bars     = CFG["vol_forward_bars"],
        bar_offset       = CFG["vol_bar_offset"],
        regime_pct_high  = CFG["vol_pct_high"],
        regime_pct_low   = CFG["vol_pct_low"],
        rolling_baseline = CFG["vol_rolling_baseline"],
    )

    # Align on common index, drop NaN labels
    joined = pd.concat([feat, labels], axis=1).dropna(subset=["vol_regime"])
    feat_cols = list(feat.columns)
    X = joined[feat_cols].values.astype(np.float32)
    y = joined["vol_regime"].values.astype(np.int32)

    print(f"  [{asset_name}] Samples: {len(X):,} | "
          f"High-Vol: {y.mean():.2%} | Low-Vol: {1-y.mean():.2%}")

    return joined, X, y, feat_cols, macro_ok


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PREPROCESSING PIPELINE (in-fold, no leakage)
# ═════════════════════════════════════════════════════════════════════════════

def make_preprocessor(n_components: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("pca",     PCA(n_components=n_components, whiten=True)),
    ])


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL TRAINING (LightGBM + Logistic Regression)
# ═════════════════════════════════════════════════════════════════════════════

def train_lgbm(X_tr, y_tr, X_va, y_va, n_trials=20) -> lgb.Booster:
    def objective(trial):
        params = {
            "objective":          "binary",
            "metric":             "auc",
            "verbosity":          -1,
            "feature_pre_filter": False,
            "learning_rate":      trial.suggest_float("lr",       1e-3, 0.1, log=True),
            "num_leaves":         trial.suggest_int("num_leaves", 16,  128),
            "max_depth":          trial.suggest_int("max_depth",   3,    8),
            "min_child_samples":  trial.suggest_int("min_child",  20,   80),
            "subsample":          trial.suggest_float("subsample", 0.5,  1.0),
            "colsample_bytree":   trial.suggest_float("col_frac",  0.5,  1.0),
            "reg_alpha":          trial.suggest_float("alpha",    1e-4, 5.0, log=True),
            "reg_lambda":         trial.suggest_float("lambda",   1e-4, 5.0, log=True),
        }
        n_est = trial.suggest_int("n_est", 100, 500)
        dtr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        dva = lgb.Dataset(X_va, label=y_va, free_raw_data=False, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=n_est,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(40, verbose=False),
                                 lgb.log_evaluation(-1)])
        pred = m.predict(X_va)
        return roc_auc_score(y_va, pred) if len(np.unique(y_va)) > 1 else 0.5

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params.copy()
    n_est = best.pop("n_est")
    best.update({"objective": "binary", "metric": "auc", "verbosity": -1,
                 "feature_pre_filter": False})
    # rename trial keys back
    best["learning_rate"]     = best.pop("lr",        best.get("learning_rate", 0.05))
    best["min_child_samples"] = best.pop("min_child", best.get("min_child_samples", 20))
    best["colsample_bytree"]  = best.pop("col_frac",  best.get("colsample_bytree", 0.8))
    best["reg_alpha"]         = best.pop("alpha",     best.get("reg_alpha", 0.1))
    best["reg_lambda"]        = best.pop("lambda",    best.get("reg_lambda", 0.1))

    dtr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, free_raw_data=False, reference=dtr)
    model = lgb.train(best, dtr, num_boost_round=n_est,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(40, verbose=False),
                                 lgb.log_evaluation(-1)])
    return model


def train_lr(X_tr, y_tr) -> LogisticRegression:
    lr = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", solver="lbfgs")
    lr.fit(X_tr, y_tr)
    return lr


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — WALK-FORWARD LOOP
# ═════════════════════════════════════════════════════════════════════════════

def run_wfv(asset_name: str,
            joined: pd.DataFrame,
            X_full: np.ndarray,
            y_full: np.ndarray) -> tuple[list, list, np.ndarray, np.ndarray]:
    """
    Run purged WFV for one asset.
    Returns: results_exp, results_rol, holdout_idx, y_holdout_labels
    """
    full_index = joined.index

    wfv = PurgedWalkForwardSplit(
        index                 = full_index,
        min_train_months      = CFG["min_train_months"],
        embargo_bars          = CFG["embargo_bars"],
        rolling_window_months = CFG["rolling_window_months"],
        step_months           = CFG["step_months"],
        holdout_months        = CFG["holdout_months"],
    )

    wfv.print_fold_summary("expanding")

    holdout_idx = wfv.holdout_idx()
    n_pca = min(CFG["n_pca_components"], X_full.shape[1] - 1)

    results_expanding = []
    results_rolling   = []

    for window_type, results_list in [("expanding", results_expanding),
                                      ("rolling",   results_rolling)]:
        print(f"\n  [{asset_name}] ── {window_type.upper()} WINDOW ──")
        folds = list(wfv.generate_folds(window_type))
        print(f"  [{asset_name}] {len(folds)} folds to run\n")

        for fold in folds:
            t0 = time.time()

            X_tr = X_full[fold.train_idx]
            y_tr = y_full[fold.train_idx]
            X_va = X_full[fold.val_idx]
            y_va = y_full[fold.val_idx]
            X_te = X_full[fold.test_idx]
            y_te = y_full[fold.test_idx]

            if len(np.unique(y_va)) < 2 or len(np.unique(y_te)) < 2:
                print(f"  [Fold {fold.fold_id}] Skipped — single class in val/test")
                continue

            # Sanitize: replace ±inf with NaN before preprocessing (imputer handles NaN, not inf)
            X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
            X_va = np.where(np.isinf(X_va), np.nan, X_va)
            X_te = np.where(np.isinf(X_te), np.nan, X_te)

            # Preprocessing (fit on train only)
            prep = make_preprocessor(n_pca)
            X_tr_p = prep.fit_transform(X_tr)
            X_va_p = prep.transform(X_va)
            X_te_p = prep.transform(X_te)

            # LightGBM
            lgbm = train_lgbm(X_tr_p, y_tr, X_va_p, y_va, CFG["lgbm_trials"])
            p_lgbm_te = lgbm.predict(X_te_p)
            p_lgbm_va = lgbm.predict(X_va_p)

            # Logistic Regression
            lr_m = train_lr(X_tr_p, y_tr)
            p_lr_te = lr_m.predict_proba(X_te_p)[:, 1]
            p_lr_va = lr_m.predict_proba(X_va_p)[:, 1]

            # Simple ensemble (average)
            p_ens_te = (p_lgbm_te + p_lr_te) / 2
            p_ens_va = (p_lgbm_va + p_lr_va) / 2

            def safe_metrics(y_true, y_pred):
                try:
                    return {
                        "auc":     roc_auc_score(y_true, y_pred),
                        "brier":   brier_score_loss(y_true, y_pred),
                        "logloss": log_loss(y_true, y_pred),
                        "acc":     accuracy_score(y_true, (y_pred > 0.5).astype(int)),
                    }
                except Exception:
                    return {"auc": np.nan, "brier": np.nan, "logloss": np.nan, "acc": np.nan}

            result = {
                "fold_id":     fold.fold_id,
                "window_type": window_type,
                "test_start":  fold.test_start,
                "test_end":    fold.test_end,
                "train_n":     len(X_tr),
                "test_n":      len(X_te),
                "y_te":        y_te,
                "p_lgbm":      p_lgbm_te,
                "p_lr":        p_lr_te,
                "p_ens":       p_ens_te,
                "metrics": {
                    "LGBM":     safe_metrics(y_te, p_lgbm_te),
                    "LR":       safe_metrics(y_te, p_lr_te),
                    "Ensemble": safe_metrics(y_te, p_ens_te),
                },
                "val_metrics": {
                    "LGBM":     safe_metrics(y_va, p_lgbm_va),
                    "LR":       safe_metrics(y_va, p_lr_va),
                    "Ensemble": safe_metrics(y_va, p_ens_va),
                },
                "feature_importance": pd.Series(
                    lgbm.feature_importance("gain"),
                    name=f"fold_{fold.fold_id}"
                ),
            }
            results_list.append(result)

            meta_auc = result["metrics"]["Ensemble"]["auc"]
            elapsed  = time.time() - t0
            print(f"  [{asset_name}|{window_type[:3].upper()}] "
                  f"Fold {fold.fold_id:02d} | "
                  f"Train {len(X_tr):,} → OOS {len(X_te):,} | "
                  f"LGBM AUC: {result['metrics']['LGBM']['auc']:.4f} | "
                  f"Ens AUC: {meta_auc:.4f} | "
                  f"Brier: {result['metrics']['Ensemble']['brier']:.4f} | "
                  f"{elapsed:.1f}s")

    return results_expanding, results_rolling, holdout_idx


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — HOLDOUT EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_holdout(asset_name: str,
                     X_full: np.ndarray,
                     y_full: np.ndarray,
                     holdout_idx: np.ndarray,
                     results_exp: list) -> dict:
    """Train on all non-holdout data, evaluate on the locked holdout zone."""
    print(f"\n  [{asset_name}] Holdout evaluation ...")
    n_pca     = min(CFG["n_pca_components"], X_full.shape[1] - 1)
    ho_set    = set(holdout_idx)
    train_idx = np.array([i for i in range(len(X_full)) if i not in ho_set])

    X_tr = np.where(np.isinf(X_full[train_idx]), np.nan, X_full[train_idx])
    y_tr = y_full[train_idx]
    X_ho = np.where(np.isinf(X_full[holdout_idx]), np.nan, X_full[holdout_idx])
    y_ho = y_full[holdout_idx]

    prep   = make_preprocessor(n_pca)
    X_tr_p = prep.fit_transform(X_tr)
    X_ho_p = prep.transform(X_ho)

    lgbm   = train_lgbm(X_tr_p, y_tr, X_ho_p, y_ho, n_trials=20)
    lr_m   = train_lr(X_tr_p, y_tr)

    p_lgbm = lgbm.predict(X_ho_p)
    p_lr   = lr_m.predict_proba(X_ho_p)[:, 1]
    p_ens  = (p_lgbm + p_lr) / 2

    def safe_m(y_true, y_pred):
        try:
            return {
                "auc":     roc_auc_score(y_true, y_pred),
                "brier":   brier_score_loss(y_true, y_pred),
                "logloss": log_loss(y_true, y_pred),
                "acc":     accuracy_score(y_true, (y_pred > 0.5).astype(int)),
            }
        except Exception:
            return {"auc": np.nan, "brier": np.nan, "logloss": np.nan, "acc": np.nan}

    ho_results = {
        "y_ho":   y_ho,
        "p_lgbm": p_lgbm,
        "p_lr":   p_lr,
        "p_ens":  p_ens,
        "metrics": {
            "LGBM":     safe_m(y_ho, p_lgbm),
            "LR":       safe_m(y_ho, p_lr),
            "Ensemble": safe_m(y_ho, p_ens),
        },
    }

    for name, m in ho_results["metrics"].items():
        print(f"  [{asset_name}] Holdout {name:10s} → "
              f"AUC: {m['auc']:.4f} | Brier: {m['brier']:.4f} | "
              f"Acc: {m['acc']:.4f}")

    return ho_results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PERMUTATION NULL TEST
# ═════════════════════════════════════════════════════════════════════════════

def permutation_null_test(asset_name: str,
                          X_full: np.ndarray,
                          y_full: np.ndarray,
                          n_shuffles: int = 5) -> list:
    """Shuffle labels, refit LightGBM, record AUC to build null distribution."""
    print(f"\n  [{asset_name}] Permutation null test ({n_shuffles} shuffles) ...")
    n     = len(X_full)
    split = int(n * 0.75)
    n_pca = min(CFG["n_pca_components"], X_full.shape[1] - 1)

    prep   = make_preprocessor(n_pca)
    X_tr_p = prep.fit_transform(np.where(np.isinf(X_full[:split]), np.nan, X_full[:split]))
    X_te_p = prep.transform(np.where(np.isinf(X_full[split:]), np.nan, X_full[split:]))
    y_te   = y_full[split:]

    null_aucs = []
    for i in range(n_shuffles):
        rng   = np.random.default_rng(seed=i + 99)
        y_shuf = y_full[:split].copy()
        rng.shuffle(y_shuf)

        if len(np.unique(y_shuf)) < 2:
            null_aucs.append(0.5)
            continue

        params = {"objective": "binary", "metric": "auc", "verbosity": -1,
                  "num_leaves": 31, "learning_rate": 0.05}
        dtr = lgb.Dataset(X_tr_p, label=y_shuf)
        m   = lgb.train(params, dtr, num_boost_round=150,
                        callbacks=[lgb.log_evaluation(-1)])
        try:
            auc = roc_auc_score(y_te, m.predict(X_te_p))
        except Exception:
            auc = 0.5
        null_aucs.append(auc)
        print(f"    Shuffle {i+1}: AUC = {auc:.4f}")

    print(f"  [{asset_name}] Null AUC mean={np.mean(null_aucs):.4f} ± {np.std(null_aucs):.4f}")
    return null_aucs


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TERMINAL SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════

def print_summary_table(asset_name: str, results_exp: list, results_rol: list,
                        ho_results: dict, null_aucs: list, macro_ok: bool):
    """Print a formatted summary to terminal."""
    SEP = "═" * 100

    def agg(results, model):
        aucs = [r["metrics"][model]["auc"] for r in results
                if not np.isnan(r["metrics"][model]["auc"])]
        briers = [r["metrics"][model]["brier"] for r in results
                  if not np.isnan(r["metrics"][model]["brier"])]
        return (np.mean(aucs), np.std(aucs), np.mean(briers)) if aucs else (np.nan, np.nan, np.nan)

    print(f"\n{SEP}")
    print(f"  VOLATILITY REGIME — VALIDATION REPORT : {asset_name}")
    print(f"  Macro Features : {'INCLUDED (VIX / DXY / TNX / TIPS / COT)' if macro_ok else 'EXCLUDED (network fetch failed — pure vol math)'}")
    print(SEP)

    header = f"  {'Window':<12} {'Model':<12} {'Folds':>5} {'Mean AUC':>10} {'Std AUC':>9} {'Mean Brier':>11}"
    print(header)
    print(f"  {'─'*98}")

    for wtype, results in [("Expanding", results_exp), ("Rolling", results_rol)]:
        for model in ["LGBM", "LR", "Ensemble"]:
            mean_auc, std_auc, mean_brier = agg(results, model)
            tag = " ← PRIMARY" if model == "Ensemble" and wtype == "Expanding" else ""
            print(f"  {wtype:<12} {model:<12} {len(results):>5} "
                  f"{mean_auc:>10.4f} {std_auc:>9.4f} {mean_brier:>11.4f}{tag}")
        print()

    print(f"  {'─'*98}")
    print(f"  HOLDOUT (locked {CFG['holdout_months']}m)")
    for model, m in ho_results["metrics"].items():
        print(f"  {'':12} {model:<12} {'—':>5} {m['auc']:>10.4f} {'—':>9} {m['brier']:>11.4f}")

    print(f"\n  {'─'*98}")
    null_mean = np.mean(null_aucs)
    null_std  = np.std(null_aucs)
    best_real = agg(results_exp, "Ensemble")[0]
    z_score   = (best_real - null_mean) / (null_std + 1e-9) if null_std > 0 else np.nan
    verdict   = "SIGNAL REAL (p < 0.05)" if z_score > 1.645 else "INCONCLUSIVE"
    print(f"  PERMUTATION NULL TEST — Mean AUC: {null_mean:.4f} ± {null_std:.4f} | "
          f"Real AUC: {best_real:.4f} | Z = {z_score:.2f} | {verdict}")
    print(SEP + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════

def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[:].set_color("#333333")
    ax.grid(True, color="#1e1e1e", linewidth=0.7, linestyle="--")
    if title:  ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)
    if xlabel: ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)


def plot_auc_heatmap(asset_name: str,
                     results_exp: list, results_rol: list, out_path: str):
    models = ["LGBM", "LR", "Ensemble"]

    def build_mat(results):
        data   = {m: [] for m in models}
        f_ids  = []
        for r in results:
            f_ids.append(f"F{r['fold_id']}")
            for m in models:
                data[m].append(r["metrics"][m]["auc"])
        return pd.DataFrame(data, index=f_ids).T

    fig, (ax1, ax2) = plt.subplots(1, 2,
                                    figsize=(max(10, len(results_exp) * 0.7 + 4), 6))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f"{asset_name} — OOS AUC per Fold", color="white",
                 fontsize=13, fontweight="bold")

    for ax, results, title in [(ax1, results_exp, "Expanding"),
                                (ax2, results_rol, "Rolling")]:
        if not results:
            ax.set_visible(False)
            continue
        mat = build_mat(results)
        _style(ax, title=title)
        sns.heatmap(mat, ax=ax, cmap="RdYlGn", vmin=0.45, vmax=0.65,
                    annot=True, fmt=".3f", linewidths=0.5,
                    cbar_kws={"label": "AUC"},
                    annot_kws={"size": 8, "color": "white"})
        ax.tick_params(colors="white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_calibration(asset_name: str,
                     results_exp: list, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)
    _style(ax, title=f"{asset_name} — Probability Calibration (OOS, Expanding)",
           xlabel="Mean Predicted Probability", ylabel="Fraction of Positives")
    ax.plot([0, 1], [0, 1], "w--", linewidth=1, label="Perfect")

    color_map = {"LGBM": TEAL_CLR, "LR": PINK_CLR, "Ensemble": GOLD_CLR}
    for model, color in color_map.items():
        all_p, all_l = [], []
        for r in results_exp:
            key = f"p_{model.lower()}"
            if key not in r:
                continue
            pred  = r[key]
            label = r["y_te"]
            valid = ~np.isnan(pred)
            if valid.sum() > 10:
                all_p.append(pred[valid])
                all_l.append(label[valid])
        if not all_p:
            continue
        probs  = np.concatenate(all_p)
        labels = np.concatenate(all_l)
        try:
            frac, mean_p = calibration_curve(labels, probs, n_bins=10)
            ax.plot(mean_p, frac, "o-", color=color, label=model, linewidth=2, markersize=5)
        except Exception:
            pass

    ax.legend(facecolor="#1e1e1e", edgecolor="none", labelcolor="white", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_exp_vs_rolling(asset_name: str,
                        results_exp: list, results_rol: list, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f"{asset_name} — Expanding vs Rolling AUC per Fold",
                 color="white", fontsize=13, fontweight="bold")

    for ax, model in zip(axes, ["LGBM", "LR", "Ensemble"]):
        _style(ax, title=model, xlabel="Fold ID", ylabel="OOS AUC")

        for results, color, label in [(results_exp, BLUE_CLR, "Expanding"),
                                       (results_rol, ORNG_CLR, "Rolling")]:
            if not results:
                continue
            fids = [r["fold_id"] for r in results]
            aucs = [r["metrics"][model]["auc"] for r in results]
            ax.plot(fids, aucs, "o-", color=color, label=label,
                    linewidth=1.8, markersize=5, alpha=0.9)
            # 3-fold rolling mean
            s = pd.Series(aucs)
            ax.plot(fids, s.rolling(3, min_periods=1).mean().values,
                    color=color, linewidth=1, linestyle="--", alpha=0.5)

        ax.axhline(0.5, color="#555555", linewidth=1, linestyle=":")
        ax.legend(facecolor="#1e1e1e", edgecolor="none",
                  labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_null_test(asset_name: str, null_aucs: list,
                   real_auc: float, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style(ax, title=f"{asset_name} — Permutation Null Test",
           xlabel="Null AUC (shuffled labels)", ylabel="Count")

    ax.hist(null_aucs, bins=max(5, len(null_aucs)),
            color=PINK_CLR, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(real_auc, color=GOLD_CLR, linewidth=2.5,
               linestyle="--", label=f"Real AUC = {real_auc:.4f}")
    ax.axvline(0.5, color="#555555", linewidth=1.2, linestyle=":",
               label="Chance (0.50)")
    ax.legend(facecolor="#1e1e1e", edgecolor="none", labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_holdout(asset_name: str, ho_results: dict, out_path: str):
    """Calibration + metric bar chart for the locked holdout zone."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f"{asset_name} — Locked Holdout Evaluation",
                 color="white", fontsize=13, fontweight="bold")

    # Left: calibration
    _style(ax1, title="Calibration",
           xlabel="Mean Predicted Prob.", ylabel="Fraction of Positives")
    ax1.plot([0, 1], [0, 1], "w--", linewidth=1, label="Perfect")

    color_map = {"p_lgbm": (TEAL_CLR, "LGBM"),
                 "p_lr":   (PINK_CLR, "LR"),
                 "p_ens":  (GOLD_CLR, "Ensemble")}
    y_ho = ho_results["y_ho"]
    for key, (color, label) in color_map.items():
        pred = ho_results.get(key, np.array([]))
        if len(pred) == 0:
            continue
        try:
            frac, mean_p = calibration_curve(y_ho, pred, n_bins=10)
            ax1.plot(mean_p, frac, "o-", color=color,
                     label=label, linewidth=2, markersize=5)
        except Exception:
            pass
    ax1.legend(facecolor="#1e1e1e", edgecolor="none", labelcolor="white")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

    # Right: AUC bar chart
    _style(ax2, title="Holdout AUC / Brier", xlabel="Model", ylabel="Value")
    models  = list(ho_results["metrics"].keys())
    aucs    = [ho_results["metrics"][m]["auc"]   for m in models]
    briers  = [ho_results["metrics"][m]["brier"] for m in models]
    x       = np.arange(len(models))
    w       = 0.35
    bars1 = ax2.bar(x - w/2, aucs,   w, label="AUC",   color=TEAL_CLR, alpha=0.85)
    bars2 = ax2.bar(x + w/2, briers, w, label="Brier", color=PINK_CLR, alpha=0.85)
    ax2.axhline(0.5, color="#555555", linewidth=1, linestyle=":")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, color="white")
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}", ha="center", va="bottom",
                 color="white", fontsize=9)
    ax2.legend(facecolor="#1e1e1e", edgecolor="none", labelcolor="white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — SAVE CSVS
# ═════════════════════════════════════════════════════════════════════════════

def save_fold_csv(asset_name: str, results_exp: list, results_rol: list) -> pd.DataFrame:
    rows = []
    for results, wtype in [(results_exp, "expanding"), (results_rol, "rolling")]:
        for r in results:
            for model in ["LGBM", "LR", "Ensemble"]:
                m = r["metrics"][model]
                rows.append({
                    "asset":       asset_name,
                    "window_type": wtype,
                    "fold_id":     r["fold_id"],
                    "model":       model,
                    "test_start":  str(r["test_start"].date()),
                    "test_end":    str(r["test_end"].date()),
                    "train_n":     r["train_n"],
                    "test_n":      r["test_n"],
                    "auc":         m["auc"],
                    "brier":       m["brier"],
                    "logloss":     m["logloss"],
                    "acc":         m["acc"],
                })
    df = pd.DataFrame(rows)
    tag  = asset_name.replace("/", "").replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"fold_metrics_{tag}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


def save_summary_csv(all_dfs: list):
    combined = pd.concat(all_dfs, ignore_index=True)
    path = os.path.join(OUTPUT_DIR, "summary_comparison.csv")
    combined.to_csv(path, index=False)
    print(f"  Saved: {path}")

    # Print a quick pivot
    pivot = combined.groupby(["asset", "window_type", "model"])["auc"].agg(["mean","std"]).round(4)
    print("\n  CROSS-ASSET AUC SUMMARY")
    print(pivot.to_string())


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run_asset(asset_name: str, df_raw: pd.DataFrame,
              news_path: str = None) -> pd.DataFrame:
    """Full pipeline for one asset. Returns fold metrics DataFrame."""
    tag = asset_name.replace("/", "").replace(" ", "_")

    # — Features & Labels
    joined, X_full, y_full, feat_cols, macro_ok = build_dataset(
        df_raw, asset_name, news_path
    )

    # — Walk-Forward Validation
    results_exp, results_rol, holdout_idx = run_wfv(
        asset_name, joined, X_full, y_full
    )

    # — Holdout
    ho_results = evaluate_holdout(
        asset_name, X_full, y_full, holdout_idx, results_exp
    )

    # — Null Test
    null_aucs  = permutation_null_test(asset_name, X_full, y_full,
                                       CFG["null_shuffles"])

    # — Terminal Summary
    print_summary_table(asset_name, results_exp, results_rol,
                        ho_results, null_aucs, macro_ok)

    # — Plots
    print(f"\n  [{asset_name}] Generating charts ...")
    plot_auc_heatmap(asset_name, results_exp, results_rol,
                     os.path.join(OUTPUT_DIR, f"01_auc_heatmap_{tag}.png"))

    plot_calibration(asset_name, results_exp,
                     os.path.join(OUTPUT_DIR, f"02_calibration_{tag}.png"))

    plot_exp_vs_rolling(asset_name, results_exp, results_rol,
                        os.path.join(OUTPUT_DIR, f"03_exp_vs_rolling_{tag}.png"))

    real_auc = np.nanmean([r["metrics"]["Ensemble"]["auc"] for r in results_exp])
    plot_null_test(asset_name, null_aucs, real_auc,
                   os.path.join(OUTPUT_DIR, f"04_null_test_{tag}.png"))

    plot_holdout(asset_name, ho_results,
                 os.path.join(OUTPUT_DIR, f"05_holdout_{tag}.png"))

    # — CSV
    df_metrics = save_fold_csv(asset_name, results_exp, results_rol)
    return df_metrics


def main():
    START = time.time()

    print("\n" + "═" * 100)
    print("  VOLATILITY REGIME INDICATOR — DUAL-ASSET WALK-FORWARD VALIDATION")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("═" * 100)

    all_metric_dfs = []

    # ── ASSET 1: NAS100 ──────────────────────────────────────────────────────
    print("\n" + "─" * 100)
    print("  ASSET 1 of 2 : NAS100 (NQ)")
    print("─" * 100)
    try:
        df_nq = load_nq_1h()
        df_nq_metrics = run_asset(
            asset_name = "NAS100",
            df_raw     = df_nq,
            news_path  = NEWS_FILE if os.path.exists(NEWS_FILE) else None,
        )
        all_metric_dfs.append(df_nq_metrics)
    except Exception as e:
        print(f"  [ERROR] NAS100 pipeline failed: {e}")
        import traceback; traceback.print_exc()

    # ── ASSET 2: GOLD ─────────────────────────────────────────────────────────
    print("\n" + "─" * 100)
    print("  ASSET 2 of 2 : GOLD (XAUUSD)")
    print("─" * 100)
    try:
        df_gold = load_gold_1h()
        df_gold_metrics = run_asset(
            asset_name = "GOLD",
            df_raw     = df_gold,
            news_path  = None,   # No dedicated news file for GOLD
        )
        all_metric_dfs.append(df_gold_metrics)
    except Exception as e:
        print(f"  [ERROR] GOLD pipeline failed: {e}")
        import traceback; traceback.print_exc()

    # ── CROSS-ASSET SUMMARY ───────────────────────────────────────────────────
    if all_metric_dfs:
        print("\n" + "═" * 100)
        print("  CROSS-ASSET SUMMARY")
        print("═" * 100)
        save_summary_csv(all_metric_dfs)

    elapsed = time.time() - START
    print(f"\n  Total runtime: {elapsed/60:.1f} minutes")
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print("═" * 100 + "\n")


if __name__ == "__main__":
    main()
