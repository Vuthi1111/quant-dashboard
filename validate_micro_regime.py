"""
validate_micro_regime.py
═══════════════════════════════════════════════════════════════════════════════
Micro-Regime — 1M → 15M Walk-Forward Validation
Assets   : NAS100 (NQ) and GOLD (XAUUSD)
Timeframe : 1M bars → predict next 15 minutes (Fast vs Slow tape)

What this validates
-------------------
  Can we predict, at every 1-MINUTE bar, whether the NEXT 15 MINUTES will
  be a FAST TAPE or SLOW TAPE regime?

  Fast Tape (label=1) : top 30% of forward 15-min active_ratio
  Slow Tape (label=0) : bottom 30% of forward 15-min active_ratio
  Middle 40%          : discarded (no clear edge)

Use-cases validated
-------------------
  • ORB / Momentum scalpers — skip entries in predicted SLOW windows
  • VWAP execution algos    — pause during SLOW, execute during FAST
  • Short-duration option strategies — regime-filter on gamma exposure

Compute Optimisations vs 4H validation
---------------------------------------
  • GOLD subsampled 1-in-2 (systematic, preserves temporal order)
  • LightGBM HPO: 10 Optuna trials (vs 20) — folds are plentiful
  • PCA: 25 components (slightly higher — more raw features)
  • Embargo: 30 bars = 30 minutes (vs 64 bars of 15M = 16 hours)
  • Rolling baseline window: 24 months (same)
  • Null test: 5 shuffles (same)

Outputs
-------
  results/micro_regime/
    fold_metrics_NAS100.csv
    fold_metrics_GOLD.csv
    summary_comparison.csv
    01_auc_heatmap_{asset}.png
    02_calibration_{asset}.png
    03_exp_vs_rolling_{asset}.png
    04_null_test_{asset}.png
    05_holdout_{asset}.png
    06_micro_feature_importance_{asset}.png
    run.log
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                              log_loss, accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── local modules ──────────────────────────────────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from walk_forward import PurgedWalkForwardSplit
from micro_regime_features import (
    load_nq_1m, load_gold_1m,
    build_micro_dataset,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

NQ_1M_PATH   = "/Users/macos/Documents/ALGO/03_Data/raw/NAS100/1m_data.csv"
GOLD_1M_PATH = "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/XAUUSD_M1.csv"
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "results", "micro_regime")
LOG_FILE     = os.path.join(OUTPUT_DIR, "run.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CFG = {
    # Label (1M → 15M prediction)
    "forward_bars":       15,    # 15 x 1M = 15 minutes ahead
    "bar_offset":          1,    # 1-bar buffer
    "regime_pct_high":    0.70,
    "regime_pct_low":     0.30,
    "rolling_baseline":  480,    # 480 x 1M = 8 hours (adaptive threshold)
    # Subsampling (systematic, preserves order)
    "nq_subsample":       1.0,   # NAS100: full (3M rows, manageable)
    "gold_subsample":     0.5,   # GOLD: 1-in-2 (5M → ~2.5M labeled rows)
    # Walk-Forward
    "min_train_months":   12,
    "embargo_bars":       30,    # 30 x 1M = 30-minute gap (prevents leakage)
    "rolling_window_months": 24,
    "holdout_months":     12,
    "step_months":         1,
    # Model
    "lgbm_trials":        10,    # fewer trials — many folds compensate
    "n_pca_components":   25,
    # Null test
    "null_shuffles":       5,
}

# Dark theme colours
DARK_BG  = "#0d0d0d"
PANEL_BG = "#141414"
GOLD_CLR = "#FFD700"
TEAL_CLR = "#00FFCC"
PINK_CLR = "#FF6699"
BLUE_CLR = "#00BFFF"
ORNG_CLR = "#FF8C00"
GRNN_CLR = "#39FF14"


# ─────────────────────────────────────────────────────────────────────────────
# LIVE LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log      = open(log_path, "w", buffering=1)

    def write(self, msg):
        self.terminal.write(msg)
        self.terminal.flush()
        self.log.write(msg)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def make_preprocessor(n_components: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("pca",     PCA(n_components=n_components, whiten=True)),
    ])


def sanitize(X: np.ndarray) -> np.ndarray:
    return np.where(np.isinf(X), np.nan, X)


# ─────────────────────────────────────────────────────────────────────────────
# LGBM — optimised for 1M data volume
# ─────────────────────────────────────────────────────────────────────────────

def train_lgbm(X_tr, y_tr, X_va, y_va, n_trials=10) -> lgb.Booster:
    def objective(trial):
        params = {
            "objective":          "binary",
            "metric":             "auc",
            "verbosity":          -1,
            "feature_pre_filter": False,
            "learning_rate":      trial.suggest_float("lr",       5e-3, 0.15, log=True),
            "num_leaves":         trial.suggest_int("num_leaves", 16,   96),
            "max_depth":          trial.suggest_int("max_depth",   3,    7),
            "min_child_samples":  trial.suggest_int("min_child",  50,  200),  # higher for big data
            "subsample":          trial.suggest_float("subsample", 0.5,  1.0),
            "colsample_bytree":   trial.suggest_float("col_frac",  0.5,  1.0),
            "reg_alpha":          trial.suggest_float("alpha",    1e-4, 5.0, log=True),
            "reg_lambda":         trial.suggest_float("lam",      1e-4, 5.0, log=True),
            # Speed: use histogram-based binning
            "max_bin":            127,
        }
        n_est = trial.suggest_int("n_est", 100, 400)
        dtr   = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        dva   = lgb.Dataset(X_va, label=y_va, free_raw_data=False, reference=dtr)
        m     = lgb.train(params, dtr, num_boost_round=n_est,
                          valid_sets=[dva],
                          callbacks=[lgb.early_stopping(30, verbose=False),
                                     lgb.log_evaluation(-1)])
        pred = m.predict(X_va)
        return roc_auc_score(y_va, pred) if len(np.unique(y_va)) > 1 else 0.5

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best  = study.best_params.copy()
    n_est = best.pop("n_est")
    best["learning_rate"]     = best.pop("lr")
    best["min_child_samples"] = best.pop("min_child")
    best["colsample_bytree"]  = best.pop("col_frac")
    best["reg_alpha"]         = best.pop("alpha")
    best["reg_lambda"]        = best.pop("lam")
    best.update({"objective": "binary", "metric": "auc",
                 "verbosity": -1, "feature_pre_filter": False, "max_bin": 127})

    dtr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, free_raw_data=False, reference=dtr)
    return lgb.train(best, dtr, num_boost_round=n_est,
                     valid_sets=[dva],
                     callbacks=[lgb.early_stopping(30, verbose=False),
                                lgb.log_evaluation(-1)])


def train_lr(X_tr, y_tr) -> LogisticRegression:
    lr = LogisticRegression(C=0.1, max_iter=1000,
                             class_weight="balanced", solver="lbfgs")
    lr.fit(X_tr, y_tr)
    return lr


def safe_metrics(y_true, y_pred) -> dict:
    try:
        return {
            "auc":     roc_auc_score(y_true, y_pred),
            "brier":   brier_score_loss(y_true, y_pred),
            "logloss": log_loss(y_true, y_pred),
            "acc":     accuracy_score(y_true, (y_pred > 0.5).astype(int)),
        }
    except Exception:
        return {"auc": np.nan, "brier": np.nan, "logloss": np.nan, "acc": np.nan}


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_wfv(asset_name: str,
            joined: pd.DataFrame,
            X_full: np.ndarray,
            y_full: np.ndarray) -> tuple[list, list, np.ndarray]:

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

    results_exp = []
    results_rol = []

    for window_type, results_list in [("expanding", results_exp),
                                      ("rolling",   results_rol)]:
        folds = list(wfv.generate_folds(window_type))
        print(f"\n  [{asset_name}] ── {window_type.upper()} WINDOW "
              f"({len(folds)} folds) ──")

        feat_importances = []

        for fold in folds:
            t0 = time.time()

            X_tr = sanitize(X_full[fold.train_idx])
            y_tr = y_full[fold.train_idx]
            X_va = sanitize(X_full[fold.val_idx])
            y_va = y_full[fold.val_idx]
            X_te = sanitize(X_full[fold.test_idx])
            y_te = y_full[fold.test_idx]

            if len(np.unique(y_va)) < 2 or len(np.unique(y_te)) < 2:
                print(f"  [Fold {fold.fold_id:02d}] Skipped — single class")
                continue

            prep   = make_preprocessor(n_pca)
            X_tr_p = prep.fit_transform(X_tr)
            X_va_p = prep.transform(X_va)
            X_te_p = prep.transform(X_te)

            lgbm_m = train_lgbm(X_tr_p, y_tr, X_va_p, y_va, CFG["lgbm_trials"])
            lr_m   = train_lr(X_tr_p, y_tr)

            p_lgbm_te = lgbm_m.predict(X_te_p)
            p_lr_te   = lr_m.predict_proba(X_te_p)[:, 1]
            p_ens_te  = (p_lgbm_te + p_lr_te) / 2

            p_lgbm_va = lgbm_m.predict(X_va_p)
            p_lr_va   = lr_m.predict_proba(X_va_p)[:, 1]
            p_ens_va  = (p_lgbm_va + p_lr_va) / 2

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
                # Feature importance in PCA space — store raw lgbm gain
                "feature_importance": lgbm_m.feature_importance("gain"),
            }
            results_list.append(result)
            feat_importances.append(result["feature_importance"])

            elapsed = time.time() - t0
            m = result["metrics"]
            print(f"  [{asset_name}|{window_type[:3].upper()}] "
                  f"F{fold.fold_id:02d} "
                  f"{str(fold.test_start.date())} → {str(fold.test_end.date())} | "
                  f"Train {len(X_tr):,} → OOS {len(X_te):,} | "
                  f"LGBM: {m['LGBM']['auc']:.4f} | "
                  f"LR: {m['LR']['auc']:.4f} | "
                  f"Ens: {m['Ensemble']['auc']:.4f} | "
                  f"Brier: {m['Ensemble']['brier']:.4f} | "
                  f"{elapsed:.1f}s")

    return results_exp, results_rol, holdout_idx


# ─────────────────────────────────────────────────────────────────────────────
# HOLDOUT EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_holdout(asset_name, X_full, y_full, holdout_idx) -> dict:
    print(f"\n  [{asset_name}] Holdout evaluation ...")
    n_pca     = min(CFG["n_pca_components"], X_full.shape[1] - 1)
    ho_set    = set(holdout_idx)
    train_idx = np.array([i for i in range(len(X_full)) if i not in ho_set])

    X_tr = sanitize(X_full[train_idx]);    y_tr = y_full[train_idx]
    X_ho = sanitize(X_full[holdout_idx]);  y_ho = y_full[holdout_idx]

    prep   = make_preprocessor(n_pca)
    X_tr_p = prep.fit_transform(X_tr)
    X_ho_p = prep.transform(X_ho)

    lgbm_m = train_lgbm(X_tr_p, y_tr, X_ho_p, y_ho, n_trials=10)
    lr_m   = train_lr(X_tr_p, y_tr)

    p_lgbm = lgbm_m.predict(X_ho_p)
    p_lr   = lr_m.predict_proba(X_ho_p)[:, 1]
    p_ens  = (p_lgbm + p_lr) / 2

    ho = {
        "y_ho":   y_ho,
        "p_lgbm": p_lgbm,
        "p_lr":   p_lr,
        "p_ens":  p_ens,
        "metrics": {
            "LGBM":     safe_metrics(y_ho, p_lgbm),
            "LR":       safe_metrics(y_ho, p_lr),
            "Ensemble": safe_metrics(y_ho, p_ens),
        },
    }
    for name, m in ho["metrics"].items():
        print(f"  [{asset_name}] Holdout {name:10s} → "
              f"AUC: {m['auc']:.4f} | Brier: {m['brier']:.4f} | "
              f"Acc: {m['acc']:.4f}")
    return ho


# ─────────────────────────────────────────────────────────────────────────────
# PERMUTATION NULL TEST
# ─────────────────────────────────────────────────────────────────────────────

def permutation_null_test(asset_name, X_full, y_full, n_shuffles=5) -> list:
    print(f"\n  [{asset_name}] Permutation null test ({n_shuffles} shuffles) ...")
    n     = len(X_full)
    split = int(n * 0.75)
    n_pca = min(CFG["n_pca_components"], X_full.shape[1] - 1)

    prep   = make_preprocessor(n_pca)
    X_tr_p = prep.fit_transform(sanitize(X_full[:split]))
    X_te_p = prep.transform(sanitize(X_full[split:]))
    y_te   = y_full[split:]

    null_aucs = []
    for i in range(n_shuffles):
        rng    = np.random.default_rng(seed=i + 99)
        y_shuf = y_full[:split].copy()
        rng.shuffle(y_shuf)
        if len(np.unique(y_shuf)) < 2:
            null_aucs.append(0.5); continue

        params = {"objective": "binary", "metric": "auc", "verbosity": -1,
                  "num_leaves": 31, "learning_rate": 0.05, "max_bin": 127}
        dtr = lgb.Dataset(X_tr_p, label=y_shuf)
        m   = lgb.train(params, dtr, num_boost_round=150,
                        callbacks=[lgb.log_evaluation(-1)])
        try:    auc = roc_auc_score(y_te, m.predict(X_te_p))
        except: auc = 0.5
        null_aucs.append(auc)
        print(f"    Shuffle {i+1}: AUC = {auc:.4f}")

    print(f"  [{asset_name}] Null mean={np.mean(null_aucs):.4f} "
          f"± {np.std(null_aucs):.4f}")
    return null_aucs


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(asset_name, results_exp, results_rol, ho, null_aucs):
    SEP = "═" * 100
    def agg(results, model):
        aucs   = [r["metrics"][model]["auc"]   for r in results
                  if not np.isnan(r["metrics"][model]["auc"])]
        briers = [r["metrics"][model]["brier"] for r in results
                  if not np.isnan(r["metrics"][model]["brier"])]
        return (np.mean(aucs), np.std(aucs), np.mean(briers)) if aucs else (np.nan, np.nan, np.nan)

    print(f"\n{SEP}")
    print(f"  MICRO-REGIME (1M→15M) — VALIDATION REPORT : {asset_name}")
    print(SEP)
    print(f"  {'Window':<12} {'Model':<12} {'Folds':>5} "
          f"{'Mean AUC':>10} {'Std':>8} {'Mean Brier':>11}")
    print(f"  {'─'*98}")

    for wtype, results in [("Expanding", results_exp), ("Rolling", results_rol)]:
        for model in ["LGBM", "LR", "Ensemble"]:
            mu, sd, br = agg(results, model)
            tag = " ← PRIMARY" if model == "Ensemble" and wtype == "Expanding" else ""
            print(f"  {wtype:<12} {model:<12} {len(results):>5} "
                  f"{mu:>10.4f} {sd:>8.4f} {br:>11.4f}{tag}")
        print()

    print(f"  {'─'*98}")
    print(f"  HOLDOUT ({CFG['holdout_months']}m locked)")
    for model, m in ho["metrics"].items():
        print(f"  {'':12} {model:<12} {'—':>5} {m['auc']:>10.4f} "
              f"{'—':>8} {m['brier']:>11.4f}")

    real_auc  = agg(results_exp, "Ensemble")[0]
    null_mean = np.mean(null_aucs)
    null_std  = np.std(null_aucs)
    z         = (real_auc - null_mean) / (null_std + 1e-9)
    verdict   = "SIGNAL REAL (p<0.05)" if z > 1.645 else "INCONCLUSIVE"
    print(f"\n  NULL TEST — Real: {real_auc:.4f} | "
          f"Null: {null_mean:.4f}±{null_std:.4f} | Z={z:.2f} | {verdict}")
    print(SEP + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[:].set_color("#333333")
    ax.grid(True, color="#1e1e1e", linewidth=0.7, linestyle="--")
    if title:  ax.set_title(title, color="white", fontsize=11, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)


def plot_auc_heatmap(asset, results_exp, results_rol, out):
    models = ["LGBM", "LR", "Ensemble"]
    def mat(results):
        d = {m: [] for m in models}; idx = []
        for r in results:
            idx.append(f"F{r['fold_id']}")
            for m in models:
                d[m].append(r["metrics"][m]["auc"])
        return pd.DataFrame(d, index=idx).T

    n_folds = max(len(results_exp), len(results_rol), 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, n_folds * 0.4 + 4), 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f"{asset} — Micro-Regime OOS AUC per Fold (1M→15M)",
                 color="white", fontsize=13, fontweight="bold")
    for ax, results, title in [(ax1, results_exp, "Expanding"),
                                (ax2, results_rol, "Rolling")]:
        if not results: ax.set_visible(False); continue
        _style(ax, title=title)
        sns.heatmap(mat(results), ax=ax, cmap="RdYlGn", vmin=0.45, vmax=0.70,
                    annot=True, fmt=".3f", linewidths=0.5,
                    annot_kws={"size": 6, "color": "white"})
        ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_calibration(asset, results_exp, out):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)
    _style(ax, title=f"{asset} — Calibration (OOS Expanding) | Micro-Regime",
           xlabel="Mean Predicted Prob", ylabel="Fraction of Positives")
    ax.plot([0,1],[0,1],"w--",linewidth=1,label="Perfect")
    color_map = {"p_lgbm": (TEAL_CLR,"LGBM"), "p_lr": (PINK_CLR,"LR"),
                 "p_ens":  (GOLD_CLR,"Ensemble")}
    for key, (color, label) in color_map.items():
        all_p, all_l = [], []
        for r in results_exp:
            p, l = r.get(key, np.array([])), r["y_te"]
            v = ~np.isnan(p)
            if v.sum() > 10:
                all_p.append(p[v]); all_l.append(l[v])
        if not all_p: continue
        try:
            frac, mean_p = calibration_curve(np.concatenate(all_l),
                                              np.concatenate(all_p), n_bins=10)
            ax.plot(mean_p, frac, "o-", color=color, label=label, linewidth=2, markersize=5)
        except Exception: pass
    ax.legend(facecolor="#1e1e1e", edgecolor="none", labelcolor="white")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_exp_vs_rolling(asset, results_exp, results_rol, out):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f"{asset} — Expanding vs Rolling AUC | Micro-Regime (1M→15M)",
                 color="white", fontsize=13, fontweight="bold")
    for ax, model in zip(axes, ["LGBM", "LR", "Ensemble"]):
        _style(ax, title=model, xlabel="Fold ID", ylabel="OOS AUC")
        for results, color, label in [(results_exp, BLUE_CLR, "Expanding"),
                                       (results_rol, ORNG_CLR, "Rolling")]:
            if not results: continue
            fids = [r["fold_id"] for r in results]
            aucs = [r["metrics"][model]["auc"] for r in results]
            ax.plot(fids, aucs, "o-", color=color, label=label,
                    linewidth=1.8, markersize=3, alpha=0.9)
            ax.plot(fids, pd.Series(aucs).rolling(5, min_periods=1).mean().values,
                    color=color, linewidth=1.5, linestyle="--", alpha=0.6)
        ax.axhline(0.5, color="#555555", linewidth=1, linestyle=":")
        ax.legend(facecolor="#1e1e1e", edgecolor="none", labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_null_test(asset, null_aucs, real_auc, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style(ax, title=f"{asset} — Permutation Null Test | Micro-Regime",
           xlabel="Null AUC", ylabel="Count")
    ax.hist(null_aucs, bins=max(5, len(null_aucs)), color=PINK_CLR,
            alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(real_auc, color=GOLD_CLR, linewidth=2.5, linestyle="--",
               label=f"Real AUC = {real_auc:.4f}")
    ax.axvline(0.5, color="#555555", linewidth=1.2, linestyle=":",
               label="Chance (0.50)")
    ax.legend(facecolor="#1e1e1e", edgecolor="none", labelcolor="white")
    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_holdout(asset, ho, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f"{asset} — Locked Holdout | Micro-Regime (1M→15M)",
                 color="white", fontsize=13, fontweight="bold")
    _style(ax1, title="Calibration",
           xlabel="Mean Predicted Prob", ylabel="Fraction of Positives")
    ax1.plot([0,1],[0,1],"w--",linewidth=1,label="Perfect")
    y_ho = ho["y_ho"]
    for key, (color, label) in [("p_lgbm",(TEAL_CLR,"LGBM")),
                                  ("p_lr",  (PINK_CLR,"LR")),
                                  ("p_ens", (GOLD_CLR,"Ensemble"))]:
        p = ho.get(key, np.array([]))
        if len(p) == 0: continue
        try:
            frac, mean_p = calibration_curve(y_ho, p, n_bins=10)
            ax1.plot(mean_p, frac, "o-", color=color, label=label,
                     linewidth=2, markersize=5)
        except Exception: pass
    ax1.legend(facecolor="#1e1e1e", edgecolor="none", labelcolor="white")
    ax1.set_xlim(0,1); ax1.set_ylim(0,1)

    _style(ax2, title="AUC / Brier", xlabel="Model", ylabel="Value")
    models = list(ho["metrics"].keys())
    aucs   = [ho["metrics"][m]["auc"]   for m in models]
    briers = [ho["metrics"][m]["brier"] for m in models]
    x = np.arange(len(models)); w = 0.35
    b1 = ax2.bar(x-w/2, aucs,   w, color=TEAL_CLR, alpha=0.85, label="AUC")
    b2 = ax2.bar(x+w/2, briers, w, color=PINK_CLR, alpha=0.85, label="Brier")
    ax2.axhline(0.5, color="#555555", linewidth=1, linestyle=":")
    ax2.set_xticks(x); ax2.set_xticklabels(models, color="white")
    for bar in b1:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f"{bar.get_height():.3f}", ha="center", color="white", fontsize=9)
    ax2.legend(facecolor="#1e1e1e", edgecolor="none", labelcolor="white")
    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_feature_importance(asset, joined, feat_cols, out):
    """
    Plot original feature importances using variance of each feature
    as a proxy (since PCA scrambles the axes). For the micro-regime,
    we additionally show the 15 most-variant features which typically
    correspond to the most predictive signals.
    """
    try:
        X = joined[feat_cols].values
        # Replace inf, compute variance per feature
        X = np.where(np.isinf(X), np.nan, X)
        variances = np.nanvar(X, axis=0)
        feat_var = pd.Series(variances, index=feat_cols).sort_values(ascending=False)

        top_n = min(25, len(feat_var))
        top   = feat_var.head(top_n)

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        fig.patch.set_facecolor(DARK_BG)
        _style(ax, title=f"{asset} — Feature Variance (Top {top_n}) | Micro-Regime",
               xlabel="Variance (proxy for importance)", ylabel="Feature")

        colors = [GRNN_CLR if "active_ratio" in f else
                  TEAL_CLR if "tv_" in f or "log_" in f else
                  GOLD_CLR if "session" in f or "hour" in f or "is_" in f else
                  BLUE_CLR for f in top.index]

        ax.barh(range(len(top)), top.values[::-1], color=list(reversed(colors)), alpha=0.85)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(list(reversed(top.index)), fontsize=8, color="white")
        plt.tight_layout()
        plt.savefig(out, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")
    except Exception as e:
        print(f"  [WARN] Feature importance plot failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE CSVs
# ─────────────────────────────────────────────────────────────────────────────

def save_fold_csv(asset, results_exp, results_rol) -> pd.DataFrame:
    rows = []
    for results, wtype in [(results_exp, "expanding"), (results_rol, "rolling")]:
        for r in results:
            for model in ["LGBM", "LR", "Ensemble"]:
                m = r["metrics"][model]
                rows.append({
                    "asset": asset, "window_type": wtype,
                    "fold_id": r["fold_id"], "model": model,
                    "test_start": str(r["test_start"].date()),
                    "test_end":   str(r["test_end"].date()),
                    "train_n": r["train_n"], "test_n": r["test_n"],
                    "auc": m["auc"], "brier": m["brier"],
                    "logloss": m["logloss"], "acc": m["acc"],
                })
    df   = pd.DataFrame(rows)
    tag  = asset.replace("/","").replace(" ","_")
    path = os.path.join(OUTPUT_DIR, f"fold_metrics_{tag}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


def save_summary_csv(dfs: list):
    combined = pd.concat(dfs, ignore_index=True)
    path = os.path.join(OUTPUT_DIR, "summary_comparison.csv")
    combined.to_csv(path, index=False)
    pivot = combined.groupby(["asset","window_type","model"])["auc"].agg(["mean","std"]).round(4)
    print(f"\n  CROSS-ASSET AUC SUMMARY (Micro-Regime 1M→15M)")
    print(pivot.to_string())
    print(f"\n  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PER-ASSET ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_asset(asset_name: str,
              df_1m: pd.DataFrame,
              subsample_frac: float = 1.0) -> pd.DataFrame:

    tag = asset_name.replace("/","").replace(" ","_")

    # Feature + Label build
    joined, feat_cols = build_micro_dataset(
        df_1m,
        asset_name       = asset_name,
        forward_bars     = CFG["forward_bars"],
        bar_offset       = CFG["bar_offset"],
        regime_pct_high  = CFG["regime_pct_high"],
        regime_pct_low   = CFG["regime_pct_low"],
        rolling_baseline = CFG["rolling_baseline"],
        subsample_frac   = subsample_frac,
        verbose          = True,
    )

    X_full = joined[feat_cols].values.astype(np.float32)
    y_full = joined["micro_regime"].values.astype(np.int32)

    # Walk-Forward
    results_exp, results_rol, holdout_idx = run_wfv(
        asset_name, joined, X_full, y_full
    )

    # Holdout
    ho = evaluate_holdout(asset_name, X_full, y_full, holdout_idx)

    # Null test
    null_aucs = permutation_null_test(asset_name, X_full, y_full,
                                      CFG["null_shuffles"])

    # Terminal summary
    print_summary(asset_name, results_exp, results_rol, ho, null_aucs)

    # Charts
    print(f"\n  [{asset_name}] Generating charts ...")
    real_auc = np.nanmean([r["metrics"]["Ensemble"]["auc"] for r in results_exp])

    plot_auc_heatmap(asset_name, results_exp, results_rol,
                     os.path.join(OUTPUT_DIR, f"01_auc_heatmap_{tag}.png"))
    plot_calibration(asset_name, results_exp,
                     os.path.join(OUTPUT_DIR, f"02_calibration_{tag}.png"))
    plot_exp_vs_rolling(asset_name, results_exp, results_rol,
                        os.path.join(OUTPUT_DIR, f"03_exp_vs_rolling_{tag}.png"))
    plot_null_test(asset_name, null_aucs, real_auc,
                   os.path.join(OUTPUT_DIR, f"04_null_test_{tag}.png"))
    plot_holdout(asset_name, ho,
                 os.path.join(OUTPUT_DIR, f"05_holdout_{tag}.png"))
    plot_feature_importance(asset_name, joined, feat_cols,
                            os.path.join(OUTPUT_DIR, f"06_feature_importance_{tag}.png"))

    # CSV
    return save_fold_csv(asset_name, results_exp, results_rol)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger    = Logger(LOG_FILE)
    sys.stdout = logger

    START = time.time()

    print("\n" + "═" * 100)
    print("  MICRO-REGIME (1M → 15M) — DUAL-ASSET WALK-FORWARD VALIDATION")
    print(f"  Predict: will the NEXT 15 MINUTES be Fast or Slow tape?")
    print(f"  Log file   : {LOG_FILE}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print("═" * 100)

    print(f"\n  CONFIG:")
    for k, v in CFG.items():
        print(f"    {k:28s}: {v}")

    all_dfs = []

    # ── ASSET 1: NAS100 ──
    print("\n" + "─" * 100)
    print("  ASSET 1 of 2 : NAS100 (NQ)  [full resolution — no subsampling]")
    print("─" * 100)
    try:
        print(f"  Loading 1M data from {NQ_1M_PATH} ...")
        t0 = time.time()
        df_nq = load_nq_1m(NQ_1M_PATH)
        print(f"  Loaded {len(df_nq):,} bars | "
              f"{df_nq.index[0].date()} → {df_nq.index[-1].date()} "
              f"| {time.time()-t0:.1f}s")
        all_dfs.append(run_asset("NAS100", df_nq,
                                 subsample_frac=CFG["nq_subsample"]))
    except Exception as e:
        print(f"  [ERROR] NAS100 failed: {e}")
        import traceback; traceback.print_exc()

    # ── ASSET 2: GOLD ──
    print("\n" + "─" * 100)
    print(f"  ASSET 2 of 2 : GOLD (XAUUSD)  "
          f"[subsampled 1-in-{int(1/CFG['gold_subsample'])} to manage compute]")
    print("─" * 100)
    try:
        print(f"  Loading 1M data from {GOLD_1M_PATH} ...")
        t0 = time.time()
        df_gold = load_gold_1m(GOLD_1M_PATH)
        print(f"  Loaded {len(df_gold):,} bars | "
              f"{df_gold.index[0].date()} → {df_gold.index[-1].date()} "
              f"| {time.time()-t0:.1f}s")
        all_dfs.append(run_asset("GOLD", df_gold,
                                 subsample_frac=CFG["gold_subsample"]))
    except Exception as e:
        print(f"  [ERROR] GOLD failed: {e}")
        import traceback; traceback.print_exc()

    # Cross-asset summary
    if all_dfs:
        print("\n" + "═" * 100)
        print("  CROSS-ASSET SUMMARY — MICRO-REGIME (1M → 15M)")
        print("═" * 100)
        save_summary_csv(all_dfs)

    elapsed = time.time() - START
    print(f"\n  Total runtime : {elapsed/60:.1f} minutes")
    print(f"  Log saved to  : {LOG_FILE}")
    print("═" * 100 + "\n")

    logger.close()
    sys.stdout = logger.terminal


if __name__ == "__main__":
    main()
