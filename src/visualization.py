"""
visualization.py
═══════════════════════════════════════════════════════════════════════════════
All plots for the Walk-Forward Pipeline Report

Plots generated:
  1. Walk-forward equity curves (OOS stitched, Expanding vs Rolling)
  2. Per-fold AUC heatmap (Expanding vs Rolling)
  3. Calibration curves (all models — Val + OOS Test)
  4. Feature importance across folds (LightGBM PCA components)
  5. Max drawdown curve with 4% / 8% risk limits overlay
  6. Final holdout confusion matrices
  7. Adversarial validation score per fold
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Dict, List


DARK_BG  = "#0d0d0d"
PANEL_BG = "#141414"
COLORS   = {
    "Meta": "#FFD700",     # Gold
    "LGBM": "#00FFCC",     # Teal
    "LR":   "#FF6699",     # Pink
    "LSTM": "#AA88FF",     # Purple
    "expanding": "#00BFFF",
    "rolling":   "#FF8C00",
    "drawdown":  "#FF3333",
    "dd_daily":  "#FF6666",
    "dd_max":    "#CC0000",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[:].set_color("#333333")
    ax.grid(True, color="#1e1e1e", linewidth=0.7, linestyle="--")
    if title:   ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)
    if xlabel:  ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=9)
    if ylabel:  ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)


def _drawdown(cum_ret: pd.Series) -> pd.Series:
    roll_max = cum_ret.cummax()
    return (cum_ret - roll_max) / (roll_max.abs() + 1e-9)


def _equity_from_probs(probs: pd.Series, labels: pd.Series,
                       threshold: float = 0.5,
                       risk_per_trade: float = 0.01) -> pd.Series:
    """
    Simulate a naive directional equity curve from probability predictions.
    Position = +1 when prob > threshold, else -1.
    Return per bar = position * actual_return (labels are 0/1 direction).
    This is a simplified PnL proxy — replace with actual price returns for precision.
    """
    pos     = (probs > threshold).astype(float) * 2 - 1   # +1 / -1
    ret     = (labels * 2 - 1) * pos * risk_per_trade      # directional return
    cum_ret = (1 + ret).cumprod()
    return cum_ret


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: Walk-Forward OOS Equity Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_equity_curves(results_exp: list, results_rol: list,
                       index_full: pd.DatetimeIndex, output_path: str):
    """
    Stitch OOS test predictions across all folds into continuous equity curves.
    Expanding vs Rolling shown on separate panels, all models overlaid.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=False)
    fig.patch.set_facecolor(DARK_BG)

    for ax, results, wtype in [(ax1, results_exp, "Expanding"),
                                (ax2, results_rol, "Rolling")]:
        _style_ax(ax, title=f"Walk-Forward OOS Equity Curve — {wtype} Window",
                  xlabel="Fold (OOS Test Month)", ylabel="Cumulative Return")

        for model_name, color in [("Meta", COLORS["Meta"]),
                                   ("LGBM", COLORS["LGBM"]),
                                   ("LR",   COLORS["LR"]),
                                   ("LSTM", COLORS["LSTM"])]:
            all_preds, all_labels = [], []
            for r in results:
                pred  = r.test_preds.get(model_name, np.array([]))
                label = r.test_labels if r.test_labels is not None else np.array([])
                valid = ~np.isnan(pred) if len(pred) > 0 else np.array([], dtype=bool)
                if valid.sum() > 0:
                    all_preds.append(pred[valid])
                    all_labels.append(label[valid])

            if all_preds:
                p_series = pd.Series(np.concatenate(all_preds))
                l_series = pd.Series(np.concatenate(all_labels))
                eq = _equity_from_probs(p_series, l_series)
                ax.plot(eq.values, label=model_name, color=color,
                        linewidth=1.5 if model_name == "Meta" else 1.0,
                        alpha=1.0 if model_name == "Meta" else 0.7)

        # Fold boundaries
        for i, r in enumerate(results):
            ax.axvline(x=sum(len(r2.test_labels if r2.test_labels is not None else [])
                             for r2 in results[:i]),
                       color="#333333", linewidth=0.8, linestyle=":")
            ax.text(sum(len(r2.test_labels if r2.test_labels is not None else []) for r2 in results[:i]) + 2,
                    ax.get_ylim()[0] * 1.001, f"F{r.fold_id}",
                    color="#555555", fontsize=7)

        ax.axhline(y=1.0, color="#555555", linewidth=0.8, linestyle="--")
        ax.legend(loc="upper left", facecolor="#1e1e1e",
                  edgecolor="none", labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Equity curves saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: Per-Fold AUC Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_auc_heatmap(results_exp: list, results_rol: list, output_path: str):
    """
    Heatmap of OOS test AUC per fold per model.
    Side by side: Expanding vs Rolling.
    """
    def build_matrix(results):
        models = ["Meta", "LGBM", "LR", "LSTM"]
        data = {m: [] for m in models}
        fold_ids = []
        for r in results:
            fold_ids.append(f"F{r.fold_id}")
            for m in models:
                data[m].append(r.metrics.get(m, {}).get("auc", np.nan))
        return pd.DataFrame(data, index=fold_ids).T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(4, len(results_exp) * 0.4 + 2)))
    fig.patch.set_facecolor(DARK_BG)

    for ax, results, title in [(ax1, results_exp, "Expanding"),
                                (ax2, results_rol, "Rolling")]:
        mat = build_matrix(results)
        _style_ax(ax, title=f"OOS Test AUC per Fold — {title}")
        sns.heatmap(mat, ax=ax, cmap="RdYlGn", vmin=0.45, vmax=0.65,
                    annot=True, fmt=".3f", linewidths=0.5,
                    cbar_kws={"label": "AUC"},
                    annot_kws={"size": 8, "color": "white"})
        ax.tick_params(colors="white")
        ax.set_xlabel("Fold", color="#aaaaaa")
        ax.set_ylabel("Model", color="#aaaaaa")
        ax.yaxis.label.set_color("white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  ✓ AUC heatmap saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: Calibration Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration(results: list, output_path: str, title_prefix: str = ""):
    """
    Aggregate OOS test predictions across all folds, plot calibration curves.
    A well-calibrated model's line should lie close to the diagonal.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax, title=f"{title_prefix}Probability Calibration — OOS Test",
              xlabel="Mean Predicted Probability", ylabel="Fraction of Positives")

    ax.plot([0, 1], [0, 1], "w--", linewidth=1, label="Perfect calibration")

    for model_name, color in [("Meta", COLORS["Meta"]), ("LGBM", COLORS["LGBM"]),
                               ("LR",  COLORS["LR"]),   ("LSTM", COLORS["LSTM"])]:
        all_preds, all_labels = [], []
        for r in results:
            pred  = r.test_preds.get(model_name, np.array([]))
            label = r.test_labels if r.test_labels is not None else np.array([])
            valid = ~np.isnan(pred)
            if valid.sum() > 10:
                all_preds.append(pred[valid])
                all_labels.append(label[valid])
        if not all_preds:
            continue
        probs  = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        try:
            frac, mean_pred = calibration_curve(labels, probs, n_bins=10)
            ax.plot(mean_pred, frac, "o-", color=color, label=model_name,
                    linewidth=2, markersize=5)
        except Exception:
            pass

    ax.legend(loc="upper left", facecolor="#1e1e1e",
              edgecolor="none", labelcolor="white", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Calibration curve saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: Drawdown Curve with Risk Limits
# ─────────────────────────────────────────────────────────────────────────────

def plot_drawdown(results_exp: list, results_rol: list, output_path: str):
    """
    OOS equity curves with drawdown panel.
    Horizontal lines for 4% daily and 8% max drawdown limits.
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 14),
                              gridspec_kw={"height_ratios": [2, 1, 2, 1]})
    fig.patch.set_facecolor(DARK_BG)

    for row, (results, wtype) in enumerate([(results_exp, "Expanding"),
                                             (results_rol, "Rolling")]):
        ax_eq  = axes[row * 2]
        ax_dd  = axes[row * 2 + 1]
        _style_ax(ax_eq, title=f"OOS Meta Equity — {wtype}", ylabel="Cum. Return")
        _style_ax(ax_dd, title=f"Drawdown — {wtype}", ylabel="Drawdown %")

        all_p, all_l = [], []
        for r in results:
            pred  = r.test_preds.get("Meta", np.array([]))
            label = r.test_labels if r.test_labels is not None else np.array([])
            valid = ~np.isnan(pred)
            if valid.sum() > 0:
                all_p.append(pred[valid]); all_l.append(label[valid])

        if not all_p:
            continue

        p_s = pd.Series(np.concatenate(all_p))
        l_s = pd.Series(np.concatenate(all_l))
        eq  = _equity_from_probs(p_s, l_s)
        dd  = _drawdown(eq) * 100

        ax_eq.plot(eq.values, color=COLORS[wtype.lower()], linewidth=1.5)
        ax_eq.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
        ax_eq.fill_between(range(len(eq)), 1, eq.values,
                           where=(eq.values >= 1), alpha=0.1, color="#00FF88")
        ax_eq.fill_between(range(len(eq)), 1, eq.values,
                           where=(eq.values < 1),  alpha=0.1, color="#FF3333")

        ax_dd.fill_between(range(len(dd)), 0, dd.values, color=COLORS["drawdown"],
                           alpha=0.5)
        ax_dd.plot(dd.values, color=COLORS["drawdown"], linewidth=0.8)
        ax_dd.axhline(-4.0, color=COLORS["dd_daily"], linewidth=1.5, linestyle="--",
                      label="4% Daily Limit")
        ax_dd.axhline(-8.0, color=COLORS["dd_max"], linewidth=1.5, linestyle="-",
                      label="8% Max DD")
        ax_dd.legend(loc="lower left", facecolor="#1e1e1e",
                     edgecolor="none", labelcolor="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Drawdown chart saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5: Expanding vs Rolling Comparative AUC Line
# ─────────────────────────────────────────────────────────────────────────────

def plot_exp_vs_rolling(results_exp: list, results_rol: list, output_path: str):
    """
    Side-by-side fold-by-fold Meta AUC comparison: expanding vs rolling.
    Reveals whether old data helps or hurts across time.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax, title="Expanding vs Rolling — Meta Model OOS AUC per Fold",
              xlabel="Fold ID", ylabel="AUC")

    for results, color, label in [(results_exp, COLORS["expanding"], "Expanding"),
                                   (results_rol, COLORS["rolling"],   "Rolling")]:
        fold_ids = [r.fold_id for r in results]
        aucs     = [r.metrics.get("Meta", {}).get("auc", np.nan) for r in results]
        ax.plot(fold_ids, aucs, "o-", color=color, label=label,
                linewidth=2, markersize=6)
        # Rolling average
        s = pd.Series(aucs)
        ax.plot(fold_ids, s.rolling(3, min_periods=1).mean().values,
                color=color, linewidth=1, linestyle="--", alpha=0.5)

    ax.axhline(0.5, color="#555555", linewidth=1, linestyle=":")
    ax.legend(facecolor="#1e1e1e", edgecolor="none", labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Expanding vs Rolling AUC saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6: Final Holdout Confusion Matrix + Summary Table
# ─────────────────────────────────────────────────────────────────────────────

def plot_holdout_summary(holdout_preds: Dict[str, np.ndarray],
                         holdout_labels: np.ndarray,
                         holdout_metrics: Dict[str, Dict],
                         output_path: str):
    """
    Final locked holdout evaluation:
    Confusion matrices + performance table.
    """
    models = [m for m in ["Meta", "LGBM", "LR", "LSTM"] if m in holdout_preds]
    n = len(models)

    fig = plt.figure(figsize=(5 * n, 10))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(2, n, figure=fig)

    for i, model_name in enumerate(models):
        pred  = holdout_preds[model_name]
        valid = ~np.isnan(pred)
        pred_class = (pred[valid] > 0.5).astype(int)
        true_class = holdout_labels[valid]

        # Confusion matrix
        ax_cm = fig.add_subplot(gs[0, i])
        _style_ax(ax_cm, title=f"{model_name}\n(Holdout Test)")
        cm = confusion_matrix(true_class, pred_class)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    ax=ax_cm, cbar=False,
                    xticklabels=["Pred↓", "Pred↑"],
                    yticklabels=["True↓", "True↑"],
                    annot_kws={"size": 12})
        ax_cm.tick_params(colors="white")

        # Metrics text
        ax_txt = fig.add_subplot(gs[1, i])
        ax_txt.set_facecolor(PANEL_BG)
        ax_txt.axis("off")
        m = holdout_metrics.get(model_name, {})
        txt = (f"AUC:      {m.get('auc', np.nan):.4f}\n"
               f"Accuracy: {m.get('accuracy', np.nan):.4f}\n"
               f"LogLoss:  {m.get('logloss', np.nan):.4f}")
        ax_txt.text(0.5, 0.5, txt, transform=ax_txt.transAxes,
                    ha="center", va="center", color="white",
                    fontsize=12, family="monospace",
                    bbox=dict(facecolor="#1e1e1e", edgecolor="#333333",
                              boxstyle="round,pad=0.6"))

    fig.suptitle("Final Locked Holdout Evaluation", color="white",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Holdout summary saved: {output_path}")
