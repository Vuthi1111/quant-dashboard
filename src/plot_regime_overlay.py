"""
Regime Overlay Plot
-------------------
Train on first 80% of labeled data, predict on last 20% (pure OOS).
Overlay the model's probability signal on the NQ 1H price chart.

Color coding:
  🔴 RED background   = Model predicted HIGH Vol  AND it was correct
  🔷 BLUE background  = Model predicted LOW Vol   AND it was correct
  🟡 YELLOW background = Model was wrong
  ⬜ WHITE background  = Model was uncertain (middle 40%)
"""

import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "/Users/macos/Documents/ALGO/04_Models/walk_forward_ml")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from feature_engineering import load_mt5_csv, build_features, build_vol_regime_labels

# ── 1. Load & Label ──────────────────────────────────────────────────────────
print("Loading data...")
df = load_mt5_csv("/Users/macos/Documents/ALGO/03_Data/raw/NAS100/1h_data.csv")
feat_df  = build_features(df)
label_df = build_vol_regime_labels(df, forward_bars=4, bar_offset=4,
                                    regime_pct_high=0.70, regime_pct_low=0.30,
                                    rolling_baseline=480)

joined = pd.concat([feat_df, label_df, df["Close"]], axis=1).dropna(subset=["vol_regime"])

feature_cols = feat_df.columns.tolist()
X = joined[feature_cols].values.astype(np.float32)
y = joined["vol_regime"].values.astype(np.int32)
close = joined["Close"].values
dates = joined.index

# ── 2. Train / OOS Split (80/20) ─────────────────────────────────────────────
split = int(len(X) * 0.80)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
close_te   = close[split:]
dates_te   = dates[split:]

print(f"Training on {split:,} bars | OOS test on {len(X_te):,} bars")
print(f"OOS period: {dates_te[0].date()} → {dates_te[-1].date()}")

sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s  = sc.transform(X_te)

# Fast LightGBM (no hyperopt, just clean signal)
model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    class_weight="balanced", random_state=42, verbose=-1
)
model.fit(X_tr_s, y_tr)

prob_te = model.predict_proba(X_te_s)[:, 1]   # P(High Vol)
auc_oos = roc_auc_score(y_te, prob_te)
print(f"OOS AUC: {auc_oos:.4f}")

# ── 3. Build Overlay DataFrame ────────────────────────────────────────────────
oos_df = pd.DataFrame({
    "date":    dates_te,
    "close":   close_te,
    "prob":    prob_te,
    "actual":  y_te,
}).set_index("date")

# Predicted regime (using same 0.70/0.30 thresholds)
oos_df["pred_regime"] = "uncertain"
oos_df.loc[oos_df["prob"] > 0.70, "pred_regime"] = "high"
oos_df.loc[oos_df["prob"] < 0.30, "pred_regime"] = "low"

# Correctness
def classify_bar(row):
    if row["pred_regime"] == "uncertain":
        return "uncertain"
    elif row["pred_regime"] == "high" and row["actual"] == 1:
        return "correct_high"
    elif row["pred_regime"] == "low"  and row["actual"] == 0:
        return "correct_low"
    else:
        return "wrong"

oos_df["status"] = oos_df.apply(classify_bar, axis=1)

# Count accuracy stats
total_calls = (oos_df["pred_regime"] != "uncertain").sum()
correct      = oos_df["status"].isin(["correct_high","correct_low"]).sum()
wrong        = (oos_df["status"] == "wrong").sum()
uncertain    = (oos_df["status"] == "uncertain").sum()
accuracy     = correct / total_calls if total_calls > 0 else 0

print(f"\nWhen model was confident (>{70}% or <{30}%):")
print(f"  Correct  : {correct:,}  ({100*accuracy:.1f}%)")
print(f"  Wrong    : {wrong:,}  ({100*wrong/total_calls:.1f}%)")
print(f"  Uncertain: {uncertain:,} bars (model sat out)")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
# Show only last 18 months for readability
cutoff = oos_df.index[-1] - pd.DateOffset(months=18)
plot_df = oos_df[oos_df.index >= cutoff]

color_map = {
    "correct_high": "#ff4444",     # Red
    "correct_low":  "#4488ff",     # Blue
    "wrong":        "#ffcc00",     # Yellow
    "uncertain":    "#2a2a2a",     # Dark grey
}

fig, axes = plt.subplots(3, 1, figsize=(18, 12),
                          gridspec_kw={"height_ratios": [3, 1.5, 1]},
                          facecolor="#0d0d0d")
fig.subplots_adjust(hspace=0.08)

# ── Top panel: Price + background shading ─────────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor("#0d0d0d")

# Shade background by status in blocks
i = 0
while i < len(plot_df):
    status = plot_df["status"].iloc[i]
    j = i + 1
    while j < len(plot_df) and plot_df["status"].iloc[j] == status:
        j += 1
    ax1.axvspan(plot_df.index[i], plot_df.index[j-1],
                alpha=0.25, color=color_map[status], linewidth=0)
    i = j

ax1.plot(plot_df.index, plot_df["close"], color="#e8e8e8", linewidth=0.9, zorder=5)
ax1.set_ylabel("NQ100 Price (USD)", color="#aaaaaa", fontsize=11)
ax1.tick_params(colors="#aaaaaa", labelsize=9)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
for spine in ax1.spines.values(): spine.set_color("#333333")
ax1.set_xlim(plot_df.index[0], plot_df.index[-1])
ax1.xaxis.set_visible(False)
ax1.set_title(
    f"NQ100 — Volatility Regime Overlay (OOS: {plot_df.index[0].strftime('%b %Y')} → {plot_df.index[-1].strftime('%b %Y')})\n"
    f"OOS AUC: {auc_oos:.3f}  |  Directional Accuracy When Confident: {100*accuracy:.1f}%  "
    f"| Correct: {correct:,}  Wrong: {wrong:,}  Sat-Out: {uncertain:,}",
    color="white", fontsize=12, pad=10
)

legend_handles = [
    mpatches.Patch(color="#ff4444", alpha=0.7, label="✅ HIGH Vol Predicted — Correctly"),
    mpatches.Patch(color="#4488ff", alpha=0.7, label="✅ LOW Vol Predicted — Correctly"),
    mpatches.Patch(color="#ffcc00", alpha=0.7, label="❌ Wrong Prediction"),
    mpatches.Patch(color="#444444", alpha=0.7, label="⬜ Uncertain — Model Sat Out"),
]
ax1.legend(handles=legend_handles, loc="upper left", framealpha=0.3,
           facecolor="#1a1a1a", edgecolor="#444444", labelcolor="white", fontsize=9)

# ── Middle panel: Probability signal ─────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#0d0d0d")

ax2.fill_between(plot_df.index, plot_df["prob"], 0.5,
                  where=plot_df["prob"] > 0.5, alpha=0.6, color="#ff4444", interpolate=True)
ax2.fill_between(plot_df.index, plot_df["prob"], 0.5,
                  where=plot_df["prob"] < 0.5, alpha=0.6, color="#4488ff", interpolate=True)
ax2.plot(plot_df.index, plot_df["prob"], color="#e8e8e8", linewidth=0.6, zorder=5)
ax2.axhline(0.70, color="#ff4444", linewidth=1.0, linestyle="--", alpha=0.8, label="High Vol Trigger (0.70)")
ax2.axhline(0.30, color="#4488ff", linewidth=1.0, linestyle="--", alpha=0.8, label="Low Vol Trigger (0.30)")
ax2.axhline(0.50, color="#555555", linewidth=0.6, linestyle=":")
ax2.set_ylabel("P(High Vol)", color="#aaaaaa", fontsize=10)
ax2.set_ylim(0, 1)
ax2.tick_params(colors="#aaaaaa", labelsize=9)
for spine in ax2.spines.values(): spine.set_color("#333333")
ax2.set_xlim(plot_df.index[0], plot_df.index[-1])
ax2.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a1a",
           edgecolor="#444444", labelcolor="white", fontsize=8)
ax2.xaxis.set_visible(False)

# ── Bottom panel: Actual realized vol ────────────────────────────────────────
ax3 = axes[2]
ax3.set_facecolor("#0d0d0d")

log_hl = np.log(df["High"] / df["Low"]) ** 2
log_co = np.log(df["Close"] / df["Open"]) ** 2
bar_gk = np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
actual_vol = bar_gk.rolling(8).mean()
actual_vol_plot = actual_vol.reindex(plot_df.index)

ax3.fill_between(plot_df.index, actual_vol_plot, alpha=0.7, color="#888888")
ax3.set_ylabel("Actual\nRealized Vol", color="#aaaaaa", fontsize=9)
ax3.tick_params(colors="#aaaaaa", labelsize=8)
for spine in ax3.spines.values(): spine.set_color("#333333")
ax3.set_xlim(plot_df.index[0], plot_df.index[-1])
ax3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b '%y"))
ax3.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=2))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, ha="center", color="#aaaaaa", fontsize=8)

out_path = "/Users/macos/.gemini/antigravity/brain/a79aad02-781a-4a03-8ce4-594372872646/scratch/regime_overlay.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
print(f"\n✓ Saved: {out_path}")
