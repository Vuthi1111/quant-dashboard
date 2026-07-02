"""
=========================================================
  NAS100 PRE-MARKET VOLATILITY BRIEFING (LIVE INFERENCE)
=========================================================
Reads live data dropped from MT5 (via ExportLive1H.mq5),
computes volatility features, and predicts today's regime.
"""

import warnings; warnings.filterwarnings("ignore")
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

# Import our custom feature engineering
from feature_engineering import load_mt5_csv, build_features, build_vol_regime_labels

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path("/Users/macos/Documents/ALGO")

def get_historical_path(asset: str) -> Path:
    if asset == "NAS100":
        return ROOT / "03_Data" / "raw" / "NAS100" / "1h_data.csv"
    elif asset == "GOLD":
        return ROOT / "03_Data" / "raw" / "GOLD_XAUUSD" / "XAUUSD_1H.csv"
    raise ValueError(f"Unknown asset: {asset}")

# We dynamically point this directly to your CrossOver MT5 folder!
LIVE_NAS100_PATH = Path(os.path.expanduser("~/Library/Application Support/CrossOver/Bottles/MT5/drive_c/Program Files/MetaTrader 5/MQL5/Files/nas100_live.csv"))
LIVE_GOLD_PATH = Path(os.path.expanduser("~/Library/Application Support/CrossOver/Bottles/MT5/drive_c/Program Files/MetaTrader 5/MQL5/Files/xauusd_live.csv"))

PROB_HIGH = 0.70
PROB_LOW  = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# 1. TRAIN THE MODEL (Takes < 1 second)
# ─────────────────────────────────────────────────────────────────────────────
def train_production_model(asset: str = "NAS100"):
    """Trains the model on all historical data up to yesterday."""
    hist_path = get_historical_path(asset)
    df_hist = load_mt5_csv(str(hist_path))
    feat_df = build_features(df_hist)
    label_df = build_vol_regime_labels(
        df_hist, forward_bars=4, bar_offset=4, 
        regime_pct_high=PROB_HIGH, regime_pct_low=PROB_LOW, rolling_baseline=480
    )
    
    joined = pd.concat([feat_df, label_df], axis=1).dropna(subset=["vol_regime"])
    feature_cols = feat_df.columns.tolist()
    
    X = joined[feature_cols].values.astype(np.float32)
    y = joined["vol_regime"].values.astype(np.int32)
    
    sc = StandardScaler().fit(X)
    X_scaled = sc.transform(X)
    
    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    model.fit(X_scaled, y)
    
    return model, sc, feature_cols

# ─────────────────────────────────────────────────────────────────────────────
# 2. RUN LIVE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def generate_briefing():
    os.system('clear' if os.name == 'posix' else 'cls')
    print("="*65)
    print("  NAS100 PRE-MARKET VOLATILITY BRIEFING")
    print("="*65)

    if not LIVE_NAS100_PATH.exists():
        print(f"\n[ERROR] Live data not found at:\n{LIVE_NAS100_PATH}")
        print("\nPlease run the 'ExportLive1H.mq5' script in MT5 and ensure")
        print("the file is saved/copied to that location.")
        return

    # Train model
    print("[1/3] Booting AI Core and mapping historical volatility...")
    model, scaler, feature_cols = train_production_model(asset="NAS100")

    # Load Live Data
    print("[2/3] Reading live MT5 data drop...")
    df_live = load_mt5_csv(str(LIVE_NAS100_PATH))
    
    if len(df_live) < 500:
        print(f"[ERROR] Live CSV only has {len(df_live)} bars. We need at least 500 to calculate HAR and EWMA.")
        return

    # Compute features for live data
    print("[3/3] Computing real-time Garman-Klass & RiskMetrics...")
    live_features = build_features(df_live)
    
    # Get the very last row (the current live state of the market)
    current_state = live_features.iloc[[-1]][feature_cols]
    last_dt = live_features.index[-1]
    
    # Predict
    X_live = scaler.transform(current_state.values.astype(np.float32))
    prob_high = model.predict_proba(X_live)[0][1]

    # Metrics for the dashboard
    gk_current = current_state["GK_10"].values[0]
    ewma_current = current_state["RM2006"].values[0]
    har_month = current_state["HAR_M"].values[0]

    gk_avg = live_features["GK_10"].rolling(24*30).mean().iloc[-1]
    gk_ratio = gk_current / gk_avg if gk_avg > 0 else 1.0
    
    ewma_trend = "ACCELERATING 📈" if ewma_current > live_features["RM2006"].iloc[-24] else "DECELERATING 📉"
    
    if prob_high > PROB_HIGH:
        regime_text = "HIGH VOLATILITY EXPECTED (Trending/Expansive)"
        orb_advice = "GREEN LIGHT 🟢 - Market conditions strongly favor ORB breakouts.\n   Sizing: Consider 0.5x due to wider expected ATR stops."
    elif prob_high < PROB_LOW:
        regime_text = "LOW VOLATILITY EXPECTED (Choppy/Compressing)"
        orb_advice = "RED LIGHT 🔴 - ORB breakouts highly likely to fail today.\n   Action: Skip ORB. Market favors Mean Reversion fading."
    else:
        regime_text = "UNCERTAIN / MIXED"
        orb_advice = "YELLOW LIGHT 🟡 - No statistical edge detected.\n   Action: Trade at your own discretion."

    # Print the Terminal Dashboard
    print("\n" + "="*65)
    print(f"[DATA] Read {len(df_live)} live 1H bars. Last bar: {last_dt.strftime('%Y-%m-%d %H:%M')}")
    print("\n1. ML REGIME PREDICTION")
    print(f"   Probability of HIGH VOL:   {prob_high*100:.1f}%")
    print(f"   Current Regime State:      {regime_text}")
    print("\n2. VOLATILITY METRICS")
    print(f"   Garman-Klass (Current):    {gk_current:.6f}  ({gk_ratio:.1f}x vs 30D avg)")
    print(f"   EWMA Trend:                {ewma_trend}")
    print(f"   HAR Monthly Baseline:      {har_month:.6f}")
    print("\n3. ORB STRATEGY IMPACT")
    print(f"   {orb_advice}")
    print("="*65 + "\n")

if __name__ == "__main__":
    generate_briefing()
