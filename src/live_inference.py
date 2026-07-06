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
from feature_engineering import load_mt5_csv, build_features, build_vol_regime_labels, resample_to_4h, resample_to_15m

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path("/Users/macos/Documents/ALGO")
PROJECT_ROOT = ROOT / "projects" / "volatility_regime_model"
sys.path.insert(0, str(PROJECT_ROOT))

def get_historical_path(asset: str) -> Path:
    if asset == "NAS100":
        return ROOT / "03_Data" / "raw" / "NAS100" / "1h_data.csv"
    elif asset == "GOLD":
        return ROOT / "03_Data" / "raw" / "GOLD_XAUUSD" / "XAUUSD_1H.csv"
    raise ValueError(f"Unknown asset: {asset}")

def get_historical_1m_path(asset: str) -> Path:
    if asset == "NAS100":
        return ROOT / "03_Data" / "raw" / "NAS100" / "1m_data.csv"
    elif asset == "GOLD":
        return ROOT / "03_Data" / "raw" / "GOLD_XAUUSD" / "XAUUSD_M1.csv"
    raise ValueError(f"Unknown asset: {asset}")

# We dynamically point this directly to your CrossOver MT5 folder!
LIVE_NAS100_PATH = Path(os.path.expanduser("~/Library/Application Support/CrossOver/Bottles/MT5/drive_c/Program Files/MetaTrader 5/MQL5/Files/nas100_live.csv"))
LIVE_GOLD_PATH = Path(os.path.expanduser("~/Library/Application Support/CrossOver/Bottles/MT5/drive_c/Program Files/MetaTrader 5/MQL5/Files/xauusd_live.csv"))

LIVE_NAS100_PATH_1M = Path(os.path.expanduser("~/Library/Application Support/CrossOver/Bottles/MT5/drive_c/Program Files/MetaTrader 5/MQL5/Files/nas100_live_1m.csv"))
LIVE_GOLD_PATH_1M = Path(os.path.expanduser("~/Library/Application Support/CrossOver/Bottles/MT5/drive_c/Program Files/MetaTrader 5/MQL5/Files/xauusd_live_1m.csv"))

PROB_HIGH = 0.70
PROB_LOW  = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# 1. TRAIN THE MODEL (Takes < 1 second)
# ─────────────────────────────────────────────────────────────────────────────
def _train_single_model(df_hist: pd.DataFrame, is_4h: bool = False, asset: str = "NAS100"):
    # 4H data has a forward window of 1 bar (1 x 4H) instead of 4 (4 x 1H)
    forward_bars = 1 if is_4h else 4
    bar_offset = 1 if is_4h else 4
    rolling_baseline = 120 if is_4h else 480
    
    feat_df = build_features(df_hist)
    label_df = build_vol_regime_labels(
        df_hist, forward_bars=forward_bars, bar_offset=bar_offset, 
        regime_pct_high=PROB_HIGH, regime_pct_low=PROB_LOW, rolling_baseline=rolling_baseline
    )
    
    joined = pd.concat([feat_df, label_df], axis=1).dropna(subset=["vol_regime"])
    feature_cols = feat_df.columns.tolist()
    
    # Clean infinities and NaNs from features before training
    joined.replace([np.inf, -np.inf], np.nan, inplace=True)
    joined.dropna(subset=feature_cols, inplace=True)
    
    X = joined[feature_cols].values.astype(np.float32)
    y = joined["vol_regime"].values.astype(np.int32)
    
    sc = StandardScaler().fit(X)
    X_scaled = sc.transform(X)
    
    if is_4h:
        if asset == "GOLD":
            model = lgb.LGBMClassifier(
                learning_rate=0.0329,
                num_leaves=31,
                max_depth=9,
                min_child_samples=73,
                subsample=0.8308,
                colsample_bytree=0.5040,
                reg_alpha=0.0005,
                reg_lambda=0.0123,
                n_estimators=332,
                class_weight="balanced", random_state=42, verbose=-1,
            )
        else:
            model = lgb.LGBMClassifier(
                learning_rate=0.0560,
                num_leaves=41,
                max_depth=6,
                min_child_samples=76,
                subsample=0.7236,
                colsample_bytree=0.7958,
                reg_alpha=0.4243,
                reg_lambda=0.0005,
                n_estimators=210,
                class_weight="balanced", random_state=42, verbose=-1,
            )
    else:
        # Default 1H parameters
        model = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=42, verbose=-1,
        )
        
    model.fit(X_scaled, y)
    
    return model, sc, feature_cols

def train_production_model(asset: str = "NAS100"):
    """Trains the 1H and 4H models on all historical data up to yesterday."""
    hist_path = get_historical_path(asset)
    df_hist_1h = load_mt5_csv(str(hist_path))
    
    # Train 1H
    model_1h, sc_1h, feat_cols_1h = _train_single_model(df_hist_1h, is_4h=False, asset=asset)
    
    # Train 4H
    df_hist_4h = resample_to_4h(df_hist_1h)
    model_4h, sc_4h, feat_cols_4h = _train_single_model(df_hist_4h, is_4h=True, asset=asset)
    
    return {
        "1H": (model_1h, sc_1h, feat_cols_1h),
        "4H": (model_4h, sc_4h, feat_cols_4h)
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1b. VWAP COPILOT MODEL (15M Intraday Scalping)
# ─────────────────────────────────────────────────────────────────────────────

def _hurst_rs(series, min_n=8):
    """Rescaled Range Hurst Exponent."""
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


def _build_vwap_features(df):
    """Build the full 15M intraday feature set for VWAP Copilot."""
    from feature_engineering import (
        garman_klass, parkinson, rogers_satchell,
        range_ratio, tick_vol_acceleration, resample_to_15m
    )

    feat = pd.DataFrame(index=df.index)

    # Time encodings
    hours = df.index.hour + df.index.minute / 60.0
    feat['hour_sin'] = pd.Series(np.sin(2 * np.pi * hours / 24.0), index=df.index).shift(1)
    feat['hour_cos'] = pd.Series(np.cos(2 * np.pi * hours / 24.0), index=df.index).shift(1)
    dow = df.index.dayofweek
    feat['dow_sin'] = pd.Series(np.sin(2 * np.pi * dow / 5.0), index=df.index).shift(1)
    feat['dow_cos'] = pd.Series(np.cos(2 * np.pi * dow / 5.0), index=df.index).shift(1)

    # Daily Anchored VWAP
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

    # Return lags
    lr = np.log(df['Close'] / df['Close'].shift(1))
    for lag in [1, 2, 4, 8, 16]:
        feat[f'ret_lag{lag}'] = lr.shift(lag)

    # Volatility estimators
    for w in [4, 8, 16]:
        feat[f'GK_{w}'] = garman_klass(df, w).shift(1)
        feat[f'PK_{w}'] = parkinson(df, w).shift(1)
        feat[f'RS_{w}'] = rogers_satchell(df, w).shift(1)

    feat['HV_16'] = lr.rolling(16).std().shift(1)
    feat['HV_96'] = lr.rolling(96).std().shift(1)
    feat['vol_ratio'] = feat['HV_16'] / (feat['HV_96'] + 1e-9)

    # Momentum
    feat['roc_4'] = (df['Close'] / df['Close'].shift(4) - 1).shift(1)
    feat['roc_16'] = (df['Close'] / df['Close'].shift(16) - 1).shift(1)
    feat['roc_96'] = (df['Close'] / df['Close'].shift(96) - 1).shift(1)

    # RSI
    delta = lr.copy()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    feat['rsi_14'] = (100 - 100 / (1 + rs)).shift(1)

    # Bollinger Band position
    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    feat['bb_pos'] = ((df['Close'] - ma20) / (2 * std20 + 1e-9)).shift(1)

    # Microstructure
    feat['range_ratio'] = range_ratio(df).shift(1)
    feat['tickvol_accel'] = tick_vol_acceleration(df).shift(1)

    # Relative tick volume
    vol_4 = df['Tick_Volume'].rolling(4).sum()
    vol_96_avg = df['Tick_Volume'].rolling(96).mean() * 4
    feat['rtv'] = (vol_4 / vol_96_avg.replace(0, np.nan)).shift(1)

    # Rolling Hurst (32-bar)
    closes = df['Close'].values
    hurst_vals = np.full(len(closes), np.nan)
    for i in range(32, len(closes)):
        hurst_vals[i] = _hurst_rs(closes[i-32:i])
    feat['hurst_32'] = pd.Series(hurst_vals, index=df.index).shift(1)

    return feat


# The feature columns that Model 3 uses (must match training)
VWAP_META_COLS = [
    "Open", "High", "Low", "Close", "Tick_Volume", "Volume", "Spread",
    "spread", "vwap_label", "fpt_horizon", "uniqueness_weight",
    "date", "typ_price", "tv", "tv2"
]


def train_vwap_copilot_model():
    """
    Train the 15M VWAP Copilot LightGBM model on all historical data.
    Called once at dashboard boot time.
    Returns (model, feature_cols).
    """
    from feature_engineering import resample_to_15m, create_vwap_scalp_labels

    hist_path = ROOT / "03_Data" / "raw" / "GOLD_XAUUSD" / "XAUUSD_M5.csv"
    df_raw = load_mt5_csv(str(hist_path))
    df_15m = resample_to_15m(df_raw)

    feat_df = _build_vwap_features(df_15m)
    df_full = pd.concat([df_15m, feat_df], axis=1)
    df_labeled = create_vwap_scalp_labels(df_full, max_horizon=16)

    feature_cols = [c for c in df_labeled.columns if c not in VWAP_META_COLS]

    valid_mask = (
        ~df_labeled[feature_cols].isna().any(axis=1)
        & ~df_labeled['vwap_label'].isna()
    )
    df_clean = df_labeled[valid_mask].copy()

    X = df_clean[feature_cols].values
    y = df_clean['vwap_label'].values
    w = df_clean['uniqueness_weight'].values

    model = lgb.LGBMClassifier(
        objective='binary', metric='auc', boosting_type='gbdt',
        learning_rate=0.05, num_leaves=16, max_depth=5,
        feature_fraction=0.8, verbose=-1, n_estimators=200
    )
    model.fit(X, y, sample_weight=w)

    return model, feature_cols


def compute_vwap_copilot_state(df_live_m1, vwap_model, feature_cols):
    """
    Compute the full VWAP Copilot state from live M1 data.

    Parameters
    ----------
    df_live_m1   : pd.DataFrame of live M1 OHLCV bars from MT5
    vwap_model   : trained LightGBM model
    feature_cols : list of feature column names

    Returns
    -------
    dict with keys:
        vwap_zscore     : float — current VWAP z-score
        ml_probability  : float — LightGBM reversion probability
        hurst           : float — 32-bar Hurst exponent
        vol_ratio       : float — short-term vs structural volatility
        regime_context  : str   — "MEAN_REVERTING" / "TRENDING" / "RANDOM_WALK"
        signal          : str   — "LONG_SETUP" / "SHORT_SETUP" / "NO_SETUP"
        signal_color    : str   — Rich color string for the panel
    """
    from feature_engineering import resample_to_15m

    # Resample M1 → 15M
    df_15m = resample_to_15m(df_live_m1)

    if len(df_15m) < 100:
        return {
            'vwap_zscore': np.nan, 'ml_probability': np.nan,
            'hurst': np.nan, 'vol_ratio': np.nan,
            'regime_context': 'INSUFFICIENT DATA',
            'signal': 'WAITING', 'signal_color': 'dim'
        }

    # Build features on the live 15M bars
    feat_df = _build_vwap_features(df_15m)
    df_full = pd.concat([df_15m, feat_df], axis=1)

    # Get the latest bar's features
    latest = df_full.iloc[[-1]]

    # Check we have all required columns
    missing = [c for c in feature_cols if c not in latest.columns]
    if missing:
        # Add missing columns as NaN
        for c in missing:
            latest[c] = np.nan

    X = latest[feature_cols].values

    # If any features are NaN, return a safe state
    if np.any(np.isnan(X)):
        return {
            'vwap_zscore': float(latest['vwap_zscore'].values[0]) if 'vwap_zscore' in latest.columns else np.nan,
            'ml_probability': np.nan,
            'hurst': float(latest['hurst_32'].values[0]) if 'hurst_32' in latest.columns else np.nan,
            'vol_ratio': float(latest['vol_ratio'].values[0]) if 'vol_ratio' in latest.columns else np.nan,
            'regime_context': 'COMPUTING...',
            'signal': 'WARMING UP', 'signal_color': 'dim'
        }

    # Run inference
    prob = float(vwap_model.predict_proba(X)[:, 1][0])

    z = float(latest['vwap_zscore'].values[0]) if 'vwap_zscore' in latest.columns else np.nan
    hurst = float(latest['hurst_32'].values[0]) if 'hurst_32' in latest.columns else np.nan
    vr = float(latest['vol_ratio'].values[0]) if 'vol_ratio' in latest.columns else np.nan

    # Hurst regime context
    if pd.isna(hurst):
        regime_ctx = "COMPUTING..."
    elif hurst < 0.45:
        regime_ctx = "MEAN-REVERTING"
    elif hurst > 0.55:
        regime_ctx = "TRENDING"
    else:
        regime_ctx = "RANDOM WALK"

    # Signal logic — copilot recommendation
    if pd.isna(z) or pd.isna(prob):
        signal = "INSUFFICIENT DATA"
        signal_color = "dim"
    elif z <= -2.0 and prob >= 0.70:
        signal = "STATISTICAL EXTREME (DOWNSIDE)"
        signal_color = "cyan"
    elif z >= 2.0 and prob >= 0.70:
        signal = "STATISTICAL EXTREME (UPSIDE)"
        signal_color = "magenta"
    elif abs(z) >= 2.0 and prob >= 0.55:
        signal = "ELEVATED DEVIATION"
        signal_color = "yellow"
    else:
        signal = "NOMINAL (WITHIN 2σ)"
        signal_color = "dim"

    return {
        'vwap_zscore': z,
        'ml_probability': prob,
        'hurst': hurst,
        'vol_ratio': vr,
        'regime_context': regime_ctx,
        'signal': signal,
        'signal_color': signal_color,
    }


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

# ─────────────────────────────────────────────────────────────────────────────
# 3. SPEED OF TAPE MODEL (1M→15M, predicts 4H tape regime)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_tape_features_live(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute tape speed features from 1M live data (no labels)."""
    from tape_speed_features import compute_bar_activity, aggregate_to_15m, add_rolling_context, add_session_features

    df_active = compute_bar_activity(df_1m)
    df_15m = aggregate_to_15m(df_active)
    df_15m = add_rolling_context(df_15m)
    df_15m = add_session_features(df_15m)
    return df_15m


def train_speed_of_tape_model(asset: str = "NAS100"):
    """Train Speed of Tape LightGBM on all historical 1M data.
    Returns (model, feature_cols)."""
    from tape_speed_features import build_tape_dataset

    hist_1m_path = get_historical_1m_path(asset)
    if asset == "NAS100":
        from tape_speed_features import load_nq_1m as loader_1m
    else:
        from tape_speed_features import load_gold_1m as loader_1m

    df_1m = loader_1m(str(hist_1m_path))
    joined, feat_cols = build_tape_dataset(df_1m, asset_name=asset, verbose=False)

    X = joined[feat_cols].values.astype(np.float32)
    y = joined["tape_regime"].values.astype(np.int32)

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=7,
        subsample=0.8, colsample_bytree=0.7,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    model.fit(X, y)

    return model, feat_cols


def compute_speed_of_tape_state(df_live_1m: pd.DataFrame, model, feature_cols: list) -> dict:
    """Compute the current Speed of Tape state from live 1M data.

    Returns dict with keys:
        tape_regime_prob   : float — probability of Fast Tape
        active_ratio       : float — current fraction of price-changing bars
        active_ratio_ma20  : float — 20-bar trend of active ratio
        tv_zscore_20       : float — tick volume z-score vs 20-bar baseline
        regime_label       : str   — "FAST_TAPE" / "SLOW_TAPE" / "UNCERTAIN"
        regime_color       : str   — Rich color string
    """
    if len(df_live_1m) < 500:
        return {
            'tape_regime_prob': np.nan, 'active_ratio': np.nan,
            'active_ratio_ma20': np.nan, 'tv_zscore_20': np.nan,
            'regime_label': 'INSUFFICIENT DATA', 'regime_color': 'dim'
        }

    df_15m = _compute_tape_features_live(df_live_1m)

    if len(df_15m) < 20:
        return {
            'tape_regime_prob': np.nan, 'active_ratio': np.nan,
            'active_ratio_ma20': np.nan, 'tv_zscore_20': np.nan,
            'regime_label': 'WARMING UP', 'regime_color': 'dim'
        }

    latest = df_15m.iloc[[-1]]

    missing = [c for c in feature_cols if c not in latest.columns]
    for c in missing:
        latest[c] = np.nan

    X = latest[feature_cols].values

    if np.any(np.isnan(X)):
        ar = float(latest['active_ratio'].values[0]) if 'active_ratio' in latest.columns else np.nan
        ar_ma = float(latest['active_ratio_ma20'].values[0]) if 'active_ratio_ma20' in latest.columns else np.nan
        tvz = float(latest['tv_zscore_20'].values[0]) if 'tv_zscore_20' in latest.columns else np.nan
        return {
            'tape_regime_prob': np.nan, 'active_ratio': ar,
            'active_ratio_ma20': ar_ma, 'tv_zscore_20': tvz,
            'regime_label': 'COMPUTING...', 'regime_color': 'dim'
        }

    prob = float(model.predict_proba(X)[:, 1][0])

    ar = float(latest['active_ratio'].values[0]) if 'active_ratio' in latest.columns else np.nan
    ar_ma = float(latest['active_ratio_ma20'].values[0]) if 'active_ratio_ma20' in latest.columns else np.nan
    tvz = float(latest['tv_zscore_20'].values[0]) if 'tv_zscore_20' in latest.columns else np.nan

    if prob > 0.70:
        label = "FAST TAPE"
        color = "bright_green"
    elif prob < 0.30:
        label = "SLOW TAPE"
        color = "bright_red"
    else:
        label = "UNCERTAIN"
        color = "bright_yellow"

    return {
        'tape_regime_prob': prob,
        'active_ratio': ar,
        'active_ratio_ma20': ar_ma,
        'tv_zscore_20': tvz,
        'regime_label': label,
        'regime_color': color,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. MICRO-REGIME MODEL (1M, predicts 15-min tape regime)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_micro_features_live(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute micro-regime features from 1M live data (no labels)."""
    from micro_regime_features import compute_instant_features, compute_rolling_features, compute_lag_features, add_session_features

    df = compute_instant_features(df_1m)
    df["active_ratio"] = df["is_active"]
    df = compute_rolling_features(df)
    df = compute_lag_features(df)
    df = add_session_features(df)
    return df


def train_micro_regime_model(asset: str = "NAS100"):
    """Train Micro-Regime LightGBM on all historical 1M data.
    Returns (model, feature_cols)."""
    from micro_regime_features import build_micro_dataset

    hist_1m_path = get_historical_1m_path(asset)
    if asset == "NAS100":
        from tape_speed_features import load_nq_1m as loader_1m
    else:
        from tape_speed_features import load_gold_1m as loader_1m

    df_1m = loader_1m(str(hist_1m_path))
    subsample = 0.5 if asset == "GOLD" else 1.0
    joined, feat_cols = build_micro_dataset(df_1m, asset_name=asset,
                                            subsample_frac=subsample, verbose=False)

    X = joined[feat_cols].values.astype(np.float32)
    y = joined["micro_regime"].values.astype(np.int32)

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=7,
        subsample=0.8, colsample_bytree=0.7,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    model.fit(X, y)

    return model, feat_cols


def compute_micro_regime_state(df_live_1m: pd.DataFrame, model, feature_cols: list) -> dict:
    """Compute the current Micro-Regime state from live 1M data.

    Returns dict with keys:
        micro_regime_prob   : float — probability of Fast Tape next 15 min
        active_ratio_15     : float — current 15-bar active ratio
        tv_momentum         : float — tick volume momentum
        silent_ratio_15     : float — fraction of silent bars
        regime_label        : str   — "FAST" / "SLOW" / "UNCERTAIN"
        regime_color        : str   — Rich color string
    """
    if len(df_live_1m) < 100:
        return {
            'micro_regime_prob': np.nan, 'active_ratio_15': np.nan,
            'tv_momentum': np.nan, 'silent_ratio_15': np.nan,
            'regime_label': 'INSUFFICIENT DATA', 'regime_color': 'dim'
        }

    df_feat = _compute_micro_features_live(df_live_1m)

    if len(df_feat) < 60:
        return {
            'micro_regime_prob': np.nan, 'active_ratio_15': np.nan,
            'tv_momentum': np.nan, 'silent_ratio_15': np.nan,
            'regime_label': 'WARMING UP', 'regime_color': 'dim'
        }

    latest = df_feat.iloc[[-1]]

    missing = [c for c in feature_cols if c not in latest.columns]
    for c in missing:
        latest[c] = np.nan

    X = latest[feature_cols].values

    if np.any(np.isnan(X)):
        ar15 = float(latest['active_ratio_15'].values[0]) if 'active_ratio_15' in latest.columns else np.nan
        tvm = float(latest['tv_momentum'].values[0]) if 'tv_momentum' in latest.columns else np.nan
        sr15 = float(latest['silent_ratio_15'].values[0]) if 'silent_ratio_15' in latest.columns else np.nan
        return {
            'micro_regime_prob': np.nan, 'active_ratio_15': ar15,
            'tv_momentum': tvm, 'silent_ratio_15': sr15,
            'regime_label': 'COMPUTING...', 'regime_color': 'dim'
        }

    prob = float(model.predict_proba(X)[:, 1][0])

    ar15 = float(latest['active_ratio_15'].values[0]) if 'active_ratio_15' in latest.columns else np.nan
    tvm = float(latest['tv_momentum'].values[0]) if 'tv_momentum' in latest.columns else np.nan
    sr15 = float(latest['silent_ratio_15'].values[0]) if 'silent_ratio_15' in latest.columns else np.nan

    if prob > 0.70:
        label = "FAST"
        color = "bright_green"
    elif prob < 0.30:
        label = "SLOW"
        color = "bright_red"
    else:
        label = "UNCERTAIN"
        color = "bright_yellow"

    return {
        'micro_regime_prob': prob,
        'active_ratio_15': ar15,
        'tv_momentum': tvm,
        'silent_ratio_15': sr15,
        'regime_label': label,
        'regime_color': color,
    }


if __name__ == "__main__":
    generate_briefing()
