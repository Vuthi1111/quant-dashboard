"""
feature_engineering.py
═══════════════════════════════════════════════════════════════════════════════
Adaptive Supervised Walk-Forward Pipeline — NAS100
Feature Engineering Layer

All features computed in a vectorised manner using only past information (no
forward-looking windows). This module is called INSIDE each walk-forward fold
after the train/val/test split so that no future statistics contaminate training.
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_mt5_csv(path: str, sep: str = "\t", _is_retry: bool = False) -> pd.DataFrame:
    """Load an MT5-exported OHLCV CSV, handle both tab and comma delimiters."""
    try:
        df = pd.read_csv(path, sep=sep)
        df.columns = df.columns.str.strip().str.lower()
        
        # If it read it as one column, the delimiter is wrong
        if len(df.columns) < 4:
            raise ValueError("Insufficient columns, likely wrong delimiter.")
            
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"],
                                            format="%Y.%m.%d %H:%M:%S",
                                            errors="coerce")
            df.set_index("datetime", inplace=True)
        elif "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df.set_index("time", inplace=True)
            
        df.sort_index(inplace=True)
        if "close" in df.columns:
            df = df[df["close"] > 0].copy()
        
        # Capitalize them back to what the rest of the code expects
        rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", 
                      "tick_volume": "Tick_Volume", "volume": "Volume"}
        df.rename(columns=rename_map, inplace=True)
        
        # Spread might be missing in some historical data exports
        if "spread" not in df.columns:
            df["spread"] = 0
            
        return df
    except Exception as e:
        if not _is_retry:
            return load_mt5_csv(path, sep="," if sep == "\t" else "\t", _is_retry=True)
        else:
            print(f"CRITICAL ERROR loading {path}: {e}")
            raise e


def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample any OHLCV dataframe to 4-hour candles."""
    if df.empty:
        return df
        
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }
    if 'Tick_Volume' in df.columns:
        agg_dict['Tick_Volume'] = 'sum'
    if 'Volume' in df.columns:
        agg_dict['Volume'] = 'sum'
    if 'Spread' in df.columns:
        agg_dict['Spread'] = 'mean'
        
    # Resample
    df_4h = df.resample('4H').agg(agg_dict)
    
    # Drop rows where there is no data
    df_4h.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    return df_4h

def resample_to_15m(df: pd.DataFrame) -> pd.DataFrame:
    """Resample any OHLCV dataframe to 15-minute candles."""
    if df.empty:
        return df
        
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }
    if 'Tick_Volume' in df.columns:
        agg_dict['Tick_Volume'] = 'sum'
    if 'Volume' in df.columns:
        agg_dict['Volume'] = 'sum'
    if 'Spread' in df.columns:
        agg_dict['Spread'] = 'mean'
        
    # Resample
    df_15m = df.resample('15Min').agg(agg_dict)
    
    # Drop rows where there is no data
    df_15m.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
    return df_15m


def load_news_mask(json_path: str, index: pd.DatetimeIndex,
                   buffer_min: int = 2) -> pd.Series:
    """
    Convert ForexFactory high-impact news timestamps to a boolean Series.
    Any bar within ±buffer_min minutes of a news release is flagged True.
    """
    try:
        with open(json_path, "r") as f:
            raw = json.load(f)
        news_times = []
        for item in raw:
            for k in ("date", "datetime", "time", "timestamp"):
                if k in item:
                    try:
                        news_times.append(pd.to_datetime(item[k]))
                    except Exception:
                        pass
                    break
        mask = pd.Series(False, index=index)
        for nt in news_times:
            window = pd.Timedelta(minutes=buffer_min)
            mask |= (index >= nt - window) & (index <= nt + window)
        return mask
    except Exception:
        return pd.Series(False, index=index)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  VOLATILITY ESTIMATORS
# ─────────────────────────────────────────────────────────────────────────────

def garman_klass(df: pd.DataFrame, window: int = 1) -> pd.Series:
    """Garman-Klass high-frequency volatility estimator."""
    hl = np.log(df["High"] / df["Low"]) ** 2
    co = np.log(df["Close"] / df["Open"]) ** 2
    gk = np.sqrt(0.5 * hl - (2 * np.log(2) - 1) * co)
    return gk.rolling(window).mean() if window > 1 else gk


def parkinson(df: pd.DataFrame, window: int = 1) -> pd.Series:
    """Parkinson range-based volatility estimator."""
    pk = np.sqrt((1 / (4 * np.log(2))) * (np.log(df["High"] / df["Low"])) ** 2)
    return pk.rolling(window).mean() if window > 1 else pk


def rogers_satchell(df: pd.DataFrame, window: int = 1) -> pd.Series:
    """Rogers-Satchell estimator — handles trend better than Parkinson."""
    rs = (np.log(df["High"] / df["Close"]) * np.log(df["High"] / df["Open"]) +
          np.log(df["Low"] / df["Close"]) * np.log(df["Low"] / df["Open"]))
    rs = np.sqrt(rs.clip(lower=0))
    return rs.rolling(window).mean() if window > 1 else rs


def har_decomposition(log_ret: pd.Series,
                      d_window: int = 1,
                      w_window: int = 5,
                      m_window: int = 22) -> pd.DataFrame:
    """
    Heterogeneous Autoregressive (HAR) decomposition.
    Decomposes realized variance into daily, weekly, monthly components.
    Returns a DataFrame of three HAR components.
    """
    rv = log_ret ** 2
    har = pd.DataFrame(index=log_ret.index)
    har["HAR_D"] = rv.rolling(d_window).mean()
    har["HAR_W"] = rv.rolling(w_window).mean()
    har["HAR_M"] = rv.rolling(m_window).mean()
    return har


def rm2006_ewma(log_ret: pd.Series, alpha: float = 0.06) -> pd.Series:
    """RiskMetrics 2006 EWMA volatility proxy."""
    return log_ret.ewm(alpha=alpha, adjust=False).std()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MICROSTRUCTURE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def range_ratio(df: pd.DataFrame) -> pd.Series:
    """Intrabar range normalised by close price."""
    return (df["High"] - df["Low"]) / df["Close"]


def tick_vol_acceleration(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Rate of change of tick volume — proxy for order flow intensity.
    Requires TickVolume column (present in MT5 data).
    """
    if "Tick_Volume" in df.columns:
        tv = df["Tick_Volume"].astype(float)
        return tv.pct_change().rolling(window).mean()
    return pd.Series(0, index=df.index)


def vwap_deviation(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Rolling VWAP deviation proxy using typical price and tick volume.
    (TP - VWAP) / VWAP — measures if price is extended above/below flow.
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["TickVolume"].astype(float) if "TickVolume" in df.columns else pd.Series(1, index=df.index)
    cumtv = (tp * vol).rolling(window).sum()
    cumv  = vol.rolling(window).sum()
    vwap  = cumtv / cumv.replace(0, np.nan)
    return (tp - vwap) / vwap.replace(0, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TIME / CALENDAR FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def cyclical_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Encode hour-of-day and day-of-week as continuous cyclical features
    using sin/cos pairs to preserve periodicity.
    """
    cal = pd.DataFrame(index=index)
    h = index.hour
    d = index.dayofweek
    cal["hour_sin"] = np.sin(2 * np.pi * h / 24)
    cal["hour_cos"] = np.cos(2 * np.pi * h / 24)
    cal["dow_sin"]  = np.sin(2 * np.pi * d / 5)
    cal["dow_cos"]  = np.cos(2 * np.pi * d / 5)
    cal["month_sin"] = np.sin(2 * np.pi * index.month / 12)
    cal["month_cos"] = np.cos(2 * np.pi * index.month / 12)
    return cal


# ─────────────────────────────────────────────────────────────────────────────
# 5.  LABEL CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_labels(df: pd.DataFrame,
                 horizon: int = 1,
                 min_move_pct: float = 0.0) -> pd.DataFrame:
    """
    Construct direction and magnitude labels.

    Parameters
    ----------
    df          : OHLCV DataFrame
    horizon     : forward bars to compute return over (default: 1 bar)
    min_move_pct: minimum % move to generate a label (below this → NaN, skip)

    Returns
    -------
    DataFrame with columns:
        label_dir         : 1 (up) / 0 (down) — binary classification target
        label_ret         : raw log return — regression target
        label_abs         : absolute log return — for position sizing confidence
        label_vol_adj_ret : EWMA True Range normalized return — robust regression target
    """
    fwd_ret = np.log(df["Close"].shift(-horizon) / df["Close"])
    
    # Calculate True Range for EWMA normalization
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # EWMA of True Range (normalized as a percentage of close price)
    # Using span=20 which on 4H is roughly 3.3 days, but EWMA reacts much faster to recent spikes
    ewma_tr_pct = (tr / df["Close"]).ewm(span=20, min_periods=20).mean()
    
    labels  = pd.DataFrame(index=df.index)
    labels["label_ret"]  = fwd_ret
    labels["label_abs"]  = fwd_ret.abs()
    labels["label_dir"]  = (fwd_ret > 0).astype(int)
    
    # Volatility-Adjusted Target (Z-score like)
    # Divided by the EWMA TR percentage to normalize returns against current market speed
    labels["label_vol_adj_ret"] = fwd_ret / ewma_tr_pct
    
    if min_move_pct > 0:
        too_small = fwd_ret.abs() < (min_move_pct / 100)
        labels.loc[too_small, "label_dir"] = np.nan
        
    return labels


def build_vol_regime_labels(df: pd.DataFrame,
                            forward_bars: int = 4,
                            bar_offset: int = 1,
                            regime_pct_high: float = 0.70,
                            regime_pct_low: float  = 0.30,
                            rolling_baseline: int  = 480) -> pd.DataFrame:
    """
    Volatility Regime Classification Label (Pivot 1).

    Predicts whether a FUTURE window will be HIGH or LOW realized volatility.

    Parameters
    ----------
    df               : OHLCV DataFrame (1H bars)
    forward_bars     : number of bars in the forward vol window (default 4 = 4H)
    bar_offset       : bars to skip before the window starts (default 1).
                       Set to 1 to add a 1-bar buffer so the label window
                       starts at t+2 instead of t+1, breaking the same-bar
                       GK candlestick boundary overlap artifact.
    regime_pct_high  : top percentile threshold -> label = 1 (High Vol)
    regime_pct_low   : bottom percentile threshold -> label = 0 (Low Vol)
    rolling_baseline : rolling window for percentile calculation (480 bars = 20 trading days)
    """
    log_hl = np.log(df["High"] / df["Low"]) ** 2
    log_co = np.log(df["Close"] / df["Open"]) ** 2
    bar_gk = np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)

    # Forward window starts at t+(bar_offset+1) and spans forward_bars bars
    # bar_offset=1 means window is t+2 ... t+(forward_bars+1)
    # This cleanly separates the current bar's OHLCV from the label's OHLCV
    fwd_rv = (
        bar_gk
        .shift(-(bar_offset + 1))           # shift past the offset
        .rolling(forward_bars).mean()        # roll forward_bars bars
        .shift(-(forward_bars - 1))          # align to bar t
    )

    # Rolling percentile thresholds computed on PAST data only
    roll_high = fwd_rv.shift(1).rolling(rolling_baseline).quantile(regime_pct_high)
    roll_low  = fwd_rv.shift(1).rolling(rolling_baseline).quantile(regime_pct_low)

    labels = pd.DataFrame(index=df.index)
    labels["fwd_rv"]     = fwd_rv
    labels["vol_regime"] = np.nan
    labels.loc[fwd_rv >= roll_high, "vol_regime"] = 1   # High Vol
    labels.loc[fwd_rv <= roll_low,  "vol_regime"] = 0   # Low Vol
    # Middle 40% stays NaN -- discarded during training

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MASTER FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame,
                   news_mask_path: str = None,
                   news_buffer_min: int = 2) -> pd.DataFrame:
    """
    Assemble the full feature matrix from raw OHLCV data.

    leakage — features reflect information available at bar open only.
    """
    feat = pd.DataFrame(index=df.index)

    # — Log returns & lags
    lr = np.log(df["Close"] / df["Close"].shift(1))
    for lag in [1, 2, 3, 4, 8, 16, 32]:
        feat[f"ret_lag{lag}"] = lr.shift(lag)

    # — Volatility estimators (rolling)
    for w in [5, 10, 20]:
        feat[f"GK_{w}"]  = garman_klass(df, w).shift(1)
        feat[f"PK_{w}"]  = parkinson(df, w).shift(1)
        feat[f"RS_{w}"]  = rogers_satchell(df, w).shift(1)

    # — HAR components
    har = har_decomposition(lr)
    feat["HAR_D"] = har["HAR_D"].shift(1)
    feat["HAR_W"] = har["HAR_W"].shift(1)
    feat["HAR_M"] = har["HAR_M"].shift(1)

    # — EWMA / RiskMetrics
    feat["RM2006"]   = rm2006_ewma(lr).shift(1)
    feat["HV_20"]    = lr.rolling(20).std().shift(1)
    feat["MA120_vol"]= feat["HV_20"].rolling(120).mean()

    # — Vol ratio: intraday vs structural
    feat["vol_ratio"] = feat["GK_10"] / (feat["MA120_vol"] + 1e-9)

    # — Microstructure
    feat["range_ratio"]       = range_ratio(df).shift(1)
    feat["tickvol_accel"]     = tick_vol_acceleration(df).shift(1)
    feat["vwap_dev"]          = vwap_deviation(df).shift(1)

    # — Momentum proxies
    feat["roc_5"]  = (df["Close"] / df["Close"].shift(5) - 1).shift(1)
    feat["roc_20"] = (df["Close"] / df["Close"].shift(20) - 1).shift(1)

    # — RSI proxy (14-bar)
    delta    = lr.copy()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta).clip(lower=0).rolling(14).mean()
    rs       = gain / (loss + 1e-9)
    feat["rsi_14"] = (100 - 100 / (1 + rs)).shift(1)

    # — Bollinger Band position
    ma20     = df["Close"].rolling(20).mean()
    std20    = df["Close"].rolling(20).std()
    feat["bb_pos"] = ((df["Close"] - ma20) / (2 * std20 + 1e-9)).shift(1)

    # — Calendar / time
    cal = cyclical_time_features(df.index)
    for col in cal.columns:
        feat[col] = cal[col]

    # — News mask
    if news_mask_path:
        feat["news_flag"] = load_news_mask(news_mask_path, df.index,
                                           news_buffer_min).astype(int)
    else:
        feat["news_flag"] = 0
        
    # — Macro Data Fetching (VIX, DXY, TNX, TIPS, GLD, COT)
    try:
        from macro_data_fetcher import merge_macro_features
        df_macro = merge_macro_features(df)
        macro_cols = [
            "macro_vix", "macro_vix_pct", 
            "macro_dxy", "macro_dxy_pct", 
            "macro_tnx", "macro_tnx_pct", 
            "macro_hyg", "macro_hyg_pct",
            "macro_tips", "macro_tips_pct",
            "gld_volume", "gld_volume_pct",
            "cot_mm_net_long", "cot_mm_pct_oi"
        ]
        for col in macro_cols:
            if col in df_macro.columns:
                feat[col] = df_macro[col]
            else:
                feat[col] = 0.0 # Fallback if fetch fails
                
        # — Institutional Math Engine Features
        try:
            import quant_math_engine as qme
            
            # 1. Rolling Hurst Exponent (Trend vs Chop) on 4H bars
            # 90 bars = 3 trading weeks (increased for stability as per Claude's feedback)
            feat["hurst_90"] = df["Close"].rolling(90).apply(lambda x: qme.calculate_hurst(pd.Series(x), min_window=30))
            
            # 2. Cointegration Z-Score (Fair Value Gap)
            if "macro_dxy" in feat.columns and "macro_tips" in feat.columns:
                coint_df = qme.calculate_dynamic_fair_value(
                    df["Close"], 
                    feat["macro_dxy"], 
                    feat["macro_tips"], 
                    window=120 # 20 days
                )
                feat["coint_z_score"] = coint_df["coint_z_score"]
                feat["coint_fv"] = coint_df["coint_fv"]
                feat["coint_std"] = coint_df["coint_std"]
            
            # 3. Ornstein-Uhlenbeck Process Parameters
            # We calculate this on a rolling 90-bar window for stability
            returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
            
            # Pre-allocate OU columns
            feat["ou_theta"] = np.nan
            feat["ou_mu"] = np.nan
            feat["ou_sigma"] = np.nan
            feat["ou_halflife"] = np.nan
            
            for i in range(90, len(returns)):
                idx = returns.index[i]
                window_ret = returns.iloc[i-90:i]
                theta, mu, sigma, hl = qme.calculate_ou_parameters(window_ret)
                
                feat.loc[idx, "ou_theta"] = theta
                feat.loc[idx, "ou_mu"] = mu
                feat.loc[idx, "ou_sigma"] = sigma
                feat.loc[idx, "ou_halflife"] = hl
                
        except Exception as e_math:
            print(f"[Warning] Failed to calculate Math Engine features: {e_math}")
            
    except Exception as e:
        print(f"[Warning] Failed to fetch macro features: {e}")
        # Add zeroes if fail to prevent pipeline crash
        macro_cols = [
            "macro_vix", "macro_vix_pct", 
            "macro_dxy", "macro_dxy_pct", 
            "macro_tnx", "macro_tnx_pct", 
            "macro_hyg", "macro_hyg_pct",
            "macro_tips", "macro_tips_pct",
            "gld_volume", "gld_volume_pct",
            "cot_mm_net_long", "cot_mm_pct_oi"
        ]
        for col in macro_cols:
            feat[col] = 0.0

    return feat


def create_breakout_labels(df, k=1.5):
    """
    Creates First Passage Time (Triple-Barrier) labels for the Macro Breakout process.
    Trades WITH the momentum away from the cointegration fair value.
    """
    print("    [Labels] Generating First Passage Time (FPT) labels...")
    
    labels = pd.Series(np.nan, index=df.index)
    horizons = pd.Series(0, index=df.index)
    
    # Pre-calculate components for speed
    if 'coint_z_score' not in df.columns:
        df['breakout_label'] = labels
        df['fpt_horizon'] = horizons
        df['uniqueness_weight'] = 1.0
        return df
        
    z_scores = df['coint_z_score'].values
    coint_fvs = df['coint_fv'].values
    coint_stds = df['coint_std'].values
    closes = df['Close'].values
    halflives = df['ou_halflife'].values
    
    # To calculate uniqueness weights, track active trade ranges
    active_ranges = []
    
    for i in range(len(df)):
        z = z_scores[i]
        
        # 1. Entry Filter (Must be > 1.5 stdev from cointegration fair value)
        if pd.isna(z) or abs(z) <= 1.5:
            continue
            
        log_fv = coint_fvs[i]
        log_std = coint_stds[i]
        
        if pd.isna(log_fv) or pd.isna(log_std):
            continue
            
        # Stop is the old target (Fair Value)
        stop_price = np.exp(log_fv)
        
        if z > 0: # Long Breakout setup (price is already above mean, betting it goes higher)
            log_target = log_fv + (abs(z) + k) * log_std
        else:     # Short Breakout setup (price is already below mean, betting it goes lower)
            log_target = log_fv - (abs(z) + k) * log_std
            
        target_price = np.exp(log_target)
            
        # 3. Dynamic Horizon
        hl = halflives[i] if not pd.isna(halflives[i]) and halflives[i] > 0 else 10
        W_t = int(np.ceil(hl * 1.5))
        W_t = max(4, min(W_t, 20)) # Clip between 4 and 20 bars
        
        horizons.iloc[i] = W_t
        end_idx = min(i + W_t, len(df) - 1)
        
        forward_prices = closes[i+1 : end_idx+1]
        if len(forward_prices) == 0:
            labels.iloc[i] = 0
            continue
            
        # 4. Check Barrier Hits (using Close prices for simplicity/robustness)
        if z > 0: # Long Breakout (Price is moving UP)
            hit_target = np.any(forward_prices >= target_price)
            hit_stop = np.any(forward_prices <= stop_price)
            idx_target = np.argmax(forward_prices >= target_price) if hit_target else 9999
            idx_stop = np.argmax(forward_prices <= stop_price) if hit_stop else 9999
        else: # Short Breakout (Price is moving DOWN)
            hit_target = np.any(forward_prices <= target_price)
            hit_stop = np.any(forward_prices >= stop_price)
            idx_target = np.argmax(forward_prices <= target_price) if hit_target else 9999
            idx_stop = np.argmax(forward_prices >= stop_price) if hit_stop else 9999
            
        # 5. Path-Dependent Assignment
        if hit_target and idx_target <= idx_stop:
            labels.iloc[i] = 1
            actual_duration = idx_target + 1
        else:
            labels.iloc[i] = 0
            actual_duration = idx_stop + 1 if hit_stop and idx_stop < 9999 else len(forward_prices)
            
        # Store range for uniqueness weighting
        active_ranges.append((i, min(i + actual_duration, len(df) - 1)))
        
    df['breakout_label'] = labels
    df['fpt_horizon'] = horizons
    
    # Calculate Uniqueness Weights
    print("    [Labels] Calculating Sample Uniqueness Weights...")
    # c_t is the number of concurrent active labels at time t
    c_t = np.zeros(len(df))
    for start, end in active_ranges:
        c_t[start:end+1] += 1
        
    # u_i is the average of 1/c_t over the life of the trade
    u_i = pd.Series(np.nan, index=df.index)
    idx_range = 0
    for i in range(len(df)):
        if not pd.isna(labels.iloc[i]):
            start, end = active_ranges[idx_range]
            trade_c_t = c_t[start:end+1]
            u_i.iloc[i] = np.mean(1.0 / trade_c_t) if len(trade_c_t) > 0 else 1.0
            idx_range += 1
            
    df['uniqueness_weight'] = u_i
    
    return df

def build_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build intraday specific features (VWAP, RTV, Hurst, Time of Day)"""
    feat = pd.DataFrame(index=df.index)
    
    # 1. Time of Day Encodings
    hours = df.index.hour + df.index.minute / 60.0
    feat['hour_sin'] = pd.Series(np.sin(2 * np.pi * hours / 24.0), index=df.index).shift(1)
    feat['hour_cos'] = pd.Series(np.cos(2 * np.pi * hours / 24.0), index=df.index).shift(1)
    
    # 2. Daily Anchored VWAP
    df_temp = df.copy()
    df_temp['date'] = df_temp.index.date
    df_temp['typ_price'] = (df_temp['High'] + df_temp['Low'] + df_temp['Close']) / 3.0
    df_temp['tv'] = df_temp['typ_price'] * df_temp['Tick_Volume']
    
    cum_vol = df_temp.groupby('date')['Tick_Volume'].cumsum()
    cum_tv = df_temp.groupby('date')['tv'].cumsum()
    
    vwap = cum_tv / cum_vol.replace(0, np.nan)
    
    # VWAP Variance = E[P^2] - E[P]^2
    df_temp['tv2'] = (df_temp['typ_price'] ** 2) * df_temp['Tick_Volume']
    cum_tv2 = df_temp.groupby('date')['tv2'].cumsum()
    vwap_var = (cum_tv2 / cum_vol.replace(0, np.nan)) - (vwap ** 2)
    vwap_std = np.sqrt(vwap_var.clip(lower=1e-9))
    
    feat['vwap'] = vwap.shift(1)
    feat['vwap_std'] = vwap_std.shift(1)
    feat['vwap_zscore'] = ((df['Close'] - vwap) / vwap_std.replace(0, np.nan)).shift(1)
    
    # 3. Intraday Hurst Exponent (rolling 16 bars)
    import quant_math_engine as qme
    try:
        feat['hurst_16'] = df['Close'].rolling(32).apply(lambda x: qme.calculate_hurst(pd.Series(x), min_window=8)).shift(1)
    except:
        feat['hurst_16'] = np.nan
        
    # 4. Relative Tick Volume (RTV) - rolling 4 bars vs rolling 96 bars
    vol_4 = df['Tick_Volume'].rolling(4).sum()
    vol_96_avg = df['Tick_Volume'].rolling(96).mean() * 4
    feat['rtv'] = (vol_4 / vol_96_avg.replace(0, np.nan)).shift(1)
    
    return feat

def create_vwap_scalp_labels(df: pd.DataFrame, max_horizon: int = 16) -> pd.DataFrame:
    """
    Create Triple-Barrier labels for VWAP Mean Reversion.
    Target: Return to VWAP.
    Stop: Hit 3-sigma deviation from VWAP.
    Time Stop: max_horizon bars (default 16).
    """
    print(f"    [Labels] Generating VWAP Scalp labels (Horizon: {max_horizon})...")
    
    labels = pd.Series(np.nan, index=df.index)
    horizons = pd.Series(0, index=df.index)
    
    if 'vwap' not in df.columns or 'vwap_zscore' not in df.columns:
        df['vwap_label'] = labels
        df['fpt_horizon'] = horizons
        df['uniqueness_weight'] = 1.0
        return df
        
    prices = df['Close'].values
    vwap = df['vwap'].values
    vwap_std = df['vwap_std'].values
    vwap_zscore = df['vwap_zscore'].values
    
    active_ranges = []
    
    for i in range(len(df) - max_horizon):
        z = vwap_zscore[i]
        
        # Only take trades at extreme deviations
        if pd.isna(z) or abs(z) < 2.0:
            continue
            
        is_long = z <= -2.0  # Price is below VWAP, we expect it to revert UP to VWAP
        
        entry_price = prices[i]
        
        # We don't have dynamic targets since VWAP changes, so we evaluate bar by bar
        forward_prices = prices[i+1 : i+1+max_horizon]
        forward_vwap = vwap[i+1 : i+1+max_horizon]
        forward_vwap_std = vwap_std[i+1 : i+1+max_horizon]
        
        hit_target = False
        hit_stop = False
        idx_target = 9999
        idx_stop = 9999
        
        for j, (p, v, v_std) in enumerate(zip(forward_prices, forward_vwap, forward_vwap_std)):
            if is_long:
                target_price = v
                stop_price = v - (3.0 * v_std)
                if p >= target_price:
                    hit_target = True
                    idx_target = j
                    break
                elif p <= stop_price:
                    hit_stop = True
                    idx_stop = j
                    break
            else:
                target_price = v
                stop_price = v + (3.0 * v_std)
                if p <= target_price:
                    hit_target = True
                    idx_target = j
                    break
                elif p >= stop_price:
                    hit_stop = True
                    idx_stop = j
                    break
                    
        if hit_target and idx_target <= idx_stop:
            labels.iloc[i] = 1
            actual_duration = idx_target + 1
        elif hit_stop and idx_stop < idx_target:
            labels.iloc[i] = 0
            actual_duration = idx_stop + 1
        else:
            labels.iloc[i] = 0  # Time stop
            actual_duration = max_horizon
            
        active_ranges.append((i, min(i + actual_duration, len(df) - 1)))
        horizons.iloc[i] = actual_duration
        
    df['vwap_label'] = labels
    df['fpt_horizon'] = horizons
    
    # Calculate Uniqueness Weights
    print("    [Labels] Calculating Sample Uniqueness Weights...")
    c_t = np.zeros(len(df))
    for start, end in active_ranges:
        c_t[start:end+1] += 1
        
    u_i = pd.Series(np.nan, index=df.index)
    idx_range = 0
    for i in range(len(df)):
        if not pd.isna(labels.iloc[i]):
            start, end = active_ranges[idx_range]
            trade_c_t = c_t[start:end+1]
            u_i.iloc[i] = np.mean(1.0 / trade_c_t) if len(trade_c_t) > 0 else 1.0
            idx_range += 1
            
    df['uniqueness_weight'] = u_i
    
    return df
