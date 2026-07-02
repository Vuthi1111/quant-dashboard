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
    if "TickVolume" in df.columns:
        tv = df["TickVolume"].astype(float)
        return tv.pct_change().rolling(window).mean()
    return pd.Series(np.nan, index=df.index)


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
        label_dir  : 1 (up) / 0 (down) — binary classification target
        label_ret  : raw log return — regression target
        label_abs  : absolute log return — for position sizing confidence
    """
    fwd_ret = np.log(df["Close"].shift(-horizon) / df["Close"])
    labels  = pd.DataFrame(index=df.index)
    labels["label_ret"]  = fwd_ret
    labels["label_abs"]  = fwd_ret.abs()
    labels["label_dir"]  = (fwd_ret > 0).astype(int)
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
        
    return feat
