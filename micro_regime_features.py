"""
micro_regime_features.py
═══════════════════════════════════════════════════════════════════════════════
Micro-Regime — 1M Feature Engineering for 15M Prediction

Concept
-------
Stay entirely at the 1-MINUTE level. Each 1M bar becomes one row in the
feature matrix. The label answers:

    "Will the next 15 minutes be a FAST or SLOW tape?"

This is the scalper/execution-algo use-case:
  - ORB traders: skip entries when micro-regime = SLOW (choppy/illiquid)
  - VWAP slicers: pause execution when micro-regime = SLOW (wide spread, no flow)
  - Momentum scalpers: only enter when micro-regime = FAST (active printing)

Key Design Decisions
--------------------
1. NO resampling — we keep raw 1M bars as rows.
2. Features are rolling windows computed on the 1M series itself.
3. All features are LAGGED (shift+1) to prevent any look-ahead.
4. Label = average active_ratio over the NEXT 15 bars (15 minutes).
5. Bottom 30% = Slow Tape (0), Top 30% = Fast Tape (1), middle discarded.

Features (all computed at 1M resolution)
-----------------------------------------
Instant bar:
  is_active         : 1 if Close changed vs prior bar
  tick_vol          : raw TickVolume for this bar
  bar_range         : High - Low (pip range)
  range_per_tick    : bar_range / tick_vol (Amihud illiquidity)
  tick_density      : tick_vol / (bar_range + ε)
  body_ratio        : |Close - Open| / (High - Low + ε)
  is_silent         : 1 if TickVolume == 0

Rolling windows (5, 15, 30, 60 bars = 5m, 15m, 30m, 1h):
  active_ratio_5    : fraction of price-changing bars in last 5 min
  active_ratio_15   : same, 15 min  ← most predictive
  active_ratio_30   : same, 30 min (trend context)
  active_ratio_60   : same, 60 min (session context)
  tv_mean_5         : mean TickVolume last 5 bars
  tv_mean_15        : mean TickVolume last 15 bars
  tv_cv_15          : coefficient of variation last 15 bars (burstiness)
  tv_sum_15         : total tick flow last 15 bars
  tv_sum_60         : total tick flow last 60 bars
  tv_momentum       : tv_sum_15 / (tv_sum_60/4 + ε) — recent vs baseline
  range_sum_15      : total pip movement last 15 bars
  silent_ratio_15   : fraction of zero-tick bars in last 15 bars
  silent_streak     : consecutive silent bars (liquidity vacuum signal)

Z-scores (vs 60-bar and 240-bar baselines):
  tv_zscore_60      : z-score of current bar tick_vol vs 60-bar mean/std
  tv_zscore_240     : z-score vs 4-hour baseline
  ar_zscore_60      : z-score of active_ratio_15 vs 60-bar baseline

Lag features (t-1, t-3, t-5):
  active_ratio_15_lag1/3/5
  tv_mean_15_lag1/3/5

Session / time:
  hour_sin/cos, dow_sin/cos
  is_london, is_ny, is_overlap, is_asian
  minutes_to_close  : minutes until typical session close (21:00 UTC)
  is_first_15min    : first 15 min after session open (vol surge zone)
  is_last_15min     : last 15 min before session close
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import sys, os

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)

# Reuse data loaders from tape_speed_features
from tape_speed_features import load_nq_1m, load_gold_1m


# ─────────────────────────────────────────────────────────────────────────────
# 1. PER-BAR INSTANT FEATURES (1M resolution)
# ─────────────────────────────────────────────────────────────────────────────

def compute_instant_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Compute single-bar microstructure features.
    No look-ahead — each feature uses only data available AT bar close.
    """
    df = df_1m[["Open", "High", "Low", "Close", "TickVolume"]].copy()

    # Activity flag
    df["is_active"]   = (df["Close"] != df["Close"].shift(1)).astype(np.float32)
    df.iloc[0, df.columns.get_loc("is_active")] = 0.0

    # Tick volume
    df["tick_vol"]    = df["TickVolume"].astype(np.float32)
    df["is_silent"]   = (df["tick_vol"] == 0).astype(np.float32)

    # Price range features
    df["bar_range"]   = (df["High"] - df["Low"]).astype(np.float32)
    eps = 1e-8

    df["range_per_tick"] = (df["bar_range"] / (df["tick_vol"] + eps)).astype(np.float32)
    df["tick_density"]   = (df["tick_vol"] / (df["bar_range"] + eps)).astype(np.float32)

    # Body ratio: directional commitment (how much of the range is directional)
    body = (df["Close"] - df["Open"]).abs()
    df["body_ratio"] = (body / (df["bar_range"] + eps)).astype(np.float32)

    # Log tick volume (handles skew)
    df["log_tick_vol"] = np.log1p(df["tick_vol"]).astype(np.float32)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROLLING WINDOW FEATURES (1M resolution)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling statistics on the 1M series.
    Windows: 5, 15, 30, 60 bars (5m, 15m, 30m, 1h).
    """
    tv  = df["tick_vol"]
    act = df["is_active"]
    rng = df["bar_range"]
    sil = df["is_silent"]
    eps = 1e-8

    # ── Active ratio (fraction of price-changing bars) ──────────────────────
    for w in [5, 15, 30, 60]:
        df[f"active_ratio_{w}"] = act.rolling(w, min_periods=1).mean().astype(np.float32)

    # ── Tick volume rolling stats ────────────────────────────────────────────
    for w in [5, 15]:
        df[f"tv_mean_{w}"] = tv.rolling(w, min_periods=1).mean().astype(np.float32)

    df["tv_sum_15"]  = tv.rolling(15, min_periods=1).sum().astype(np.float32)
    df["tv_sum_60"]  = tv.rolling(60, min_periods=1).sum().astype(np.float32)
    df["tv_max_15"]  = tv.rolling(15, min_periods=1).max().astype(np.float32)

    # Coefficient of variation (burstiness) — 15-bar window
    tv_std15 = tv.rolling(15, min_periods=2).std()
    tv_mu15  = df["tv_mean_15"].replace(0, np.nan)
    df["tv_cv_15"] = (tv_std15 / tv_mu15).astype(np.float32)

    # Momentum: recent 15-bar flow vs 60-bar baseline (normalised)
    baseline = (df["tv_sum_60"] / 4.0).replace(0, np.nan)
    df["tv_momentum"] = (df["tv_sum_15"] / baseline).astype(np.float32)

    # ── Range / pip movement ─────────────────────────────────────────────────
    df["range_sum_15"]  = rng.rolling(15, min_periods=1).sum().astype(np.float32)
    df["range_mean_15"] = rng.rolling(15, min_periods=1).mean().astype(np.float32)

    # ── Silent bar streaks ───────────────────────────────────────────────────
    df["silent_ratio_15"] = sil.rolling(15, min_periods=1).mean().astype(np.float32)

    # Consecutive silent bars: cumsum reset on activity
    # Use numpy for speed
    is_sil = sil.values.astype(np.int32)
    streak = np.zeros(len(is_sil), dtype=np.float32)
    s = 0
    for i in range(len(is_sil)):
        if is_sil[i]:
            s += 1
        else:
            s = 0
        streak[i] = s
    df["silent_streak"] = streak

    # ── Z-scores ─────────────────────────────────────────────────────────────
    for w, col_name in [(60, "tv_zscore_60"), (240, "tv_zscore_240")]:
        roll_mu  = tv.rolling(w, min_periods=max(10, w // 4)).mean()
        roll_std = tv.rolling(w, min_periods=max(10, w // 4)).std().replace(0, np.nan)
        df[col_name] = ((tv - roll_mu) / roll_std).astype(np.float32)

    # Z-score of active_ratio_15 vs 60-bar rolling baseline
    ar15     = df["active_ratio_15"]
    ar_mu60  = ar15.rolling(60, min_periods=10).mean()
    ar_std60 = ar15.rolling(60, min_periods=10).std().replace(0, np.nan)
    df["ar_zscore_60"] = ((ar15 - ar_mu60) / ar_std60).astype(np.float32)

    # ── Acceleration: change in activity level ───────────────────────────────
    df["ar_accel"]   = (df["active_ratio_15"] - df["active_ratio_15"].shift(5)).astype(np.float32)
    df["tv_accel"]   = (df["tv_mean_15"]      - df["tv_mean_15"].shift(5)).astype(np.float32)

    # ── Log sums for skew reduction ──────────────────────────────────────────
    df["log_tv_sum_15"] = np.log1p(df["tv_sum_15"]).astype(np.float32)
    df["log_tv_sum_60"] = np.log1p(df["tv_sum_60"]).astype(np.float32)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. LAG FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag the most predictive features by 1, 3, 5 bars."""
    for col in ["active_ratio_15", "tv_mean_15", "ar_zscore_60", "tv_momentum"]:
        if col not in df.columns:
            continue
        for lag in [1, 3, 5]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag).astype(np.float32)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. SESSION / TIME FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Session flags and cyclical time encodings."""
    idx = df.index
    h   = idx.hour + idx.minute / 60.0

    df["hour_sin"]  = np.sin(2 * np.pi * h / 24.0).astype(np.float32)
    df["hour_cos"]  = np.cos(2 * np.pi * h / 24.0).astype(np.float32)
    df["dow_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 5.0).astype(np.float32)
    df["dow_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 5.0).astype(np.float32)

    df["is_london"]  = ((idx.hour >= 8)  & (idx.hour < 17)).astype(np.float32)
    df["is_ny"]      = ((idx.hour >= 14) & (idx.hour < 21)).astype(np.float32)
    df["is_overlap"] = ((idx.hour >= 14) & (idx.hour < 17)).astype(np.float32)
    df["is_asian"]   = ((idx.hour >= 0)  & (idx.hour < 8) ).astype(np.float32)

    # Minutes until standard NY close (21:00 UTC)
    minutes_since_midnight = idx.hour * 60 + idx.minute
    df["minutes_to_ny_close"] = np.clip(
        21 * 60 - minutes_since_midnight, 0, 24 * 60
    ).astype(np.float32)

    # First/last 15 minutes of London and NY sessions (vol surge/drain zones)
    london_open_min  = 8 * 60
    ny_open_min      = 14 * 60 + 30
    london_close_min = 16 * 60 + 30
    ny_close_min     = 21 * 60

    df["is_london_open_burst"]  = (
        (minutes_since_midnight >= london_open_min) &
        (minutes_since_midnight <  london_open_min + 15)
    ).astype(np.float32)

    df["is_ny_open_burst"] = (
        (minutes_since_midnight >= ny_open_min) &
        (minutes_since_midnight <  ny_open_min + 15)
    ).astype(np.float32)

    df["is_london_close_drain"] = (
        (minutes_since_midnight >= london_close_min - 15) &
        (minutes_since_midnight <  london_close_min)
    ).astype(np.float32)

    df["is_ny_close_drain"] = (
        (minutes_since_midnight >= ny_close_min - 15) &
        (minutes_since_midnight <  ny_close_min)
    ).astype(np.float32)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. LABEL CONSTRUCTION — NEXT 15-MINUTE TAPE REGIME
# ─────────────────────────────────────────────────────────────────────────────

def build_micro_labels(df: pd.DataFrame,
                       forward_bars:     int   = 15,   # 15 min ahead
                       bar_offset:       int   = 1,    # 1-bar buffer
                       regime_pct_high:  float = 0.70,
                       regime_pct_low:   float = 0.30,
                       rolling_baseline: int   = 480   # 8 hours = 480 x 1M bars
                       ) -> pd.DataFrame:
    """
    Label each 1M bar: will the NEXT 15 minutes be Fast or Slow tape?

    Signal: active_ratio_15 (rolling 15-bar active_ratio already computed)
    Forward window: bars t+bar_offset+1 through t+bar_offset+forward_bars

    Thresholds: rolling percentile over past 480 bars (8 hours of 1M data)
    to adapt to changing market conditions throughout the day.
    """
    signal = df["active_ratio"]  # raw 1M is_active rate; recompute forward mean

    # Forward average of active_ratio over next `forward_bars` bars
    fwd = (
        signal
        .shift(-(bar_offset + 1))
        .rolling(forward_bars).mean()
        .shift(-(forward_bars - 1))
    )

    # Rolling percentile thresholds — computed on PAST active_ratio_15 data only
    # (uses rolling backward signal, NOT fwd, to avoid self-referential label boundaries)
    roll_signal = df["active_ratio_15"].shift(1)
    roll_high = roll_signal.rolling(rolling_baseline).quantile(regime_pct_high)
    roll_low  = roll_signal.rolling(rolling_baseline).quantile(regime_pct_low)

    labels = pd.DataFrame(index=df.index)
    labels["fwd_active_ratio"] = fwd
    labels["micro_regime"]     = np.nan

    labels.loc[fwd >= roll_high, "micro_regime"] = 1.0  # Fast Tape
    labels.loc[fwd <= roll_low,  "micro_regime"] = 0.0  # Slow Tape
    # Middle 40% stays NaN — no clear edge, discarded

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 6. MASTER BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_micro_dataset(df_1m:           pd.DataFrame,
                        asset_name:      str   = "",
                        forward_bars:    int   = 15,
                        bar_offset:      int   = 1,
                        regime_pct_high: float = 0.70,
                        regime_pct_low:  float = 0.30,
                        rolling_baseline:int   = 480,
                        subsample_frac:  float = 1.0,   # optional row subsampling
                        verbose:         bool  = True
                        ) -> tuple[pd.DataFrame, list]:
    """
    Full pipeline: 1M raw → 1M feature matrix + 15M-ahead micro-regime labels.

    Parameters
    ----------
    subsample_frac : float (0 < x <= 1)
        Downsample the labeled dataset for faster WFV. Useful for GOLD (5M rows).
        Sampling is done AFTER labeling, using systematic (every-nth) selection
        to preserve temporal structure.

    Returns
    -------
    joined       : DataFrame with all features + 'micro_regime' label
    feature_cols : list of feature column names
    """
    tag = f"[{asset_name}]" if asset_name else ""

    if verbose:
        print(f"  {tag} Step 1/5 — Computing instant bar features ...")
    df = compute_instant_features(df_1m)

    # Keep 'active_ratio' as raw is_active for label building
    df["active_ratio"] = df["is_active"]

    if verbose:
        print(f"  {tag}   Rows            : {len(df):,}")
        active_pct = df["is_active"].mean()
        print(f"  {tag}   Price-change rate: {active_pct:.1%} of 1M bars")

    if verbose:
        print(f"  {tag} Step 2/5 — Computing rolling features (5/15/30/60 bars) ...")
    df = compute_rolling_features(df)

    if verbose:
        print(f"  {tag} Step 3/5 — Computing lag features ...")
    df = compute_lag_features(df)

    if verbose:
        print(f"  {tag} Step 4/5 — Adding session/time features ...")
    df = add_session_features(df)

    if verbose:
        print(f"  {tag} Step 5/5 — Building 15M-ahead micro-regime labels ...")
    labels = build_micro_labels(
        df,
        forward_bars     = forward_bars,
        bar_offset       = bar_offset,
        regime_pct_high  = regime_pct_high,
        regime_pct_low   = regime_pct_low,
        rolling_baseline = rolling_baseline,
    )

    # Define feature columns (exclude raw OHLCV and intermediate columns)
    exclude = {"Open", "High", "Low", "Close", "TickVolume",
               "active_ratio", "is_active", "tick_vol", "is_silent"}
    feat_cols = [c for c in df.columns if c not in exclude]

    # Align features + labels
    joined = pd.concat([df, labels], axis=1)
    joined = joined.dropna(subset=["micro_regime"])
    joined = joined.dropna(subset=feat_cols, how="all")

    # Optional systematic subsampling to reduce compute
    if 0 < subsample_frac < 1.0:
        step = max(1, int(round(1.0 / subsample_frac)))
        joined = joined.iloc[::step].copy()
        if verbose:
            print(f"  {tag}   Subsampled 1-in-{step} rows "
                  f"→ {len(joined):,} rows (subsample_frac={subsample_frac})")

    fast_pct = (joined["micro_regime"] == 1).mean()
    slow_pct = (joined["micro_regime"] == 0).mean()

    if verbose:
        print(f"\n  {tag} ── Micro-Regime Dataset Summary ──")
        print(f"  {tag}   Labeled rows  : {len(joined):,}")
        print(f"  {tag}   Fast Tape (1) : {fast_pct:.2%}")
        print(f"  {tag}   Slow Tape (0) : {slow_pct:.2%}")
        print(f"  {tag}   Features      : {len(feat_cols)}")
        print(f"  {tag}   Date range    : {joined.index[0].date()} → "
              f"{joined.index[-1].date()}")

        ar15 = joined.get("active_ratio_15")
        if ar15 is not None:
            ar15 = ar15.dropna()
            print(f"\n  {tag}   active_ratio_15 distribution:")
            print(f"  {tag}     p10={ar15.quantile(0.10):.3f}  "
                  f"p25={ar15.quantile(0.25):.3f}  "
                  f"p50={ar15.quantile(0.50):.3f}  "
                  f"p75={ar15.quantile(0.75):.3f}  "
                  f"p90={ar15.quantile(0.90):.3f}")

    return joined, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    NQ_1M_PATH   = "/Users/macos/Documents/ALGO/03_Data/raw/NAS100/1m_data.csv"
    GOLD_1M_PATH = "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/XAUUSD_M1.csv"

    print("\n" + "═" * 70)
    print("  MICRO-REGIME FEATURE MODULE — SMOKE TEST")
    print("═" * 70)

    for asset, loader, path, frac in [
        ("NAS100", load_nq_1m,   NQ_1M_PATH,   1.0),
        ("GOLD",   load_gold_1m, GOLD_1M_PATH, 0.5),  # subsample gold
    ]:
        print(f"\n{'─'*70}")
        print(f"  {asset}")
        print(f"{'─'*70}")
        t0 = time.time()

        df_1m = loader(path)
        print(f"  Loaded {len(df_1m):,} bars | "
              f"{df_1m.index[0].date()} → {df_1m.index[-1].date()}")

        joined, feat_cols = build_micro_dataset(df_1m, asset_name=asset,
                                                subsample_frac=frac)

        print(f"\n  Elapsed: {time.time()-t0:.1f}s")
        print(f"  Feature columns ({len(feat_cols)}): {feat_cols[:8]} ...")
