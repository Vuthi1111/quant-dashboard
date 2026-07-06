"""
tape_speed_features.py
═══════════════════════════════════════════════════════════════════════════════
Speed of Tape — Feature Engineering Module

Concept
-------
"Speed of tape" = the rate at which the market prints NEW prices.
We only count bars where price actually changed from the previous bar.
The resulting metrics describe how ACTIVE the market's tick flow is.

Data Flow
---------
  1M raw bars (TickVolume + OHLC)
       ↓  per-bar activity flags
       ↓  roll-up to 15M windows
  15M feature matrix  ← model input

Features Built
--------------
Per 1M bar:
  is_active       : 1 if Close changed from prior bar, else 0
  active_tickvol  : TickVolume * is_active (only ticks that moved price)

Aggregated to 15M (15 sub-bars each):
  avg_tickvol     : mean ticks/min over window
  sum_tickvol     : total tick flow in window
  max_tickvol     : peak 1M activity (burst detection)
  active_ratio    : fraction of 1M bars where price changed  ← core signal
  active_tickvol  : sum of ticks only on active (moving) bars
  tape_cv         : coefficient of variation of 1M tickvol (bursty vs steady)
  tape_accel      : change in avg_tickvol vs prior 15M bar (acceleration)
  range_per_tick  : (H-L) / sum_tickvol — efficiency of flow (Amihud proxy)
  tick_density    : sum_tickvol / (H-L+ε) — ticks needed to move 1 pip
  silent_ratio    : fraction of 1M bars with TickVolume == 0 (dead tape)

Rolling context (computed on the 15M series):
  tv_zscore_20    : z-score of sum_tickvol vs 20-bar rolling baseline
  tv_zscore_96    : z-score vs 96-bar rolling baseline (1 full session)
  active_ratio_ma : 20-bar MA of active_ratio (trend in tape activity)
  tape_accel_ma   : 5-bar MA of tape_accel (smoothed acceleration signal)

Time / session features:
  hour_sin/cos    : cyclical hour encoding
  dow_sin/cos     : cyclical day-of-week
  is_us_open      : 1 if 14:30–21:00 UTC (NYSE session)
  is_london_open  : 1 if 08:00–16:30 UTC
  is_overlap      : 1 if London/NY overlap (highest liquidity window)
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import sys, os

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_nq_1m(path: str) -> pd.DataFrame:
    """Load NAS100 1M MT5 tab-delimited data."""
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip().str.lower()
    df["datetime"] = pd.to_datetime(df["datetime"],
                                    format="%Y.%m.%d %H:%M:%S", errors="coerce")
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    df = df[df["close"] > 0].copy()

    # Normalise column names
    df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "tickvolume": "TickVolume"}, inplace=True)
    if "TickVolume" not in df.columns and "volume" in df.columns:
        df["TickVolume"] = df["volume"]
    df["TickVolume"] = pd.to_numeric(df["TickVolume"], errors="coerce").fillna(0)
    return df


def load_gold_1m(path: str) -> pd.DataFrame:
    """
    Load GOLD XAUUSD_M1.csv which has header:
    ,time,open,high,low,close,tick_volume,spread,real_volume
    """
    df = pd.read_csv(path, index_col=0)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    df = df[df["close"] > 0].copy()
    df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "tick_volume": "TickVolume"}, inplace=True)
    df["TickVolume"] = pd.to_numeric(df["TickVolume"], errors="coerce").fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. PER-BAR ACTIVITY FLAGS (1M resolution)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bar_activity(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Flag each 1M bar:
      is_active     = 1 if Close changed vs previous bar
      active_tickvol = TickVolume if is_active else 0
    """
    df = df_1m.copy()
    df["is_active"] = (df["Close"] != df["Close"].shift(1)).astype(int)
    # First bar has no prior — set to 0
    df.iloc[0, df.columns.get_loc("is_active")] = 0
    df["active_tickvol"] = df["TickVolume"] * df["is_active"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. RESAMPLE TO 15M WITH TAPE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_to_15m(df_1m_active: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up 1M flagged bars to 15M candles with tape speed features.
    Each 15M bar captures ~15 sub-bars.
    """
    agg = df_1m_active.resample("15min").agg(
        Open            = ("Open",           "first"),
        High            = ("High",           "max"),
        Low             = ("Low",            "min"),
        Close           = ("Close",          "last"),
        sum_tickvol     = ("TickVolume",     "sum"),
        avg_tickvol     = ("TickVolume",     "mean"),
        max_tickvol     = ("TickVolume",     "max"),
        std_tickvol     = ("TickVolume",     "std"),
        active_count    = ("is_active",      "sum"),   # how many 1M bars moved price
        bar_count       = ("is_active",      "count"), # total 1M bars in window
        active_tickvol  = ("active_tickvol", "sum"),
        silent_count    = ("TickVolume",     lambda x: (x == 0).sum()),
    ).dropna(subset=["Open", "Close"])

    df = agg.copy()

    # active_ratio: fraction of 1M bars where price changed  ← CORE TAPE SPEED SIGNAL
    df["active_ratio"]  = df["active_count"] / df["bar_count"].replace(0, np.nan)

    # silent_ratio: fraction of dead (zero-tick) bars
    df["silent_ratio"]  = df["silent_count"] / df["bar_count"].replace(0, np.nan)

    # tape_cv: coefficient of variation — bursty vs steady flow
    df["tape_cv"]       = df["std_tickvol"] / (df["avg_tickvol"].replace(0, np.nan))

    # tape_accel: change in avg_tickvol vs prior 15M bar
    df["tape_accel"]    = df["avg_tickvol"].diff(1)

    # range in price terms
    price_range         = (df["High"] - df["Low"]).replace(0, np.nan)

    # range_per_tick: price moved per unit of flow (Amihud-style illiquidity)
    df["range_per_tick"] = price_range / df["sum_tickvol"].replace(0, np.nan)

    # tick_density: ticks per pip of range (inverse — how much flow per price unit)
    df["tick_density"]  = df["sum_tickvol"] / price_range

    # active_tick_ratio: of all ticks, what fraction occurred on price-changing bars
    df["active_tick_ratio"] = df["active_tickvol"] / df["sum_tickvol"].replace(0, np.nan)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. ROLLING CONTEXT FEATURES (on 15M series)
# ─────────────────────────────────────────────────────────────────────────────

def add_rolling_context(df_15m: pd.DataFrame,
                        baseline_short: int = 20,   # 20 bars = 5 hours
                        baseline_long:  int = 96    # 96 bars = 1 full session
                        ) -> pd.DataFrame:
    """
    Z-score tape speed relative to rolling baselines.
    All features shifted by 1 to prevent look-ahead leakage.
    """
    df = df_15m.copy()

    for col, window, name in [
        ("sum_tickvol", baseline_short, "tv_zscore_20"),
        ("sum_tickvol", baseline_long,  "tv_zscore_96"),
    ]:
        roll_mean = df[col].rolling(window).mean()
        roll_std  = df[col].rolling(window).std().replace(0, np.nan)
        df[name]  = (df[col] - roll_mean) / roll_std

    # Rolling MA of active_ratio (trend in tape activity)
    df["active_ratio_ma5"]  = df["active_ratio"].rolling(5).mean()
    df["active_ratio_ma20"] = df["active_ratio"].rolling(baseline_short).mean()

    # Smoothed tape acceleration
    df["tape_accel_ma5"] = df["tape_accel"].rolling(5).mean()

    # Volume momentum: ratio of recent 4-bar sum vs baseline
    df["tv_momentum"] = (
        df["sum_tickvol"].rolling(4).sum() /
        (df["sum_tickvol"].rolling(baseline_long).mean() * 4).replace(0, np.nan)
    )

    # Log transforms for skewed distributions
    df["log_sum_tickvol"]    = np.log1p(df["sum_tickvol"])
    df["log_active_tickvol"] = np.log1p(df["active_tickvol"])
    df["log_max_tickvol"]    = np.log1p(df["max_tickvol"])

    # Lag features (past 1, 2, 4 bars)
    for lag in [1, 2, 4]:
        df[f"active_ratio_lag{lag}"] = df["active_ratio"].shift(lag)
        df[f"tv_zscore_20_lag{lag}"] = df["tv_zscore_20"].shift(lag)

    # Shift ALL features by 1 bar — no look-ahead
    feature_cols = [c for c in df.columns
                    if c not in ["Open", "High", "Low", "Close"]]
    for col in feature_cols:
        df[col] = df[col].shift(1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. SESSION / TIME FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market session flags and cyclical time encodings.
    Sessions in UTC:
      London : 08:00 - 16:30
      New York: 14:30 - 21:00
      Overlap : 14:30 - 16:30
    """
    idx = df.index
    h   = idx.hour + idx.minute / 60.0

    df["hour_sin"]  = np.sin(2 * np.pi * h / 24.0)
    df["hour_cos"]  = np.cos(2 * np.pi * h / 24.0)
    df["dow_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 5.0)
    df["dow_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 5.0)

    df["is_london"]  = ((idx.hour >= 8)  & (idx.hour < 17)).astype(int)
    df["is_ny"]      = ((idx.hour >= 14) & (idx.hour < 21)).astype(int)
    df["is_overlap"] = ((idx.hour >= 14) & (idx.hour < 17)).astype(int)
    df["is_asian"]   = ((idx.hour >= 0)  & (idx.hour < 8)).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. LABEL CONSTRUCTION — FUTURE TAPE REGIME
# ─────────────────────────────────────────────────────────────────────────────

def build_tape_regime_labels(df_15m: pd.DataFrame,
                              forward_bars:      int   = 16,   # 4 hours ahead
                              bar_offset:        int   = 1,    # 1-bar buffer
                              regime_pct_high:   float = 0.70, # top 30% = Fast Tape
                              regime_pct_low:    float = 0.30, # bot 30% = Slow Tape
                              rolling_baseline:  int   = 672   # 1 week of 15M bars
                              ) -> pd.DataFrame:
    """
    Predict whether the NEXT forward_bars of 15M bars will be
    Fast Tape (label=1) or Slow Tape (label=0).

    The tape speed signal = active_ratio (fraction of 1M sub-bars
    where price changed). This is the purest measure of price update rate.

    Structure mirrors build_vol_regime_labels() for consistency.
      bar_offset=1  → adds 1-bar buffer before forecast window starts
                      prevents same-bar boundary overlap artifact
    """
    # Use active_ratio as the tape speed signal
    # (falls back to sum_tickvol if active_ratio not present)
    if "active_ratio" in df_15m.columns:
        tape_signal = df_15m["active_ratio"]
    else:
        tape_signal = df_15m["sum_tickvol"]

    # Forward window: starts at t + bar_offset + 1, spans forward_bars bars
    fwd_tape = (
        tape_signal
        .shift(-(bar_offset + 1))
        .rolling(forward_bars).mean()
        .shift(-(forward_bars - 1))
    )

    # Rolling percentile thresholds — computed on PAST active_ratio data only
    # (uses rolling backward signal, NOT fwd, to avoid self-referential label boundaries)
    roll_signal = tape_signal.shift(1)
    roll_high = roll_signal.rolling(rolling_baseline).quantile(regime_pct_high)
    roll_low  = roll_signal.rolling(rolling_baseline).quantile(regime_pct_low)

    labels = pd.DataFrame(index=df_15m.index)
    labels["fwd_tape_speed"] = fwd_tape
    labels["tape_regime"]    = np.nan

    labels.loc[fwd_tape >= roll_high, "tape_regime"] = 1  # Fast Tape
    labels.loc[fwd_tape <= roll_low,  "tape_regime"] = 0  # Slow Tape
    # Middle 40% stays NaN — discarded during training (no edge zone)

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 7. MASTER BUILDER — full feature matrix + labels for one asset
# ─────────────────────────────────────────────────────────────────────────────

def build_tape_dataset(df_1m: pd.DataFrame,
                       asset_name: str = "",
                       forward_bars: int = 16,
                       bar_offset:   int = 1,
                       regime_pct_high: float = 0.70,
                       regime_pct_low:  float = 0.30,
                       rolling_baseline: int  = 672,
                       verbose: bool = True) -> tuple[pd.DataFrame, list]:
    """
    Full pipeline: 1M raw → 15M feature matrix + tape regime labels.

    Returns
    -------
    joined      : aligned DataFrame with features + label column 'tape_regime'
    feature_cols: list of feature column names
    """
    tag = f"[{asset_name}]" if asset_name else ""

    if verbose:
        print(f"  {tag} Step 1/5 — Computing per-bar activity flags ...")
    df_active = compute_bar_activity(df_1m)

    active_pct = df_active["is_active"].mean()
    if verbose:
        print(f"  {tag}   1M bars total    : {len(df_active):,}")
        print(f"  {tag}   Price-change bars : {df_active['is_active'].sum():,} "
              f"({active_pct:.1%} of all 1M bars)")

    if verbose:
        print(f"  {tag} Step 2/5 — Aggregating to 15M ...")
    df_15m = aggregate_to_15m(df_active)

    if verbose:
        print(f"  {tag}   15M bars         : {len(df_15m):,}")
        print(f"  {tag}   Date range       : {df_15m.index[0].date()} → "
              f"{df_15m.index[-1].date()}")

    if verbose:
        print(f"  {tag} Step 3/5 — Adding rolling context features ...")
    df_15m = add_rolling_context(df_15m)

    if verbose:
        print(f"  {tag} Step 4/5 — Adding session/time features ...")
    df_15m = add_session_features(df_15m)

    if verbose:
        print(f"  {tag} Step 5/5 — Building tape regime labels ...")
    labels = build_tape_regime_labels(
        df_15m,
        forward_bars     = forward_bars,
        bar_offset       = bar_offset,
        regime_pct_high  = regime_pct_high,
        regime_pct_low   = regime_pct_low,
        rolling_baseline = rolling_baseline,
    )

    # Align features + labels, drop NaN labels
    feat_cols = [c for c in df_15m.columns
                 if c not in ["Open", "High", "Low", "Close"]]

    joined = pd.concat([df_15m, labels], axis=1).dropna(subset=["tape_regime"])

    # Also drop rows where ALL features are NaN (early warm-up bars)
    joined = joined.dropna(subset=feat_cols, how="all")
    joined = joined.dropna(subset=["tape_regime"])

    fast_pct = (joined["tape_regime"] == 1).mean()
    slow_pct = (joined["tape_regime"] == 0).mean()

    if verbose:
        print(f"\n  {tag} ── Dataset Summary ──")
        print(f"  {tag}   Labeled samples : {len(joined):,}")
        print(f"  {tag}   Fast Tape (1)   : {fast_pct:.2%}")
        print(f"  {tag}   Slow Tape (0)   : {slow_pct:.2%}")
        print(f"  {tag}   Features        : {len(feat_cols)}")
        print(f"  {tag}   Date range      : {joined.index[0].date()} → "
              f"{joined.index[-1].date()}")

        # Show tape speed distribution
        ar = joined["active_ratio"] if "active_ratio" in joined.columns else None
        if ar is not None:
            print(f"\n  {tag}   Active ratio (price-change rate) distribution:")
            print(f"  {tag}     p10={ar.quantile(0.10):.3f}  "
                  f"p25={ar.quantile(0.25):.3f}  "
                  f"p50={ar.quantile(0.50):.3f}  "
                  f"p75={ar.quantile(0.75):.3f}  "
                  f"p90={ar.quantile(0.90):.3f}")

    return joined, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    NQ_1M_PATH   = "/Users/macos/Documents/ALGO/03_Data/raw/NAS100/1m_data.csv"
    GOLD_1M_PATH = "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/XAUUSD_M1.csv"

    print("\n" + "═" * 70)
    print("  TAPE SPEED FEATURE MODULE — SMOKE TEST")
    print("═" * 70)

    for asset, loader, path in [
        ("NAS100", load_nq_1m,   NQ_1M_PATH),
        ("GOLD",   load_gold_1m, GOLD_1M_PATH),
    ]:
        print(f"\n{'─'*70}")
        print(f"  {asset}")
        print(f"{'─'*70}")
        t0 = time.time()

        print(f"  Loading 1M data from {path} ...")
        df_1m = loader(path)
        print(f"  Loaded {len(df_1m):,} bars | "
              f"{df_1m.index[0].date()} → {df_1m.index[-1].date()}")
        print(f"  TickVolume stats: min={df_1m['TickVolume'].min():.0f}  "
              f"median={df_1m['TickVolume'].median():.0f}  "
              f"max={df_1m['TickVolume'].max():.0f}")

        joined, feat_cols = build_tape_dataset(df_1m, asset_name=asset)

        print(f"\n  Elapsed: {time.time()-t0:.1f}s")
        print(f"  Feature columns: {feat_cols}")
