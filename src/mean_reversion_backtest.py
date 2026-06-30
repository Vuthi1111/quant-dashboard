"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MEAN REVERSION STRATEGY — REGIME-FILTERED                                  ║
║   Three variants tested side-by-side:                                        ║
║     1. Session VWAP Bands   (anchor: 09:30 NY)                               ║
║     2. Overnight VWAP Bands (anchor: 16:00 NY prior session)                 ║
║     3. Bollinger Bands      (20-period rolling)                               ║
║                                                                              ║
║   Regime Filter Logic                                                        ║
║   ─────────────────────────────────────────────────────────────────────────  ║
║   LOW VOL  (prob < 0.30) → Take the fade. Market is choppy/range-bound.     ║
║   HIGH VOL (prob > 0.70) → SKIP. Market is trending. Don't fight it.        ║
║   UNCERTAIN              → SKIP. No edge.                                   ║
║                                                                              ║
║   Run : /opt/anaconda3/bin/python mean_reversion_backtest.py                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import warnings; warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, "/Users/macos/Documents/ALGO/04_Models/walk_forward_ml")
sys.path.insert(0, "/Users/macos/Documents/ALGO/projects/volatility_regime_model/src")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from feature_engineering import load_mt5_csv, build_features, build_vol_regime_labels

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT     = Path("/Users/macos/Documents/ALGO")
DATA_RAW = ROOT / "03_Data" / "raw"
OUT_DIR  = ROOT / "projects" / "volatility_regime_model" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAS_M15_PATH = DATA_RAW / "NAS100" / "15m_data.csv"
NAS_1H_PATH  = DATA_RAW / "NAS100" / "1h_data.csv"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ATR_PERIOD   = 14
BAND_WIDTH   = 2.0      # Number of standard deviations for bands
ATR_STOP_MULT   = 0.75   # Stop = 0.75× ATR beyond the band touch (tighter stop)
RR_TARGET       = 2.0    # Reward:Risk = 2:1 (fixed, not soft VWAP mid)
MIN_DIST_ATR    = 1.0    # Only enter if price is ≥1× ATR from the mid (ensures meaningful R:R)
RISK_PCT     = 0.005    # 0.5% account risk per trade
INIT_CAPITAL = 100_000
PROB_LOW     = 0.30     # Below this → LOW VOL → take mean reversion trades
PROB_HIGH    = 0.70     # Above this → HIGH VOL → skip

# NY session hours (in ET, after -7h timezone adjustment from MT5 broker time)
SESSION_OPEN_H, SESSION_OPEN_M = 9, 30
SESSION_CLOSE_H = 15
SESSION_CLOSE_M = 45
OVERNIGHT_START_H = 16   # 4 PM ET — start of overnight session

BB_PERIOD = 20           # Bollinger Band lookback

# ─── COLOURS ─────────────────────────────────────────────────────────────────
BG, BG2     = "#0d1117", "#161b22"
GRID_C      = "#21262d"
ACCENT      = "#58a6ff"
GREEN, RED  = "#3fb950", "#f85149"
TEXT, TEXT2 = "#c9d1d9", "#8b949e"
PURPLE      = "#bc8cff"
ORANGE      = "#f0883e"
YELLOW      = "#e3b341"


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_nas100_m15() -> pd.DataFrame:
    df = pd.read_csv(
        NAS_M15_PATH, sep="\t", header=0,
        names=["datetime","open","high","low","close","vol_raw","tick_volume"],
    )
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y.%m.%d %H:%M:%S")
    df = df.sort_values("datetime").reset_index(drop=True)
    df.set_index("datetime", inplace=True)
    df.index = df.index - pd.Timedelta(hours=7)
    # Use tick_volume as the primary volume proxy; drop the raw volume column
    df["volume"] = df["tick_volume"]
    df.drop(columns=["vol_raw", "tick_volume"], inplace=True)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2.  REGIME SIGNAL
# ═════════════════════════════════════════════════════════════════════════════

def build_regime_signal(train_cutoff_pct: float = 0.75) -> pd.Series:
    print("[Regime] Loading 1H data and building features...")
    df_1h    = load_mt5_csv(str(NAS_1H_PATH))
    feat_df  = build_features(df_1h)
    label_df = build_vol_regime_labels(
        df_1h, forward_bars=4, bar_offset=4,
        regime_pct_high=0.70, regime_pct_low=0.30, rolling_baseline=480,
    )
    joined = pd.concat([feat_df, label_df], axis=1).dropna(subset=["vol_regime"])
    X = joined[feat_df.columns].values.astype(np.float32)
    y = joined["vol_regime"].values.astype(np.int32)
    split   = int(len(X) * train_cutoff_pct)
    sc      = StandardScaler().fit(X[:split])
    model   = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    model.fit(sc.transform(X[:split]), y[:split])
    probs   = model.predict_proba(sc.transform(X))[:, 1]
    signal  = pd.Series(probs, index=joined.index, name="regime_prob")
    print(f"[Regime] Trained   : {joined.index[0].date()} → {joined.index[split-1].date()}")
    print(f"[Regime] OOS signal: {joined.index[split].date()} → {joined.index[-1].date()}")
    return signal


def get_regime(dt: pd.Timestamp, signal: pd.Series) -> str:
    prior = signal.index[signal.index <= dt]
    if len(prior) == 0:
        return "uncertain"
    p = signal.loc[prior[-1]]
    if p < PROB_LOW:   return "low"
    if p > PROB_HIGH:  return "high"
    return "uncertain"


# ═════════════════════════════════════════════════════════════════════════════
# 3.  INDICATOR CALCULATIONS
# ═════════════════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame) -> pd.Series:
    pc  = df["close"].shift(1)
    tr  = pd.concat([df["high"]-df["low"],
                     (df["high"]-pc).abs(),
                     (df["low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean().shift(1)


def build_session_vwap_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anchored VWAP ± 2σ bands, resetting at each NY session open (09:30).
    Uses true VWAP variance: Var = E[P²*V]/E[V] - VWAP²
    """
    typical = ((df["high"] + df["low"] + df["close"]) / 3).values
    vol     = df["volume"].replace(0, 1).values
    hours   = df.index.hour
    minutes = df.index.minute

    vwap    = np.full(len(df), np.nan)
    upper   = np.full(len(df), np.nan)
    lower   = np.full(len(df), np.nan)

    cum_pv = cum_p2v = cum_v = 0.0
    in_session = False

    for i in range(len(df)):
        h, m = int(hours[i]), int(minutes[i])

        if h == SESSION_OPEN_H and m == SESSION_OPEN_M:
            cum_pv = cum_p2v = cum_v = 0.0
            in_session = True

        if not in_session:
            continue

        p = float(typical[i])
        v = float(vol[i])
        cum_pv  += p * v
        cum_p2v += p * p * v
        cum_v   += v

        if cum_v > 0:
            w = cum_pv / cum_v
            variance = max(cum_p2v / cum_v - w * w, 0.0)
            sd = np.sqrt(variance)
            vwap[i]  = w
            upper[i] = w + BAND_WIDTH * sd
            lower[i] = w - BAND_WIDTH * sd

    out = df.copy()
    out["vwap"]  = vwap
    out["upper"] = upper
    out["lower"] = lower
    return out


def build_overnight_vwap_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overnight VWAP anchored from 16:00 ET (prior session close).
    The last overnight VWAP value is held fixed during the day session.
    """
    typical  = ((df["high"] + df["low"] + df["close"]) / 3).values
    vol      = df["volume"].replace(0, 1).values
    hours    = df.index.hour
    minutes  = df.index.minute

    ov_vwap  = np.full(len(df), np.nan)
    ov_upper = np.full(len(df), np.nan)
    ov_lower = np.full(len(df), np.nan)

    cum_pv = cum_p2v = cum_v = 0.0
    in_overnight = False
    last_w = last_sd = np.nan

    for i in range(len(df)):
        h, m = int(hours[i]), int(minutes[i])

        if h == OVERNIGHT_START_H and m == 0:
            cum_pv = cum_p2v = cum_v = 0.0
            in_overnight = True

        if h == SESSION_OPEN_H and m == SESSION_OPEN_M:
            in_overnight = False

        if in_overnight:
            p = float(typical[i])
            v = float(vol[i])
            cum_pv  += p * v
            cum_p2v += p * p * v
            cum_v   += v
            if cum_v > 0:
                last_w   = cum_pv / cum_v
                variance = max(cum_p2v / cum_v - last_w * last_w, 0.0)
                last_sd  = np.sqrt(variance)
        elif not np.isnan(last_w):
            ov_vwap[i]  = last_w
            ov_upper[i] = last_w + BAND_WIDTH * max(float(last_sd), 1.0)
            ov_lower[i] = last_w - BAND_WIDTH * max(float(last_sd), 1.0)

    out = df.copy()
    out["ov_vwap"]  = ov_vwap
    out["ov_upper"] = ov_upper
    out["ov_lower"] = ov_lower
    return out


def build_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling 20-bar Bollinger Bands. All values shifted by 1 (no lookahead)."""
    mid   = df["close"].rolling(BB_PERIOD).mean().shift(1)
    std   = df["close"].rolling(BB_PERIOD).std().shift(1)
    out   = df.copy()
    out["bb_mid"]   = mid
    out["bb_upper"] = mid + BAND_WIDTH * std
    out["bb_lower"] = mid - BAND_WIDTH * std
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 4.  TRADE DATACLASS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    direction:     str
    entry_time:    pd.Timestamp
    entry_price:   float
    stop_price:    float
    target_price:  float
    contracts:     float
    risk_usd:      float
    regime:        str
    variant:       str
    exit_time:     Optional[pd.Timestamp] = None
    exit_price:    Optional[float]        = None
    exit_reason:   str                    = ""
    pnl_usd:       float                  = 0.0
    pnl_r:         float                  = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 5.  GENERIC BACKTEST ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest(
    df:           pd.DataFrame,
    regime_sig:   pd.Series,
    upper_col:    str,
    lower_col:    str,
    mid_col:      str,
    variant:      str,
    use_regime:   bool = True,
    spread:       float = 0.8,
) -> Dict[str, Any]:
    """
    Generic mean reversion backtest engine.

    Entry rules:
      - Bar LOW <= lower band → LONG (fade the downside extension)
      - Bar HIGH >= upper band → SHORT (fade the upside extension)

    Exit rules:
      - Target: mid band (VWAP or BB mid)
      - Stop  : ATR × ATR_STOP_MULT beyond entry
      - Time  : 15:45 ET

    use_regime=True  → only trade when LOW VOL predicted
    use_regime=False → trade all signals (baseline)
    """
    df = df.copy().reset_index()
    atr_series = compute_atr(df.set_index("datetime")).values

    equity       = INIT_CAPITAL
    trades: List[Trade] = []
    open_trade: Optional[Trade] = None
    equity_curve = [equity]
    equity_dates = [df.iloc[0]["datetime"]]
    trades_today = 0
    current_day  = df.iloc[0]["datetime"].date()

    def close_trade(t, exit_px, reason, i):
        nonlocal equity
        if t.direction == "LONG":
            pnl = (exit_px - t.entry_price) * t.contracts
        else:
            pnl = (t.entry_price - exit_px) * t.contracts
        t.exit_time, t.exit_price, t.exit_reason = df.iloc[i]["datetime"], exit_px, reason
        t.pnl_usd = pnl
        t.pnl_r   = pnl / t.risk_usd if t.risk_usd else 0
        equity += pnl
        trades.append(t)

    for i in range(BB_PERIOD + 1, len(df)):
        row = df.iloc[i]
        dt  = row["datetime"]
        atr = atr_series[i]

        if dt.date() != current_day:
            trades_today = 0
            current_day  = dt.date()

        # Validate indicator values
        upper = row.get(upper_col, np.nan)
        lower = row.get(lower_col, np.nan)
        mid   = row.get(mid_col,   np.nan)
        if any(np.isnan(v) for v in [upper, lower, mid, atr]) or atr <= 0:
            equity_curve.append(equity); equity_dates.append(dt)
            continue

        # Only trade during NY session
        in_session = (
            (dt.hour > SESSION_OPEN_H or (dt.hour == SESSION_OPEN_H and dt.minute >= SESSION_OPEN_M))
            and
            (dt.hour < SESSION_CLOSE_H or (dt.hour == SESSION_CLOSE_H and dt.minute < SESSION_CLOSE_M))
        )

        # ── Manage open trade ─────────────────────────────────────────────────
        if open_trade is not None:
            t = open_trade
            is_time_stop = (dt.hour == SESSION_CLOSE_H and dt.minute >= SESSION_CLOSE_M) or dt.hour >= 16

            if t.direction == "LONG":
                if row["low"] <= t.stop_price:
                    close_trade(t, t.stop_price, "STOP", i); open_trade = None
                elif row["high"] >= t.target_price:
                    close_trade(t, t.target_price, "TAKE_PROFIT", i); open_trade = None
                elif is_time_stop:
                    close_trade(t, row["close"], "TIME_STOP", i); open_trade = None
            else:
                ask_h = row["high"]  + spread
                ask_l = row["low"]   + spread
                ask_c = row["close"] + spread
                if ask_h >= t.stop_price:
                    close_trade(t, t.stop_price, "STOP", i); open_trade = None
                elif ask_l <= t.target_price:
                    close_trade(t, t.target_price, "TAKE_PROFIT", i); open_trade = None
                elif is_time_stop:
                    close_trade(t, ask_c, "TIME_STOP", i); open_trade = None

        # ── Entry ─────────────────────────────────────────────────────────────
        if open_trade is None and trades_today == 0 and in_session:

            # Regime gate
            if use_regime:
                regime = get_regime(dt, regime_sig)
                if regime != "low":
                    equity_curve.append(equity); equity_dates.append(dt)
                    continue
            else:
                regime = "baseline"

            stop_dist  = ATR_STOP_MULT * atr
            target_dist = stop_dist * RR_TARGET
            risk_usd   = equity * RISK_PCT

            direction  = None
            entry_px   = stop_px = target_px = None

            # Only enter if mid is at least MIN_DIST_ATR away — ensures the
            # target (2R) is a realistic move, not just a few points.
            dist_to_mid = abs(row["close"] - mid)

            # LONG: price touched bottom band AND mid is meaningfully above
            if row["low"] <= lower and dist_to_mid >= MIN_DIST_ATR * atr:
                direction = "LONG"
                entry_px  = row["close"]
                stop_px   = entry_px - stop_dist
                target_px = entry_px + target_dist   # fixed 2R target

            # SHORT: price touched top band AND mid is meaningfully below
            elif row["high"] >= upper and dist_to_mid >= MIN_DIST_ATR * atr:
                direction = "SHORT"
                entry_px  = row["close"] + spread
                stop_px   = entry_px + stop_dist
                target_px = entry_px - target_dist   # fixed 2R target

            if direction is not None and (target_px - entry_px) * (1 if direction == "LONG" else -1) > 0:
                contracts  = risk_usd / stop_dist
                open_trade = Trade(
                    direction=direction, entry_time=dt, entry_price=entry_px,
                    stop_price=stop_px, target_price=target_px,
                    contracts=contracts, risk_usd=risk_usd,
                    regime=regime, variant=variant,
                )
                trades_today = 1

        equity_curve.append(equity)
        equity_dates.append(dt)

    if open_trade is not None:
        t = open_trade
        last = df.iloc[-1]
        ep   = last["close"] if t.direction == "LONG" else last["close"] + spread
        close_trade(t, ep, "EOD_FORCE", len(df)-1)

    return {
        "variant": variant,
        "use_regime": use_regime,
        "trades":  trades,
        "equity":  pd.Series(equity_curve, index=pd.to_datetime(equity_dates)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 6.  STATS
# ═════════════════════════════════════════════════════════════════════════════

def compute_stats(result: Dict[str, Any]) -> Dict[str, float]:
    trades = result["trades"]
    equity = result["equity"]
    if not trades:
        return {}

    pnls   = np.array([t.pnl_usd for t in trades])
    r_mult = np.array([t.pnl_r   for t in trades])
    wins   = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    years   = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-6)
    cagr    = ((equity.iloc[-1] / INIT_CAPITAL) ** (1/years) - 1) * 100
    roll_max = equity.cummax()
    max_dd   = ((equity - roll_max) / roll_max * 100).min()
    ret_d   = equity.resample("D").last().pct_change().dropna()
    sharpe  = ret_d.mean() / ret_d.std() * np.sqrt(252) if ret_d.std() > 0 else 0
    wr      = len(wins) / len(pnls) * 100
    pf      = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else np.inf

    return dict(
        n_trades=len(pnls), win_rate=wr, profit_factor=pf,
        avg_r=r_mult.mean(), cagr_pct=cagr, max_dd_pct=max_dd,
        sharpe=sharpe, final_equity=equity.iloc[-1],
    )


# ═════════════════════════════════════════════════════════════════════════════
# 7.  PLOT
# ═════════════════════════════════════════════════════════════════════════════

VARIANT_COLORS = {
    "Session VWAP":    ACCENT,
    "Overnight VWAP":  PURPLE,
    "Bollinger Bands": ORANGE,
}

def plot_all(results_base, results_regime, stats_base, stats_regime, regime_sig):
    variants = ["Session VWAP", "Overnight VWAP", "Bollinger Bands"]

    fig = plt.figure(figsize=(24, 20), facecolor=BG)
    fig.suptitle(
        "Mean Reversion Strategy — Baseline vs ML Regime-Filtered (LOW VOL only)\n"
        f"All three variants tested on NAS100 M15 | Regime: prob < {PROB_LOW}",
        fontsize=15, color=TEXT, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.42, wspace=0.28)

    def style(ax, title=""):
        ax.set_facecolor(BG2)
        ax.tick_params(colors=TEXT2, labelsize=9)
        for s in ax.spines.values(): s.set_color(GRID_C)
        ax.grid(True, color=GRID_C, linewidth=0.5, alpha=0.6)
        if title: ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=6)

    # ── Row 0: Equity curves, one per variant ──────────────────────────────
    for col, v in enumerate(variants):
        ax = fig.add_subplot(gs[0, col])
        eq_b = results_base[v]["equity"]
        eq_r = results_regime[v]["equity"]
        ax.plot(eq_b.index, eq_b.values, color=TEXT2, lw=1.2,
                label=f"Baseline  (Sharpe {stats_base[v].get('sharpe',0):.2f})")
        ax.plot(eq_r.index, eq_r.values, color=VARIANT_COLORS[v], lw=1.8,
                label=f"Regime   (Sharpe {stats_regime[v].get('sharpe',0):.2f})")
        ax.axhline(INIT_CAPITAL, color=TEXT2, lw=0.7, ls="--")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT, fontsize=8)
        style(ax, v)

    # ── Row 1: Drawdowns ────────────────────────────────────────────────────
    for col, v in enumerate(variants):
        ax = fig.add_subplot(gs[1, col])
        eq_b = results_base[v]["equity"]
        eq_r = results_regime[v]["equity"]
        dd_b = (eq_b - eq_b.cummax()) / eq_b.cummax() * 100
        dd_r = (eq_r - eq_r.cummax()) / eq_r.cummax() * 100
        ax.plot(dd_b.index, dd_b.values, color=TEXT2, lw=1.0,
                label=f"Baseline  MaxDD {stats_base[v].get('max_dd_pct',0):.1f}%")
        ax.plot(dd_r.index, dd_r.values, color=RED, lw=1.3,
                label=f"Regime   MaxDD {stats_regime[v].get('max_dd_pct',0):.1f}%")
        ax.fill_between(dd_r.index, dd_r.values, 0, alpha=0.2, color=RED)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.legend(facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT, fontsize=8)
        style(ax, f"{v} — Drawdown")

    # ── Row 2: Stats tables ─────────────────────────────────────────────────
    keys  = ["n_trades","win_rate","profit_factor","avg_r","cagr_pct","max_dd_pct","sharpe","final_equity"]
    names = ["Trades","Win Rate","Profit Factor","Avg R","CAGR","Max DD","Sharpe","Final Equity"]

    def fmt(val, k):
        if "equity" in k: return f"${val:,.0f}"
        if "rate" in k or "pct" in k: return f"{val:.1f}%"
        return f"{val:.2f}"

    for col, v in enumerate(variants):
        ax = fig.add_subplot(gs[2, col])
        ax.axis("off")
        rows = []
        for k, n in zip(keys, names):
            b = stats_base[v].get(k, 0)
            r = stats_regime[v].get(k, 0)
            d = r - b
            s = "+" if d > 0 else ""
            rows.append([n, fmt(b, k), fmt(r, k), f"{s}{d:.2f}"])
        t = ax.table(cellText=rows, colLabels=["Metric","Baseline","Regime","Δ"],
                     loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.9)
        for (ri, ci), cell in t.get_celld().items():
            cell.set_edgecolor(GRID_C); cell.set_facecolor(BG)
            cell.set_text_props(color=TEXT)
            if ri == 0:
                cell.set_facecolor(BG2)
                cell.set_text_props(fontweight="bold", color=VARIANT_COLORS[v])
        style(ax, f"{v} — Stats")

    # ── Row 3: Regime signal + winner summary ───────────────────────────────
    ax_sig = fig.add_subplot(gs[3, :2])
    cutoff = regime_sig.index[-1] - pd.DateOffset(months=18)
    rs = regime_sig[regime_sig.index >= cutoff]
    ax_sig.fill_between(rs.index, rs.values, PROB_HIGH,
                        where=rs.values > PROB_HIGH, color=RED,   alpha=0.55, interpolate=True, label="HIGH VOL → Skip")
    ax_sig.fill_between(rs.index, rs.values, PROB_LOW,
                        where=rs.values < PROB_LOW,  color=GREEN, alpha=0.55, interpolate=True, label="LOW VOL → Trade")
    ax_sig.plot(rs.index, rs.values, color="#e8e8e8", lw=0.5)
    ax_sig.axhline(PROB_HIGH, color=RED,   lw=1, ls="--", alpha=0.8)
    ax_sig.axhline(PROB_LOW,  color=GREEN, lw=1, ls="--", alpha=0.8)
    ax_sig.set_ylim(0, 1); ax_sig.set_ylabel("P(High Vol)", color=TEXT2)
    ax_sig.legend(facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT, fontsize=9)
    style(ax_sig, "ML Regime Signal (last 18 months) — GREEN = Trade | RED = Avoid")

    # ── Best strategy highlight ─────────────────────────────────────────────
    ax_best = fig.add_subplot(gs[3, 2])
    ax_best.axis("off")
    best_v = max(variants, key=lambda v: stats_regime[v].get("sharpe", -99))
    b  = stats_base[best_v]
    r  = stats_regime[best_v]
    lines = [
        ("🏆 BEST REGIME STRATEGY", "", ""),
        ("", "", ""),
        (f"{best_v}", "", ""),
        ("", "Baseline", "Regime"),
        ("Sharpe", f"{b.get('sharpe',0):.2f}", f"→  {r.get('sharpe',0):.2f}"),
        ("CAGR", f"{b.get('cagr_pct',0):.1f}%", f"→  {r.get('cagr_pct',0):.1f}%"),
        ("Max DD", f"{b.get('max_dd_pct',0):.1f}%", f"→  {r.get('max_dd_pct',0):.1f}%"),
        ("Win Rate", f"{b.get('win_rate',0):.1f}%", f"→  {r.get('win_rate',0):.1f}%"),
        ("Trades", f"{int(b.get('n_trades',0))}", f"→  {int(r.get('n_trades',0))}"),
    ]
    y = 0.95
    for row in lines:
        if row[0].startswith("🏆"):
            ax_best.text(0.5, y, row[0], color=YELLOW, fontsize=11,
                         fontweight="bold", ha="center", transform=ax_best.transAxes)
        elif row[1] == "":
            ax_best.text(0.1, y, row[0], color=VARIANT_COLORS.get(row[0], TEXT),
                         fontsize=10, fontweight="bold", transform=ax_best.transAxes)
        else:
            ax_best.text(0.05, y, row[0], color=TEXT2, fontsize=9, transform=ax_best.transAxes)
            ax_best.text(0.55, y, row[1], color=TEXT2,  fontsize=9, transform=ax_best.transAxes)
            ax_best.text(0.75, y, row[2], color=GREEN if "→" in row[2] else TEXT,
                         fontsize=9, fontweight="bold", transform=ax_best.transAxes)
        y -= 0.10
    ax_best.set_facecolor(BG2)
    for s in ax_best.spines.values(): s.set_color(GRID_C)

    out = OUT_DIR / "meanrev_regime_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n  ✓ Chart saved → {out}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def print_stats(label, st):
    print(f"\n  {'─'*50}\n  {label}\n  {'─'*50}")
    if not st:
        print("  No trades."); return
    for k, n in [
        ("n_trades","Trades"), ("win_rate","Win Rate"), ("profit_factor","Profit Factor"),
        ("avg_r","Avg R"), ("cagr_pct","CAGR"), ("max_dd_pct","Max DD"), ("sharpe","Sharpe"),
        ("final_equity","Final Equity"),
    ]:
        v = st.get(k, 0)
        f = f"${v:,.0f}" if "equity" in k else f"{v:.2f}{'%' if 'rate' in k or 'pct' in k else ''}"
        print(f"  {n:<20} {f:>18}")


def main():
    print("\n" + "═"*60)
    print("  MEAN REVERSION + ML REGIME FILTER")
    print("═"*60)

    # Step 1: Regime signal
    regime_sig = build_regime_signal(0.75)

    # Step 2: Load and prepare M15 data
    print("\n[Data] Loading M15 data and computing indicators...")
    df_raw = load_nas100_m15()
    df_sv  = build_session_vwap_bands(df_raw)
    df_ov  = build_overnight_vwap_bands(df_raw)
    df_bb  = build_bollinger_bands(df_raw)

    # Check overnight coverage
    ov_valid = df_ov["ov_vwap"].notna().sum()
    print(f"[Data] Session VWAP bars  : {df_sv['vwap'].notna().sum():,}")
    print(f"[Data] Overnight VWAP bars: {ov_valid:,} {'(⚠ limited overnight data)' if ov_valid < 1000 else ''}")
    print(f"[Data] Bollinger Band bars: {df_bb['bb_mid'].notna().sum():,}")

    # Prepare DataFrames with datetime column for the engine
    def prep(d):
        return d.copy().reset_index()

    configs = [
        ("Session VWAP",    prep(df_sv), "upper",    "lower",    "vwap"),
        ("Overnight VWAP",  prep(df_ov), "ov_upper", "ov_lower", "ov_vwap"),
        ("Bollinger Bands", prep(df_bb), "bb_upper", "bb_lower", "bb_mid"),
    ]

    results_base   = {}
    results_regime = {}
    stats_base     = {}
    stats_regime   = {}

    for variant, df_v, up, lo, mid in configs:
        print(f"\n[{variant}] Running baseline...")
        rb = run_backtest(df_v, regime_sig, up, lo, mid, variant, use_regime=False)
        results_base[variant]  = rb
        stats_base[variant]    = compute_stats(rb)
        print_stats(f"{variant} — Baseline", stats_base[variant])

        print(f"\n[{variant}] Running regime-filtered...")
        rr = run_backtest(df_v, regime_sig, up, lo, mid, variant, use_regime=True)
        results_regime[variant]  = rr
        stats_regime[variant]    = compute_stats(rr)
        print_stats(f"{variant} — Regime Filtered", stats_regime[variant])

    # Final comparison table
    print("\n\n" + "═"*70)
    print("  FINAL SUMMARY — Regime Filter Impact")
    print("═"*70)
    print(f"  {'Variant':<20} {'Sharpe (Base→Regime)':>22} {'CAGR (Base→Regime)':>22} {'Max DD (Base→Regime)':>24}")
    print("  " + "─"*68)
    for v in ["Session VWAP", "Overnight VWAP", "Bollinger Bands"]:
        b, r = stats_base[v], stats_regime[v]
        print(f"  {v:<20} "
              f"{b.get('sharpe',0):>6.2f} → {r.get('sharpe',0):<6.2f}       "
              f"{b.get('cagr_pct',0):>6.1f}% → {r.get('cagr_pct',0):<6.1f}%    "
              f"{b.get('max_dd_pct',0):>7.1f}% → {r.get('max_dd_pct',0):.1f}%")

    # Plot
    print("\n[Plot] Generating comparison chart...")
    plot_all(results_base, results_regime, stats_base, stats_regime, regime_sig)

    print(f"\n  All results saved to: {OUT_DIR}")
    print("═"*60)


if __name__ == "__main__":
    main()
