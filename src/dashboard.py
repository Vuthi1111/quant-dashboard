"""
=========================================================
  VOLATILITY REGIME MATRIX v4.0 — FULL TUI DASHBOARD
  All 5 inference cores: Vol Regime 1H/4H,
  Speed of Tape 1H→4H, Micro-Regime 1M→15M,
  VWAP Copilot 15M (GOLD)
=========================================================
"""

import os, sys, asyncio, time as time_module
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, TabbedContent, TabPane, Log
from textual.containers import Grid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import load_mt5_csv, build_features, resample_to_4h
from news_fetcher import get_forexfactory_calendar, check_news_blackout
from live_inference import (
    train_production_model, train_vwap_copilot_model,
    compute_vwap_copilot_state,
    train_speed_of_tape_model, compute_speed_of_tape_state,
    train_micro_regime_model, compute_micro_regime_state,
    LIVE_NAS100_PATH, LIVE_GOLD_PATH,
    LIVE_NAS100_PATH_1M, LIVE_GOLD_PATH_1M,
    PROB_HIGH, PROB_LOW,
)
from decision_logger import init_db, log_decision

NEWS_BUFFER_MINUTES = 2
PROB_HISTORY_LEN    = 60
LOG_MAX_LINES       = 200
SPARK_CHARS         = " ▁▂▃▄▅▆▇█"

def render_sparkline(history: deque) -> str:
    if len(history) < 2:
        return "[dim]──────────────────────────[/dim]"
    vals = list(history)
    spans = len(SPARK_CHARS) - 1
    chars = []
    for v in vals[-40:]:
        idx = int(round(max(0, min(spans, (v - 0.0) / (1.0 - 0.0) * spans))))
        chars.append(SPARK_CHARS[idx])
    latest = vals[-1]
    color = "green" if latest > PROB_HIGH else ("red" if latest < PROB_LOW else "yellow")
    return f"[{color}]{''.join(chars)}[/{color}]"

def render_volume_bars(volumes: np.ndarray) -> str:
    if len(volumes) < 2:
        return "[dim]──────────────────────────[/dim]"
    mn, mx = 0, max(volumes) or 1
    spans = len(SPARK_CHARS) - 1
    chars = []
    for v in volumes[-80:]:
        idx = int(round((v / mx) * spans))
        chars.append(SPARK_CHARS[idx])
    return f"[cyan]{''.join(chars)}[/cyan]"

def make_market_panel(raw_state, last_dt, asset_title: str) -> Panel:
    close = float(raw_state['Close'].values[0])
    high  = float(raw_state['High'].values[0])
    low   = float(raw_state['Low'].values[0])
    spread = int(raw_state['spread'].values[0]) if 'spread' in raw_state.columns else 0
    vol   = float(raw_state['Tick_Volume'].values[0]) if 'Tick_Volume' in raw_state.columns else 0

    t = Table(show_header=False, expand=True, box=None, padding=(0, 1))
    t.add_column("Label", style="dim", ratio=2)
    t.add_column("Value", justify="right", ratio=3)
    t.add_row("[bold white]INSTRUMENT[/bold white]", f"[b bright_cyan]{asset_title}[/b bright_cyan]")
    t.add_row("", "")
    t.add_row("LAST CLOSE", f"[b bright_white]{close:,.2f}[/b bright_white]")
    t.add_row("SESSION HIGH", f"[green]{high:,.2f}[/green]")
    t.add_row("SESSION LOW",  f"[red]{low:,.2f}[/red]")
    t.add_row("HL RANGE",     f"[yellow]{high - low:,.2f}[/yellow]")
    t.add_row("", "")
    t.add_row("SPREAD",       f"[yellow]{spread}[/yellow]")
    t.add_row("TICK VOLUME",  f"[cyan]{int(vol):,}[/cyan]")
    t.add_row("", "")
    t.add_row("DATA TIMESTAMP", f"[dim]{last_dt.strftime('%Y-%m-%d  %H:%M')}[/dim]")
    t.add_row("SYSTEM CLOCK",   f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]")
    return Panel(t, title=" ◈  MARKET TELEMETRY ", border_style="bright_blue", expand=True)

def make_vol_regime_panel(timeframe: str, prob: float, history: deque, top_drivers: str = "") -> Panel:
    if top_drivers == "INSUFFICIENT BARS":
        return Panel(
            Text(f"\n\n[dim]INSUFFICIENT DATA BARS FOR {timeframe} CORE[/dim]", justify="center"),
            title=f" ◈  {timeframe} INFERENCE CORE ", border_style="dim", expand=True)
    if prob > PROB_HIGH:
        c, s = "bright_green", "▲ EXPANSIVE"
    elif prob < PROB_LOW:
        c, s = "bright_red", "▼ COMPRESSIVE"
    else:
        c, s = "bright_yellow", "◆ UNCERTAIN"
    filled = int(prob * 50)
    bar = f"[{c}]{'█'*filled}[/{c}][dim]{'░'*(50-filled)}[/dim]"
    t = Table(show_header=False, expand=True, box=None, padding=(0, 1))
    t.add_column("L", style="dim", ratio=1)
    t.add_column("V", justify="right", ratio=2)
    t.add_row("HIGH-VOL PROBABILITY", f"[b {c}]{prob*100:5.1f}%[/]")
    t.add_row("REGIME STATE", f"[b {c}]{s}[/]")
    t.add_row("", "")
    t.add_row("[dim]CONFIDENCE[/dim]", bar)
    t.add_row("PROBABILITY HISTORY", render_sparkline(history))
    t.add_row("[b yellow]PRIMARY DRIVERS[/b yellow]", f"[yellow]{top_drivers}[/]")
    return Panel(t, title=f" ◈  {timeframe} VOL REGIME ", border_style="magenta", expand=True)

def make_speed_tape_panel(state: dict, history: deque) -> Panel:
    prob = state.get('tape_regime_prob', np.nan)
    ar = state.get('active_ratio', np.nan)
    ar_ma = state.get('active_ratio_ma20', np.nan)
    tvz = state.get('tv_zscore_20', np.nan)
    label = state.get('regime_label', 'COMPUTING...')
    color = state.get('regime_color', 'dim')

    t = Table(show_header=False, expand=True, box=None, padding=(0, 1))
    t.add_column("L", style="dim", ratio=2)
    t.add_column("V", justify="right", ratio=3)

    if not np.isnan(prob):
        p_color = "bright_green" if prob > 0.70 else ("bright_red" if prob < 0.30 else "bright_yellow")
        t.add_row("TAPE REGIME PROB", f"[b {p_color}]{prob*100:.1f}%[/b {p_color}]")
        t.add_row("TAPE STATE", f"[b {color}]{label}[/b {color}]")
    else:
        t.add_row("TAPE REGIME PROB", "[dim]—[/dim]")
        t.add_row("TAPE STATE", f"[dim]{label}[/dim]")

    if not np.isnan(ar):
        ar_color = "green" if ar > 0.6 else ("red" if ar < 0.4 else "yellow")
        t.add_row("ACTIVE RATIO (CUR)", f"[{ar_color}]{ar:.2%}[/{ar_color}]")
    else:
        t.add_row("ACTIVE RATIO (CUR)", "[dim]—[/dim]")

    if not np.isnan(ar_ma):
        trend = "▲" if ar_ma > ar else "▼"
        t.add_row("ACTIVE RATIO (20MA)", f"[dim]{ar_ma:.2%} {trend}[/dim]")
    else:
        t.add_row("ACTIVE RATIO (20MA)", "[dim]—[/dim]")

    if not np.isnan(tvz):
        tvz_color = "magenta" if abs(tvz) > 2 else "yellow" if abs(tvz) > 1 else "dim"
        t.add_row("TICK VOL Z-SCORE", f"[{tvz_color}]{tvz:+.2f}σ[/{tvz_color}]")
    else:
        t.add_row("TICK VOL Z-SCORE", "[dim]—[/dim]")

    t.add_row("", "")
    t.add_row("PROBABILITY HISTORY", render_sparkline(history))

    return Panel(t, title=" ◈  SPEED OF TAPE (1H→4H) ", border_style="cyan", expand=True)

def make_micro_regime_panel(state: dict, history: deque) -> Panel:
    prob = state.get('micro_regime_prob', np.nan)
    ar15 = state.get('active_ratio_15', np.nan)
    tvm = state.get('tv_momentum', np.nan)
    sr15 = state.get('silent_ratio_15', np.nan)
    label = state.get('regime_label', 'COMPUTING...')
    color = state.get('regime_color', 'dim')

    t = Table(show_header=False, expand=True, box=None, padding=(0, 1))
    t.add_column("L", style="dim", ratio=2)
    t.add_column("V", justify="right", ratio=3)

    if not np.isnan(prob):
        p_color = "bright_green" if prob > 0.70 else ("bright_red" if prob < 0.30 else "bright_yellow")
        t.add_row("MICRO REGIME PROB", f"[b {p_color}]{prob*100:.1f}%[/b {p_color}]")
        t.add_row("MICRO STATE", f"[b {color}]{label}[/b {color}]")
    else:
        t.add_row("MICRO REGIME PROB", "[dim]—[/dim]")
        t.add_row("MICRO STATE", f"[dim]{label}[/dim]")

    if not np.isnan(ar15):
        ar_color = "green" if ar15 > 0.6 else ("red" if ar15 < 0.4 else "yellow")
        t.add_row("ACTIVE RATIO (15M)", f"[{ar_color}]{ar15:.2%}[/{ar_color}]")
    else:
        t.add_row("ACTIVE RATIO (15M)", "[dim]—[/dim]")

    if not np.isnan(tvm):
        tv_color = "green" if tvm > 1.2 else ("red" if tvm < 0.8 else "yellow")
        t.add_row("TICK VOL MOMENTUM", f"[{tv_color}]{tvm:.2f}x[/{tv_color}]")
    else:
        t.add_row("TICK VOL MOMENTUM", "[dim]—[/dim]")

    if not np.isnan(sr15):
        sr_color = "red" if sr15 > 0.3 else ("yellow" if sr15 > 0.15 else "green")
        t.add_row("SILENT RATIO (15M)", f"[{sr_color}]{sr15:.2%}[/{sr_color}]")
    else:
        t.add_row("SILENT RATIO (15M)", "[dim]—[/dim]")

    t.add_row("", "")
    t.add_row("PROBABILITY HISTORY", render_sparkline(history))

    return Panel(t, title=" ◈  MICRO-REGIME (1M→15M) ", border_style="cyan", expand=True)

def make_execution_panel(prob_high: float, is_blackout: bool, blackout_title: str,
                         update_count: int, adr_exhaustion: float = 0.0,
                         time_in_regime: str = "0m 0s") -> Panel:
    tick_sym = "●" if update_count % 2 == 0 else "○"
    if is_blackout:
        body = (
            f"[b white on red]                                              [/b white on red]\n"
            f"[b white on red]   ⚠  NEWS BLACKOUT — ALGO TRADING HALTED   [/b white on red]\n"
            f"[b white on red]                                              [/b white on red]\n\n"
            f"[b yellow]EVENT:[/b yellow]  {blackout_title}\n"
            f"[dim]Market re-opens in {NEWS_BUFFER_MINUTES} min after event.[/dim]"
        )
        return Panel(Text.from_markup(body, justify="center"),
                     title=" ◈  EXECUTION MATRIX ", border_style="red", expand=True)
    adr_color = "bright_green" if adr_exhaustion < 40 else ("yellow" if adr_exhaustion < 80 else "bright_red")
    adr_str = f"[{adr_color}]{adr_exhaustion:.1f}%[/]"
    if prob_high > PROB_HIGH:
        lines = [
            f"[b bright_green]  ▲  BREAKOUT REGIME — ALGO UNLOCKED  [/b bright_green]\n",
            "[green]──────────────────────────────────────────[/green]\n\n",
            "[dim]STRATEGY:[/dim]  [b white]TREND FOLLOWING[/b white]\n",
            "[dim]ACTION  :[/dim]  [b bright_green]EXECUTE IN DIRECTION OF TREND[/b bright_green]\n",
            f"[dim]ADR EXH :[/dim]  {adr_str}\n",
            f"[dim]STATE TIME:[/dim] {time_in_regime}\n\n",
            f"[dim]CYCLE  {tick_sym}  {datetime.now().strftime('%H:%M:%S')}[/dim]",
        ]
        color = "bright_green"
    elif prob_high < PROB_LOW:
        lines = [
            f"[b bright_red]  ▼  COMPRESSION REGIME — ALGO LOCKED  [/b bright_red]\n",
            "[red]──────────────────────────────────────────[/red]\n\n",
            "[dim]STRATEGY:[/dim]  [b white]MEAN REVERSION[/b white]\n",
            "[dim]ACTION  :[/dim]  [b bright_red]FADE EXTREMES — AVOID TREND ENTRIES[/b bright_red]\n",
            f"[dim]ADR EXH :[/dim]  {adr_str}\n",
            f"[dim]STATE TIME:[/dim] {time_in_regime}\n\n",
            f"[dim]CYCLE  {tick_sym}  {datetime.now().strftime('%H:%M:%S')}[/dim]",
        ]
        color = "bright_red"
    else:
        lines = [
            f"[b bright_yellow]  ◆  UNCERTAIN REGIME — CASH POSITION  [/b bright_yellow]\n",
            "[yellow]──────────────────────────────────────────[/yellow]\n\n",
            "[dim]STRATEGY:[/dim]  [b white]HOLD — NO STATISTICAL EDGE[/b white]\n",
            "[dim]ACTION  :[/dim]  [b bright_yellow]SIT OUT — AWAIT REGIME CLARITY[/b bright_yellow]\n",
            f"[dim]ADR EXH :[/dim]  {adr_str}\n",
            f"[dim]STATE TIME:[/dim] {time_in_regime}\n\n",
            f"[dim]CYCLE  {tick_sym}  {datetime.now().strftime('%H:%M:%S')}[/dim]",
        ]
        color = "bright_yellow"
    return Panel(Text.from_markup("".join(lines), justify="center"),
                 title=" ◈  EXECUTION MATRIX ", border_style=color, expand=True)

def make_news_panel(events: list, is_blackout: bool) -> Panel:
    t = Table(show_header=True, expand=True, box=None, padding=(0, 1))
    t.add_column("TIME (LOCAL)", style="cyan", justify="left", ratio=1)
    t.add_column("EVENT", ratio=4)
    t.add_column("STATUS", justify="right", ratio=1)
    now = datetime.now()
    if not events:
        t.add_row("—", "[dim]No High-Impact USD events today.[/dim]", "")
    else:
        for ev in sorted(events, key=lambda x: x['dt']):
            t_str = ev['dt'].strftime("%H:%M")
            delta = ev['dt'] - now
            secs = delta.total_seconds()
            if secs < -NEWS_BUFFER_MINUTES * 60:
                past_secs = abs(secs)
                hrs, rem = divmod(int(past_secs), 3600)
                mins = rem // 60
                status = f"[dim]{hrs}h {mins}m ago[/dim]" if hrs > 0 else f"[dim]{mins}m ago[/dim]"
                label = f"[dim]{ev['title']}[/dim]"
            elif abs(secs) <= NEWS_BUFFER_MINUTES * 60:
                status = "[b white on red] LIVE [/b white on red]"
                label = f"[b bright_red]{ev['title']}[/b bright_red]"
            elif secs <= 900:
                status = f"[yellow]~{int(secs//60)}m[/yellow]"
                label = f"[yellow]{ev['title']}[/yellow]"
            else:
                hrs, rem = divmod(int(secs), 3600)
                mins = rem // 60
                status = f"[dim]{hrs}h{mins:02d}m[/dim]"
                label = ev['title']
            t.add_row(t_str, label, status)
    border = "red" if is_blackout else "dark_red"
    return Panel(t, title=" ◈  MACROECONOMIC CALENDAR  (USD HIGH IMPACT) ",
                 border_style=border, expand=True)

def make_vwap_copilot_panel(state: dict, prob_history: deque) -> Panel:
    z = state.get('vwap_zscore', float('nan'))
    prob = state.get('ml_probability', float('nan'))
    hurst = state.get('hurst', float('nan'))
    vr = state.get('vol_ratio', float('nan'))
    regime = state.get('regime_context', 'COMPUTING...')
    signal = state.get('signal', 'WARMING UP')
    sig_color = state.get('signal_color', 'dim')
    t = Table(show_header=False, expand=True, box=None, padding=(0, 1))
    t.add_column("L", style="dim", ratio=2)
    t.add_column("V", justify="right", ratio=3)
    if not np.isnan(z):
        z_color = "bright_red" if abs(z) >= 2.0 and z > 0 else ("bright_green" if abs(z) >= 2.0 else "yellow" if abs(z) >= 1.5 else "white")
        t.add_row("VWAP Z-SCORE", f"[b {z_color}]{z:+.2f}σ[/b {z_color}]")
    else:
        t.add_row("VWAP Z-SCORE", "[dim]—[/dim]")
    if not np.isnan(prob):
        p_color = "bright_green" if prob >= 0.70 else ("yellow" if prob >= 0.55 else "dim")
        filled = int(prob * 30)
        bar = f"[{p_color}]{'█'*filled}[/{p_color}][dim]{'░'*(30-filled)}[/dim]"
        t.add_row("ML REVERSION PROB", f"[b {p_color}]{prob*100:.1f}%[/b {p_color}]")
        t.add_row("[dim]CONFIDENCE[/dim]", bar)
    else:
        t.add_row("ML REVERSION PROB", "[dim]—[/dim]")
        t.add_row("[dim]CONFIDENCE[/dim]", "[dim]——————————————————————————[/dim]")
    t.add_row("", "")
    if not np.isnan(hurst):
        h_color = "cyan" if hurst < 0.45 else ("magenta" if hurst > 0.55 else "yellow")
        regime = "MEAN-REVERTING" if hurst < 0.45 else ("TRENDING" if hurst > 0.55 else "RANDOM WALK")
        t.add_row("HURST (32-bar)", f"[{h_color}]{hurst:.3f} ({regime})[/{h_color}]")
    else:
        t.add_row("HURST (32-bar)", "[dim]—[/dim]")
    if not np.isnan(vr):
        vr_color = "magenta" if vr > 1.5 else ("yellow" if vr > 1.0 else "cyan")
        t.add_row("VOL RATIO (16/96)", f"[{vr_color}]{vr:.2f}x[/{vr_color}]")
    else:
        t.add_row("VOL RATIO (16/96)", "[dim]—[/dim]")
    t.add_row("PROB HISTORY", render_sparkline(prob_history))
    t.add_row("", "")
    t.add_row("[b white]SYSTEM STATE[/b white]", f"[b {sig_color}]{signal}[/b {sig_color}]")
    return Panel(t, title=" ◈  15M STATISTICAL COPILOT ", border_style="blue", expand=True)

def make_macro_environment_panel(current_state: pd.DataFrame) -> Panel:
    t_h = Table(show_header=False, expand=True, box=None)
    t_h.add_column("1", justify="center")
    t_h.add_column("2", justify="center")
    t_h.add_column("3", justify="center")
    t_h.add_column("4", justify="center")
    def get_val(col):
        return float(current_state[col].values[0]) if col in current_state.columns else 0.0
    vix, dxy, tnx, hyg = get_val("macro_vix"), get_val("macro_dxy"), get_val("macro_tnx"), get_val("macro_hyg")
    t_h.add_row(
        f"[cyan]VIX:[/cyan] [b white]{vix:.2f}[/]",
        f"[cyan]DXY:[/cyan] [b white]{dxy:.2f}[/]",
        f"[cyan]TNX:[/cyan] [b white]{tnx:.2f}%[/]",
        f"[cyan]HYG:[/cyan] [b white]${hyg:.2f}[/]"
    )
    return Panel(t_h, title=" ◈  MACRO ENVIRONMENT (T-1) ", border_style="cyan", expand=True)

def make_waiting_panel(csv_path: Path) -> Panel:
    body = (
        "\n\n[b yellow]⏳  WAITING FOR MT5 LIVE DATA FEED[/b yellow]\n\n"
        "[dim]Ensure ExportLiveEA or ExportLiveEA_1M is attached.[/dim]\n\n"
        f"[dim]Expected: {csv_path}[/dim]\n"
    )
    return Panel(Text.from_markup(body, justify="center"), title=" ◈  AWAITING FEED ", border_style="yellow", expand=True)

# ─────────────────────────────────────────────────────────────────────────────
# TEXTUAL APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class DashboardApp(App):
    TITLE = "QUANTITATIVE VOLATILITY REGIME MATRIX  v4.0"
    SUB_TITLE = "Multi-Asset Engine · 5 Inference Cores · LightGBM"

    CSS = """
    TabbedContent { height: 1fr; }
    TabPane { height: 1fr; }

    Grid.nas100-grid {
        grid-size: 3 3;
        grid-columns: 1fr 1fr 1fr;
        grid-rows: 1fr 1fr 1fr;
        height: 1fr;
        padding: 0;
        margin: 0;
    }

    Grid.gold-grid {
        grid-size: 3 4;
        grid-columns: 1fr 1fr 1fr;
        grid-rows: 1fr 1fr 1fr 1fr;
        height: 1fr;
        padding: 0;
        margin: 0;
    }

    .span-3 { column-span: 3; }
    .row-span-2 { row-span: 2; }

    Grid.macro-grid {
        grid-size: 2 1;
        grid-columns: 1fr 1fr;
        grid-rows: 1fr;
        height: 1fr;
        padding: 0;
        margin: 0;
    }

    Static.panel {
        height: 1fr;
        width: 100%;
        padding: 0;
        margin: 0;
    }

    Log {
        height: 100%;
        border: solid green;
        background: #0a0a0a;
        color: #55ff55;
        padding: 1;
    }

    Log#boot_log {
        height: 12;
        border: solid #55ff55;
        background: #0a0a0a;
        color: #55ff55;
        margin: 0 0 1 0;
        padding: 0 1;
    }

    #macro_confluence {
        height: 3;
        dock: top;
        margin: 0;
        padding: 0;
    }
    """

    BINDINGS = [
        ("q", "quit",         "Quit"),
        ("t", "toggle_dark",  "Theme"),
        ("r", "reload_news",  "Refresh News"),
        ("w", "log_long",     "Log Long"),
        ("s", "log_short",    "Log Short"),
        ("a", "log_skip",     "Log Skip"),
        ("d", "log_exit",     "Log Exit"),
    ]

    def __init__(self):
        super().__init__()
        self.update_count = 0
        self.news_events = []

        self.prob_history = {
            "NAS100": {"1H": deque(maxlen=PROB_HISTORY_LEN), "4H": deque(maxlen=PROB_HISTORY_LEN),
                       "SPEED_TAPE": deque(maxlen=PROB_HISTORY_LEN), "MICRO_REGIME": deque(maxlen=PROB_HISTORY_LEN)},
            "GOLD":   {"1H": deque(maxlen=PROB_HISTORY_LEN), "4H": deque(maxlen=PROB_HISTORY_LEN),
                       "15M_VWAP": deque(maxlen=PROB_HISTORY_LEN),
                       "SPEED_TAPE": deque(maxlen=PROB_HISTORY_LEN), "MICRO_REGIME": deque(maxlen=PROB_HISTORY_LEN)},
        }
        self.models = {"NAS100": None, "GOLD": None}
        self.speed_tape_models = {"NAS100": None, "GOLD": None}
        self.micro_regime_models = {"NAS100": None, "GOLD": None}
        self.adr_20_map = {"NAS100": 1.0, "GOLD": 1.0}
        self.vwap_copilot = None
        self.vwap_copilot_state = {}
        self.speed_tape_state = {}
        self.micro_regime_state = {}

        self.current_regimes = {"NAS100": {"1H": None, "4H": None}, "GOLD": {"1H": None, "4H": None}}
        self.regime_start_times = {"NAS100": {"1H": None, "4H": None}, "GOLD": {"1H": None, "4H": None}}
        self.regime_is_buffer_limit = {"NAS100": {"1H": False, "4H": False}, "GOLD": {"1H": False, "4H": False}}

        self.latest_snapshots = {"NAS100": None, "GOLD": None}
        self.open_trade_uids = {"NAS100": None, "GOLD": None}
        self.macro_confluence_str = "BOOTING"

    # ── Layout ──────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Log(id="boot_log", max_lines=50, highlight=True)
        yield Static(Panel("", title=" ◈  MACRO CONFLUENCE ", border_style="dim"), id="macro_confluence")
        yield Static(Panel("", title=" ◈  MACRO ENVIRONMENT (T-1) ", border_style="dim"), id="macro_environment")
        with TabbedContent(initial="tab-nas100"):
            with TabPane("⚡ NAS100", id="tab-nas100"):
                with Grid(id="nas100-grid", classes="nas100-grid"):
                    yield Static(Panel(Text("\n\n⏳  Loading…", justify="center"), border_style="dim", expand=True), id="nas100_market_data", classes="panel")
                    yield Static(Panel(Text("\n\n⏳  Loading…", justify="center"), border_style="dim", expand=True), id="nas100_1h_core", classes="panel")
                    yield Static(Panel(Text("\n\n⏳  Loading…", justify="center"), border_style="dim", expand=True), id="nas100_speed_tape", classes="panel")
                    yield Static(Panel(Text("\n\n⏳  Loading…", justify="center"), border_style="dim", expand=True), id="nas100_news", classes="panel")
                    yield Static(Panel(Text("\n\n⏳  Loading…", justify="center"), border_style="dim", expand=True), id="nas100_4h_core", classes="panel")
                    yield Static(Panel(Text("\n\n⏳  Loading…", justify="center"), border_style="dim", expand=True), id="nas100_micro_regime", classes="panel")
                    yield Static(Panel(Text("\n\n🔒  Locked.", justify="center"), border_style="dim", expand=True), id="nas100_execution", classes="panel span-3")

            with TabPane("⚡ GOLD", id="tab-gold"):
                with Grid(id="gold-grid", classes="gold-grid"):
                    yield Static(Panel(Text("\n\n⏳  Loading…", justify="center"), border_style="dim", expand=True), id="gold_market_data", classes="panel")
                    yield Static(Panel(Text("\n\n⏳  Loading…", justify="center"), border_style="dim", expand=True), id="gold_1h_core", classes="panel")
                    yield Static(Panel(Text("\n\n⏳  Loading…", justify="center"), border_style="dim", expand=True), id="gold_vwap_copilot", classes="panel row-span-2")
                    yield Static(Panel(Text("\n\n⏳  Loading…", justify="center"), border_style="dim", expand=True), id="gold_news", classes="panel")
                    yield Static(Panel(Text("\n\n⏳  Loading…", justify="center"), border_style="dim", expand=True), id="gold_4h_core", classes="panel")
                    yield Static(Panel(Text("\n\n⏳  Loading…", justify="center"), border_style="dim", expand=True), id="gold_speed_tape", classes="panel")
                    yield Static(Panel(Text("\n\n⏳  Loading…", justify="center"), border_style="dim", expand=True), id="gold_micro_regime", classes="panel")
                    yield Static(Panel(Text("\n\n🔒  Locked.", justify="center"), border_style="dim", expand=True), id="gold_execution", classes="panel span-3")

            with TabPane("🖥  System Logs", id="tab-logs"):
                yield Log(id="syslog", max_lines=LOG_MAX_LINES)

        yield Footer()

    # ── Boot ──────────────────────────────────────────────────────────────────
    async def on_mount(self) -> None:
        boot_log = self.query_one("#boot_log", Log)
        boot_start = time_module.time()

        async def boot_msg(msg: str):
            elapsed = time_module.time() - boot_start
            boot_log.write_line(f"[{elapsed:6.1f}s]  {msg}")
            await asyncio.sleep(0.05)

        await boot_msg("Dashboard v4.0 initialised — 5 inference cores.")
        await boot_msg("Fetching ForexFactory macro calendar…")
        self.news_events = await asyncio.to_thread(get_forexfactory_calendar)
        await boot_msg(f"Calendar loaded — {len(self.news_events)} high-impact USD event(s) today.")

        await boot_msg("Initializing Discretionary Decision Log DB…")
        await asyncio.to_thread(init_db)

        await boot_msg("=" * 40)
        await boot_msg("PHASE 1/5 — Vol Regime 1H & 4H")
        await boot_msg("=" * 40)
        for asset in ["NAS100", "GOLD"]:
            try:
                if asset == "NAS100":
                    hist_path = "/Users/macos/Documents/ALGO/03_Data/raw/NAS100/1h_data.csv"
                else:
                    hist_path = "/Users/macos/Documents/ALGO/03_Data/raw/GOLD_XAUUSD/XAUUSD_1H.csv"

                await boot_msg(f"Loading & training {asset} Vol Regime…")
                df_hist = await asyncio.to_thread(load_mt5_csv, hist_path)
                df_daily = df_hist.resample('D').agg({'High': 'max', 'Low': 'min'}).dropna()
                adr_20 = (df_daily['High'] - df_daily['Low']).rolling(20).mean().iloc[-1]
                self.adr_20_map[asset] = adr_20

                models_dict = await asyncio.to_thread(train_production_model, asset)
                self.models[asset] = models_dict
                await boot_msg(f"✅ Vol Regime models (1H & 4H) for {asset} compiled.")
            except Exception as e:
                await boot_msg(f"❌ {asset} Vol Regime training failed: {e}")
                self.notify(f"⚠  {asset} Vol Regime error: {e}", severity="error", timeout=10)

        await boot_msg("=" * 40)
        await boot_msg("PHASE 2/5 — Speed of Tape (1H→4H)")
        await boot_msg("=" * 40)
        for asset in ["NAS100", "GOLD"]:
            try:
                await boot_msg(f"Training Speed of Tape ({asset}) on 1M history…")
                tape_model, tape_feat = await asyncio.to_thread(train_speed_of_tape_model, asset)
                self.speed_tape_models[asset] = (tape_model, tape_feat)
                await boot_msg(f"✅ Speed of Tape model for {asset} compiled.")
            except Exception as e:
                await boot_msg(f"❌ {asset} Speed of Tape training failed: {e}")

        await boot_msg("=" * 40)
        await boot_msg("PHASE 3/5 — Micro-Regime (1M→15M)")
        await boot_msg("=" * 40)
        for asset in ["NAS100", "GOLD"]:
            try:
                await boot_msg(f"Training Micro-Regime ({asset}) on 1M history…")
                micro_model, micro_feat = await asyncio.to_thread(train_micro_regime_model, asset)
                self.micro_regime_models[asset] = (micro_model, micro_feat)
                await boot_msg(f"✅ Micro-Regime model for {asset} compiled.")
            except Exception as e:
                await boot_msg(f"❌ {asset} Micro-Regime training failed: {e}")

        await boot_msg("=" * 40)
        await boot_msg("PHASE 4/5 — VWAP Copilot (GOLD 15M)")
        await boot_msg("=" * 40)
        try:
            await boot_msg("Training 15M VWAP Copilot model on historical Gold data…")
            self.vwap_copilot = await asyncio.to_thread(train_vwap_copilot_model)
            await boot_msg("✅ VWAP Copilot compiled.")
        except Exception as e:
            await boot_msg(f"❌ VWAP Copilot training failed: {e}")

        await boot_msg("=" * 40)
        await boot_msg("PHASE 5/5 — Starting live loop")
        await boot_msg("=" * 40)
        self.notify("✅  System Ready — All 5 cores active.", timeout=4)
        elapsed = time_module.time() - boot_start
        await boot_msg(f"Boot complete — {elapsed:.1f}s total. Entering live loop.")
        self.set_interval(1.0, self.update_dashboard)
        self.query_one("#boot_log").remove()

    # ── Live Update ──────────────────────────────────────────────────────────
    async def update_dashboard(self) -> None:
        self.update_count += 1
        is_blackout, blackout_title = check_news_blackout(self.news_events)

        news_panel = make_news_panel(self.news_events, is_blackout)
        self.query_one("#nas100_news", Static).update(news_panel)
        self.query_one("#gold_news", Static).update(news_panel)

        await asyncio.gather(
            self.update_asset_stream("NAS100", LIVE_NAS100_PATH, LIVE_NAS100_PATH_1M, is_blackout, blackout_title),
            self.update_asset_stream("GOLD",   LIVE_GOLD_PATH,   LIVE_GOLD_PATH_1M,   is_blackout, blackout_title)
        )

        nas_reg = self.current_regimes["NAS100"]["1H"]
        gold_reg = self.current_regimes["GOLD"]["1H"]
        if nas_reg and gold_reg:
            if nas_reg == "HIGH" and gold_reg == "HIGH":
                confluence = "[b bright_red]SYSTEMIC VOLATILITY SHOCK (BOTH EXPANDING)[/b bright_red]"
                color = "red"
            elif nas_reg == "LOW" and gold_reg == "LOW":
                confluence = "[b bright_green]SYSTEMIC COMPRESSION (BOTH RANGEBOUND)[/b bright_green]"
                color = "green"
            else:
                confluence = f"[b bright_yellow]DIVERGENT MACRO STATES (NAS100: {nas_reg}  |  GOLD: {gold_reg})[/b bright_yellow]"
                color = "yellow"
            self.macro_confluence_str = Text.from_markup(confluence).plain
            self.query_one("#macro_confluence", Static).update(
                Panel(Text.from_markup(confluence, justify="center"), title=" ◈  MACRO CONFLUENCE ", border_style=color))

    async def _compute_inference(self, asset: str, df_live: pd.DataFrame, timeframe: str):
        model, scaler, feature_cols = self.models[asset][timeframe]
        live_features = await asyncio.to_thread(build_features, df_live)
        current_state = live_features.iloc[[-1]]
        last_dt = live_features.index[-1]
        X_live = scaler.transform(current_state[feature_cols].values.astype(np.float32))
        shap_values = model.booster_.predict(X_live, pred_contrib=True)[0]
        raw_margin = np.sum(shap_values)
        prob_high = float(1.0 / (1.0 + np.exp(-raw_margin)))
        self.prob_history[asset][timeframe].append(prob_high)
        contributions = shap_values[:-1]
        top_indices = np.argsort(np.abs(contributions))[-3:][::-1]
        top_drivers_list = []
        for idx in top_indices:
            feat = feature_cols[idx]
            val = contributions[idx]
            sign = "🟢" if val > 0 else "🔴"
            top_drivers_list.append(f"{feat} {sign}")
        drivers_str = " | ".join(top_drivers_list)

        if self.current_regimes[asset][timeframe] is None:
            X_all = scaler.transform(live_features[feature_cols].values.astype(np.float32))
            probs_all = model.predict_proba(X_all)[:, 1]
            regimes = np.full(len(probs_all), "NEUTRAL", dtype=object)
            regimes[probs_all > PROB_HIGH] = "HIGH"
            regimes[probs_all < PROB_LOW] = "LOW"
            current_regime_val = regimes[-1]
            changed_indices = np.where(regimes != current_regime_val)[0]
            if len(changed_indices) > 0:
                last_change_idx = changed_indices[-1] + 1
                true_start_dt = live_features.index[last_change_idx]
                self.regime_is_buffer_limit[asset][timeframe] = False
            else:
                true_start_dt = live_features.index[0]
                self.regime_is_buffer_limit[asset][timeframe] = True
            self.current_regimes[asset][timeframe] = current_regime_val
            broker_elapsed = last_dt - true_start_dt
            self.regime_start_times[asset][timeframe] = datetime.now() - broker_elapsed

        if prob_high > PROB_HIGH:
            new_regime = "HIGH"
        elif prob_high < PROB_LOW:
            new_regime = "LOW"
        else:
            new_regime = "NEUTRAL"
        if self.current_regimes[asset][timeframe] != new_regime:
            self.current_regimes[asset][timeframe] = new_regime
            self.regime_start_times[asset][timeframe] = datetime.now()
            self.regime_is_buffer_limit[asset][timeframe] = False
        time_in_regime = datetime.now() - self.regime_start_times[asset][timeframe]
        mins, secs = divmod(int(time_in_regime.total_seconds()), 60)
        hrs, mins = divmod(mins, 60)
        buffer_indicator = "+" if self.regime_is_buffer_limit[asset][timeframe] else ""
        time_str = f"{hrs}h {mins}m {secs}s{buffer_indicator}" if hrs > 0 else f"{mins}m {secs}s{buffer_indicator}"
        return prob_high, drivers_str, time_str, live_features, current_state

    async def update_asset_stream(self, asset: str, csv_path: Path, csv_path_1m: Path,
                                   is_blackout: bool, blackout_title: str) -> None:
        prefix = asset.lower()
        try:
            df_live = await asyncio.to_thread(load_mt5_csv, str(csv_path))
            if len(df_live) < 500:
                return
        except Exception as exc:
            self.query_one(f"#{prefix}_market_data", Static).update(make_waiting_panel(csv_path))
            if self.update_count % 10 == 1:
                self._log(f"[WARN] MT5 1H feed for {asset} unavailable — {exc}")
            return

        try:
            prob_1h, drv_1h, time_1h, lf_1h, cs_1h = await self._compute_inference(asset, df_live, "1H")
            df_live_4h = await asyncio.to_thread(resample_to_4h, df_live)
            if len(df_live_4h) < 20:
                prob_4h, drv_4h, time_4h = prob_1h, "INSUFFICIENT BARS", "0m"
            else:
                prob_4h, drv_4h, time_4h, _, _ = await self._compute_inference(asset, df_live_4h, "4H")

            # ── Speed of Tape via 1M live data ─────────────────────────────────
            try:
                df_live_1m = await asyncio.to_thread(load_mt5_csv, str(csv_path_1m))
                if len(df_live_1m) >= 200 and self.speed_tape_models.get(asset) is not None:
                    st_model, st_feat = self.speed_tape_models[asset]
                    self.speed_tape_state = await asyncio.to_thread(
                        compute_speed_of_tape_state, df_live_1m, st_model, st_feat)
                    st_p = self.speed_tape_state.get('tape_regime_prob', np.nan)
                    if not np.isnan(st_p):
                        self.prob_history[asset]["SPEED_TAPE"].append(st_p)
            except Exception as exc:
                if self.update_count % 10 == 1:
                    self._log(f"[WARN] Speed of Tape ({asset}) — {exc}")

            # ── Micro-Regime via 1M live data ──────────────────────────────────
            try:
                if len(df_live_1m) >= 100 and self.micro_regime_models.get(asset) is not None:
                    mr_model, mr_feat = self.micro_regime_models[asset]
                    self.micro_regime_state = await asyncio.to_thread(
                        compute_micro_regime_state, df_live_1m, mr_model, mr_feat)
                    mr_p = self.micro_regime_state.get('micro_regime_prob', np.nan)
                    if not np.isnan(mr_p):
                        self.prob_history[asset]["MICRO_REGIME"].append(mr_p)
            except Exception as exc:
                if self.update_count % 10 == 1:
                    self._log(f"[WARN] Micro-Regime ({asset}) — {exc}")

            # ── VWAP Copilot (GOLD only) ────────────────────────────────────────
            if asset == "GOLD" and self.vwap_copilot is not None:
                if len(df_live_1m) >= 100:
                    vwap_model, vwap_feat_cols = self.vwap_copilot
                    self.vwap_copilot_state = await asyncio.to_thread(
                        compute_vwap_copilot_state, df_live_1m, vwap_model, vwap_feat_cols)
                    ml_p = self.vwap_copilot_state.get('ml_probability', float('nan'))
                    if not np.isnan(ml_p):
                        self.prob_history["GOLD"]["15M_VWAP"].append(ml_p)

            # ── Metrics ─────────────────────────────────────────────────────────
            gk_current = float(cs_1h["GK_10"].values[0])
            gk_avg = float(lf_1h["GK_10"].rolling(24 * 30).mean().iloc[-1])
            gk_ratio = gk_current / gk_avg if gk_avg > 0 else 1.0
            rm_now = float(cs_1h["RM2006"].values[0])
            rm_prev = float(lf_1h["RM2006"].iloc[-24])
            ewma_trend = ("[b bright_green]▲ ACCELERATING[/b bright_green]"
                          if rm_now > rm_prev else "[b bright_red]▼ DECELERATING[/b bright_red]")
            today = df_live.index[-1].date()
            df_today = df_live[df_live.index.date == today]
            if len(df_today) > 0 and self.adr_20_map.get(asset, 0) > 0:
                intraday_range = df_today['High'].max() - df_today['Low'].min()
                adr_exhaustion = (intraday_range / self.adr_20_map[asset]) * 100
            else:
                adr_exhaustion = 0.0

            # ── Render ──────────────────────────────────────────────────────────
            raw_state = df_live.iloc[[-1]]
            last_dt = df_live.index[-1]
            self.query_one(f"#{prefix}_market_data", Static).update(make_market_panel(raw_state, last_dt, asset))
            self.query_one("#macro_environment", Static).update(make_macro_environment_panel(cs_1h))

            self.query_one(f"#{prefix}_1h_core", Static).update(
                make_vol_regime_panel("1H", prob_1h, self.prob_history[asset]["1H"], drv_1h))
            self.query_one(f"#{prefix}_4h_core", Static).update(
                make_vol_regime_panel("4H", prob_4h, self.prob_history[asset]["4H"], drv_4h))

            self.query_one(f"#{prefix}_speed_tape", Static).update(
                make_speed_tape_panel(self.speed_tape_state, self.prob_history[asset]["SPEED_TAPE"]))
            self.query_one(f"#{prefix}_micro_regime", Static).update(
                make_micro_regime_panel(self.micro_regime_state, self.prob_history[asset]["MICRO_REGIME"]))

            if asset == "GOLD":
                self.query_one("#gold_vwap_copilot", Static).update(
                    make_vwap_copilot_panel(self.vwap_copilot_state, self.prob_history["GOLD"]["15M_VWAP"]))

            self.query_one(f"#{prefix}_execution", Static).update(
                make_execution_panel(prob_1h, is_blackout, blackout_title, self.update_count,
                                    adr_exhaustion, time_1h))

            if self.update_count % 60 == 0:
                self._log(
                    f"[{asset} TICK #{self.update_count:05d}]  "
                    f"P(High)={prob_1h*100:.1f}%  "
                    f"GK={gk_current:.6f}  GK-ratio={gk_ratio:.2f}x  "
                    f"BLACKOUT={'YES' if is_blackout else 'NO'}")

            time_in_regime = datetime.now() - self.regime_start_times[asset]["1H"]
            snapshot = {
                "prob_high": prob_1h,
                "regime_state": self.current_regimes[asset]["1H"],
                "state_time_seconds": int(time_in_regime.total_seconds()),
                "gk_current": gk_current,
                "gk_ratio": gk_ratio,
                "top_drivers": drv_1h.split(" | "),
                "macro_confluence": self.macro_confluence_str,
                "is_news_blackout": is_blackout
            }
            if asset == "GOLD" and self.vwap_copilot_state:
                snapshot["vwap_zscore"] = self.vwap_copilot_state.get("vwap_zscore")
                snapshot["ml_reversion_prob"] = self.vwap_copilot_state.get("ml_probability")
                snapshot["hurst"] = self.vwap_copilot_state.get("hurst")
                snapshot["copilot_signal"] = self.vwap_copilot_state.get("signal")
            self.latest_snapshots[asset] = snapshot

        except Exception as exc:
            err = Panel(Text.from_markup(f"[b red]⚠  INFERENCE ERROR ({asset})[/b red]\n\n{exc}", justify="center"),
                        border_style="red", expand=True)
            self.query_one(f"#{prefix}_1h_core", Static).update(err)
            self._log(f"[ERROR] Inference failed for {asset}: {exc}")

    # ── Actions ──────────────────────────────────────────────────────────────
    async def action_reload_news(self) -> None:
        self.news_events = get_forexfactory_calendar()
        self.notify("News calendar updated.")

    def _get_active_asset(self) -> str:
        active_tab = self.query_one(TabbedContent).active
        return "GOLD" if "gold" in active_tab else "NAS100"

    def _log_discretionary(self, action: str, direction: str = None):
        asset = self._get_active_asset()
        snap = self.latest_snapshots.get(asset)
        if not snap:
            self.notify("Waiting for telemetry before logging...", severity="warning")
            return
        uid = self.open_trade_uids.get(asset) if action == "EXIT" else None
        try:
            log_decision(asset, action, snap, direction=direction, trade_uid=uid)
            self.notify(f"Logged {action} {'(' + direction + ')' if direction else ''} for {asset}!", timeout=2)
        except Exception as e:
            self.notify(f"Log Error: {e}", severity="error")

    def action_log_long(self) -> None:
        self._log_discretionary("TAKE", "LONG")
    def action_log_short(self) -> None:
        self._log_discretionary("TAKE", "SHORT")
    def action_log_skip(self) -> None:
        self._log_discretionary("SKIP")
    def action_log_exit(self) -> None:
        self._log_discretionary("EXIT")

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        try:
            self.query_one("#syslog", Log).write_line(f"[{timestamp}]  {message}")
        except Exception:
            pass

if __name__ == "__main__":
    app = DashboardApp()
    app.run()
