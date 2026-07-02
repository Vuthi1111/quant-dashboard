"""
=========================================================
  VOLATILITY REGIME MATRIX — INTERACTIVE TUI DASHBOARD
  Powered by Textual + LightGBM + MT5
=========================================================
"""

import os
import sys
import string
import requests
import asyncio
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import pandas as pd

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich.bar import Bar
from rich.console import Group

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, TabbedContent, TabPane, Label, Log, Select, Checkbox
from textual.containers import Grid, Vertical, Horizontal

# ─────────────────────────────────────────────────────────────────────────────
# PATH RESOLUTION — works regardless of cwd
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live_inference import train_production_model, LIVE_NAS100_PATH, LIVE_GOLD_PATH, PROB_HIGH, PROB_LOW
from feature_engineering import load_mt5_csv, build_features

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
NEWS_BUFFER_MINUTES   = 2
PROB_HISTORY_LEN      = 60      # last N ticks to keep for sparkline
LOG_MAX_LINES         = 200

# ─────────────────────────────────────────────────────────────────────────────
# MACRO NEWS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def fetch_high_impact_news():
    """Fetches today's high-impact USD news from ForexFactory."""
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        response = requests.get(url, timeout=10)
        root = ET.fromstring(response.content)
        events, today_str = [], datetime.now().strftime("%m-%d-%Y")
        for event in root.findall('event'):
            country = event.find('country').text
            impact  = event.find('impact').text
            d_str   = event.find('date').text
            t_str   = event.find('time').text
            title   = event.find('title').text
            if country == "USD" and impact == "High" and d_str == today_str:
                try:
                    # Parse ForexFactory time (which is US Eastern Time)
                    naive_dt = datetime.strptime(f"{d_str} {t_str}", "%m-%d-%Y %I:%M%p")
                    eastern_dt = naive_dt.replace(tzinfo=ZoneInfo("America/New_York"))
                    
                    # Convert to system's local timezone, then make it naive for compatibility
                    local_dt = eastern_dt.astimezone()
                    local_naive = local_dt.replace(tzinfo=None)
                    
                    events.append({"title": title, "dt": local_naive})
                except Exception:
                    pass
        return events
    except Exception:
        return []

def check_news_blackout(events):
    now = datetime.now()
    for ev in events:
        if ev['dt'] - timedelta(minutes=NEWS_BUFFER_MINUTES) <= now <= ev['dt'] + timedelta(minutes=NEWS_BUFFER_MINUTES):
            return True, ev['title']
    return False, None

# ─────────────────────────────────────────────────────────────────────────────
# RENDERERS & ALGORITHMS
# ─────────────────────────────────────────────────────────────────────────────
SPARK_CHARS = " ▁▂▃▄▅▆▇█"

def render_sparkline(history: deque) -> str:
    """Render a mini probability sparkline from a deque of floats [0..1]."""
    if len(history) < 2:
        return "[dim]──────────────────────────[/dim]"
    vals = list(history)
    mn, mx = 0.0, 1.0
    spans = len(SPARK_CHARS) - 1
    chars = []
    for v in vals[-40:]:                       # show last 40 ticks
        idx = int(round(max(0, min(spans, (v - mn) / (mx - mn) * spans))))
        chars.append(SPARK_CHARS[idx])
    latest = vals[-1]
    color = "green" if latest > PROB_HIGH else ("red" if latest < PROB_LOW else "yellow")
    return f"[{color}]{''.join(chars)}[/{color}]"

def render_volume_bars(volumes: np.ndarray) -> str:
    """Render a mini volume histogram from an array of tick volumes."""
    if len(volumes) < 2:
        return "[dim]──────────────────────────[/dim]"
    
    mn, mx = 0, max(volumes)
    if mx == 0: mx = 1
    
    spans = len(SPARK_CHARS) - 1
    chars = []
    # Show last 80 bars to fit nicely in the panel
    for v in volumes[-80:]:
        idx = int(round((v / mx) * spans))
        chars.append(SPARK_CHARS[idx])
        
    return f"[cyan]{''.join(chars)}[/cyan]"



# ─────────────────────────────────────────────────────────────────────────────
# PANEL BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def make_market_panel(raw_state, last_dt, asset_title: str) -> Panel:
    close = raw_state['Close'].values[0]
    high  = raw_state['High'].values[0]
    low   = raw_state['Low'].values[0]
    spread = raw_state['spread'].values[0]
    vol   = raw_state['Tick_Volume'].values[0]

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


def make_ai_panel(prob_high: float, gk_current: float, gk_ratio: float,
                  ewma_trend: str, history: deque) -> Panel:
    prob_pct = prob_high * 100
    if prob_high > PROB_HIGH:
        prob_color = "bright_green"
        state_text = "[b bright_green]▲  EXPANSIVE REGIME[/b bright_green]"
    elif prob_high < PROB_LOW:
        prob_color = "bright_red"
        state_text = "[b bright_red]▼  COMPRESSIVE REGIME[/b bright_red]"
    else:
        prob_color = "bright_yellow"
        state_text = "[b bright_yellow]◆  UNCERTAIN / CHOPPY[/b bright_yellow]"

    filled = int(prob_high * 50)
    bar_str = (
        f"[{prob_color}]{'█' * filled}[/{prob_color}]"
        f"[dim]{'░' * (50 - filled)}[/dim]"
    )

    t = Table(show_header=False, expand=True, box=None, padding=(0, 1))
    t.add_column("Label", style="dim", ratio=2)
    t.add_column("Value", justify="right", ratio=3)

    t.add_row("HIGH-VOL PROBABILITY", f"[b {prob_color}]{prob_pct:5.1f}%[/b {prob_color}]")
    t.add_row("REGIME STATE", state_text)
    t.add_row("", "")
    t.add_row("[dim]CONFIDENCE[/dim]", bar_str)
    t.add_row("PROBABILITY HISTORY", render_sparkline(history))
    t.add_row("", "")
    t.add_row("GARMAN-KLASS (10H)", f"[cyan]{gk_current:.6f}[/cyan]")
    t.add_row("GK vs 30D BASELINE", f"[{'green' if gk_ratio > 1 else 'dim'}]{gk_ratio:.2f}x[/]")
    t.add_row("EWMA MOMENTUM",       ewma_trend)
    t.add_row("", "")
    t.add_row("THRESHOLD  HIGH / LOW", f"[green]{PROB_HIGH*100:.0f}%[/green] / [red]{PROB_LOW*100:.0f}%[/red]")

    return Panel(t, title=" ◈  REGIME INFERENCE MATRIX ", border_style="magenta", expand=True)


def make_execution_panel(prob_high: float, is_blackout: bool, blackout_title: str,
                         update_count: int, adr_exhaustion: float = 0.0) -> Panel:
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
            f"[dim]ADR EXH :[/dim]  {adr_str}\n\n",
            f"[dim]CYCLE  {tick_sym}  {datetime.now().strftime('%H:%M:%S')}[/dim]",
        ]
        color = "bright_green"
    elif prob_high < PROB_LOW:
        lines = [
            f"[b bright_red]  ▼  COMPRESSION REGIME — ALGO LOCKED  [/b bright_red]\n",
            "[red]──────────────────────────────────────────[/red]\n\n",
            "[dim]STRATEGY:[/dim]  [b white]MEAN REVERSION[/b white]\n",
            "[dim]ACTION  :[/dim]  [b bright_red]FADE EXTREMES — AVOID TREND ENTRIES[/b bright_red]\n",
            f"[dim]ADR EXH :[/dim]  {adr_str}\n\n",
            f"[dim]CYCLE  {tick_sym}  {datetime.now().strftime('%H:%M:%S')}[/dim]",
        ]
        color = "bright_red"
    else:
        lines = [
            f"[b bright_yellow]  ◆  UNCERTAIN REGIME — CASH POSITION  [/b bright_yellow]\n",
            "[yellow]──────────────────────────────────────────[/yellow]\n\n",
            "[dim]STRATEGY:[/dim]  [b white]HOLD — NO STATISTICAL EDGE[/b white]\n",
            "[dim]ACTION  :[/dim]  [b bright_yellow]SIT OUT — AWAIT REGIME CLARITY[/b bright_yellow]\n",
            f"[dim]ADR EXH :[/dim]  {adr_str}\n\n",
            f"[dim]CYCLE  {tick_sym}  {datetime.now().strftime('%H:%M:%S')}[/dim]",
        ]
        color = "bright_yellow"

    body = "".join(lines)
    return Panel(Text.from_markup(body, justify="center"),
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
            t_str   = ev['dt'].strftime("%H:%M")
            delta   = ev['dt'] - now
            secs    = delta.total_seconds()
            if secs < -NEWS_BUFFER_MINUTES * 60:
                past_secs = abs(secs)
                hrs, rem = divmod(int(past_secs), 3600)
                mins = rem // 60
                if hrs > 0:
                    status = f"[dim]{hrs}h {mins}m ago[/dim]"
                else:
                    status = f"[dim]{mins}m ago[/dim]"
                label  = f"[dim]{ev['title']}[/dim]"
            elif abs(secs) <= NEWS_BUFFER_MINUTES * 60:
                status = "[b white on red] LIVE [/b white on red]"
                label  = f"[b bright_red]{ev['title']}[/b bright_red]"
            elif secs <= 900:           # within 15 min
                status = f"[yellow]~{int(secs//60)}m[/yellow]"
                label  = f"[yellow]{ev['title']}[/yellow]"
            else:
                hrs, rem = divmod(int(secs), 3600)
                mins = rem // 60
                status = f"[dim]{hrs}h{mins:02d}m[/dim]"
                label  = ev['title']
            t.add_row(t_str, label, status)

    border = "red" if is_blackout else "dark_red"
    return Panel(t, title=" ◈  MACROECONOMIC CALENDAR  (USD HIGH IMPACT) ",
                 border_style=border, expand=True)

def make_liquidity_panel(asset_title: str, df: pd.DataFrame) -> Panel:
    volumes = df['Tick_Volume'].tail(120).values
    current_vol = volumes[-1]
    avg_vol = np.mean(volumes[:-1]) if len(volumes) > 1 else current_vol
    
    if avg_vol == 0: avg_vol = 1
    rel_vol = (current_vol / avg_vol) * 100
    
    vol_color = "bright_green" if rel_vol > 150 else ("white" if rel_vol > 80 else "dim")
    
    t = Table(show_header=False, expand=True, box=None, padding=(0, 1))
    t.add_column("Label", style="dim", ratio=2)
    t.add_column("Value", justify="right", ratio=3)
    
    t.add_row("[bold white]INSTRUMENT[/bold white]", f"[b bright_cyan]{asset_title}[/b bright_cyan]")
    t.add_row("", "")
    t.add_row("CURRENT BAR VOLUME", f"[b {vol_color}]{int(current_vol):,}[/b {vol_color}]")
    t.add_row("MOVING AVERAGE (120)", f"[white]{int(avg_vol):,}[/white]")
    t.add_row("RELATIVE LIQUIDITY", f"[b {vol_color}]{rel_vol:.1f}%[/b {vol_color}]")
    t.add_row("", "")
    t.add_row("LIQUIDITY DENSITY MAP", render_volume_bars(volumes))
    
    return Panel(t, title=" ◈  LIQUIDITY PROFILER ", border_style="blue", expand=True)


def make_waiting_panel(csv_path: Path) -> Panel:
    body = (
        "\n\n[b yellow]⏳  WAITING FOR MT5 LIVE DATA FEED[/b yellow]\n\n"
        "[dim]Ensure[/dim] [cyan]ExportLiveEA.mq5[/cyan] [dim]is attached to a chart in MetaTrader 5.[/dim]\n\n"
        f"[dim]Expected path:[/dim]\n[dim]{csv_path}[/dim]\n"
    )
    return Panel(Text.from_markup(body, justify="center"),
                 title=" ◈  AWAITING FEED ", border_style="yellow", expand=True)


# ─────────────────────────────────────────────────────────────────────────────
# TEXTUAL APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class DashboardApp(App):
    TITLE = "QUANTITATIVE VOLATILITY REGIME MATRIX  v3.1"
    SUB_TITLE = "Multi-Asset Engine · LightGBM · Garman-Klass · RiskMetrics2006"

    CSS = """
    TabbedContent {
        height: 1fr;
    }

    TabPane {
        height: 1fr;
    }

    /* Main grid — 2 columns × 2 rows */
    Grid.main-grid {
        grid-size: 2 2;
        grid-columns: 1fr 1fr;
        grid-rows: 1fr 1fr;
        height: 1fr;
        padding: 0;
        margin: 0;
    }

    /* Macro Grid — 2 columns */
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
    
    /* System Logs tab */
    Log {
        height: 100%;
        border: solid green;
        background: #0a0a0a;
        color: #55ff55;
        padding: 1;
    }
    """

    BINDINGS = [
        ("q", "quit",        "Quit"),
        ("t", "toggle_dark", "Theme"),
        ("r", "reload_news", "Refresh News"),
    ]

    # ── Internal State ──────────────────────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.update_count: int = 0
        self.news_events: list = []
        
        self.prob_history = {
            "NAS100": deque(maxlen=PROB_HISTORY_LEN),
            "GOLD": deque(maxlen=PROB_HISTORY_LEN)
        }
        self.models = {"NAS100": None, "GOLD": None}
        self.scalers = {"NAS100": None, "GOLD": None}
        self.features = {"NAS100": None, "GOLD": None}
        self.adr_20_map = {"NAS100": 1.0, "GOLD": 1.0}

    # ── Layout ──────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="tab-nas100"):
            with TabPane("⚡ NAS100", id="tab-nas100"):
                with Grid(id="nas100-grid", classes="main-grid"):
                    yield Static(Panel(Text("\n\n⏳  Loading market telemetry…", justify="center"), border_style="dim", expand=True), id="nas100_market_data", classes="panel")
                    yield Static(Panel(Text("\n\n⏳  Compiling LightGBM inference core…", justify="center"), border_style="dim", expand=True), id="nas100_ai_core", classes="panel")
                    yield Static(Panel(Text("\n\n⏳  Fetching macro calendar…", justify="center"), border_style="dim", expand=True), id="nas100_news", classes="panel")
                    yield Static(Panel(Text("\n\n🔒  Execution matrix locked (booting)…", justify="center"), border_style="dim", expand=True), id="nas100_execution", classes="panel")
                    
            with TabPane("⚡ GOLD", id="tab-gold"):
                with Grid(id="gold-grid", classes="main-grid"):
                    yield Static(Panel(Text("\n\n⏳  Loading market telemetry…", justify="center"), border_style="dim", expand=True), id="gold_market_data", classes="panel")
                    yield Static(Panel(Text("\n\n⏳  Compiling LightGBM inference core…", justify="center"), border_style="dim", expand=True), id="gold_ai_core", classes="panel")
                    yield Static(Panel(Text("\n\n⏳  Fetching macro calendar…", justify="center"), border_style="dim", expand=True), id="gold_news", classes="panel")
                    yield Static(Panel(Text("\n\n🔒  Execution matrix locked (booting)…", justify="center"), border_style="dim", expand=True), id="gold_execution", classes="panel")

            with TabPane("🌊 Liquidity Profile", id="tab-liquidity"):
                with Grid(id="liquidity-grid", classes="macro-grid"):
                    yield Static(Panel(Text("\n\n⏳  Waiting for NAS100 data stream...", justify="center"), border_style="dim", expand=True), id="nas100_liquidity", classes="panel")
                    yield Static(Panel(Text("\n\n⏳  Waiting for GOLD data stream...", justify="center"), border_style="dim", expand=True), id="gold_liquidity", classes="panel")

            with TabPane("🖥  System Logs", id="tab-logs"):
                yield Log(id="syslog", max_lines=LOG_MAX_LINES)

        yield Footer()

    # ── Boot sequence ────────────────────────────────────────────────────────
    async def on_mount(self) -> None:
        self._log("Dashboard initialised.")
        self._log("Fetching ForexFactory macro calendar…")
        self.news_events = fetch_high_impact_news()
        n = len(self.news_events)
        self._log(f"Calendar loaded — {n} high-impact USD event(s) today.")

        self._log("Training LightGBM production models on historical data…")
        self.notify("⚙  Warming up AI Cores…", timeout=5)
        
        for asset in ["NAS100", "GOLD"]:
            try:
                hist_path = f"/Users/macos/Documents/ALGO/03_Data/raw/{asset}/1h_data.csv"
                df_hist = await asyncio.to_thread(load_mt5_csv, hist_path)
                df_daily = df_hist.resample('D').agg({'High': 'max', 'Low': 'min'}).dropna()
                adr_20 = (df_daily['High'] - df_daily['Low']).rolling(20).mean().iloc[-1]
                self.adr_20_map[asset] = adr_20
                
                m, s, f = await asyncio.to_thread(train_production_model, asset)
                self.models[asset] = m
                self.scalers[asset] = s
                self.features[asset] = f
                self._log(f"LightGBM model for {asset} compiled successfully.")
            except Exception as e:
                self._log(f"[ERROR] {asset} Model training failed: {e}")
                self.notify(f"⚠  {asset} Model error: {e}", severity="error", timeout=10)

        self.notify("✅  System Ready — Multi-Asset Telemetry active.", timeout=4)

        self._log("Starting live polling loop (1 s interval).")
        self.set_interval(1.0, self.update_dashboard)

    # ── Live Update ──────────────────────────────────────────────────────────
    async def update_dashboard(self) -> None:
        self.update_count += 1
        is_blackout, blackout_title = check_news_blackout(self.news_events)

        # Always refresh news panels
        news_panel = make_news_panel(self.news_events, is_blackout)
        self.query_one("#nas100_news", Static).update(news_panel)
        self.query_one("#gold_news", Static).update(news_panel)

        await asyncio.gather(
            self.update_asset_stream("NAS100", LIVE_NAS100_PATH, is_blackout, blackout_title),
            self.update_asset_stream("GOLD", LIVE_GOLD_PATH, is_blackout, blackout_title)
        )

    async def update_asset_stream(self, asset: str, csv_path: Path, is_blackout: bool, blackout_title: str):
        asset_title = "NAS100  (US100)" if asset == "NAS100" else "GOLD  (XAUUSD)"
        prefix = asset.lower()
        
        if self.models[asset] is None:
            return  # Model failed to boot
            
        try:
            df_live = await asyncio.to_thread(load_mt5_csv, str(csv_path))
            if len(df_live) < 500:
                raise ValueError(f"Only {len(df_live)} bars (need ≥500)")
        except Exception as exc:
            self.query_one(f"#{prefix}_market_data", Static).update(make_waiting_panel(csv_path))
            if self.update_count % 10 == 1:
                self._log(f"[WARN] MT5 feed for {asset} unavailable — {exc}")
            return

        try:
            live_features = await asyncio.to_thread(build_features, df_live)
            current_state = live_features.iloc[[-1]]
            raw_state     = df_live.iloc[[-1]]
            last_dt       = live_features.index[-1]

            X_live    = self.scalers[asset].transform(
                current_state[self.features[asset]].values.astype(np.float32))
            prob_high = float(self.models[asset].predict_proba(X_live)[0][1])
            self.prob_history[asset].append(prob_high)

            gk_current = float(current_state["GK_10"].values[0])
            gk_avg     = float(live_features["GK_10"].rolling(24 * 30).mean().iloc[-1])
            gk_ratio   = gk_current / gk_avg if gk_avg > 0 else 1.0

            rm_now  = float(current_state["RM2006"].values[0])
            rm_prev = float(live_features["RM2006"].iloc[-24])
            ewma_trend = (
                "[b bright_green]▲ ACCELERATING[/b bright_green]"
                if rm_now > rm_prev else
                "[b bright_red]▼ DECELERATING[/b bright_red]"
            )

            # Calculate ADR Exhaustion
            today = df_live.index[-1].date()
            df_today = df_live[df_live.index.date == today]
            if len(df_today) > 0 and self.adr_20_map.get(asset, 0) > 0:
                intraday_range = df_today['High'].max() - df_today['Low'].min()
                adr_exhaustion = (intraday_range / self.adr_20_map[asset]) * 100
            else:
                adr_exhaustion = 0.0

            # Execution logic & UI
            self.query_one(f"#{prefix}_market_data", Static).update(
                make_market_panel(raw_state, last_dt, asset_title))
            self.query_one(f"#{prefix}_ai_core", Static).update(
                make_ai_panel(prob_high, gk_current, gk_ratio, ewma_trend, self.prob_history[asset]))
            self.query_one(f"#{prefix}_execution", Static).update(
                make_execution_panel(prob_high, is_blackout, blackout_title, self.update_count, adr_exhaustion))
                
            # Liquidity Updater
            self.query_one(f"#{prefix}_liquidity", Static).update(
                make_liquidity_panel(asset_title, df_live))
                
            if self.update_count % 60 == 0:
                self._log(
                    f"[{asset} TICK #{self.update_count:05d}]  "
                    f"P(High)={prob_high*100:.1f}%  "
                    f"GK={gk_current:.6f}  "
                    f"GK-ratio={gk_ratio:.2f}x  "
                    f"BLACKOUT={'YES' if is_blackout else 'NO'}"
                )

        except Exception as exc:
            err = Panel(
                Text.from_markup(f"[b red]⚠  INFERENCE ERROR ({asset})[/b red]\n\n{exc}", justify="center"),
                border_style="red", expand=True)
            self.query_one(f"#{prefix}_ai_core", Static).update(err)
            self._log(f"[ERROR] Inference failed for {asset}: {exc}")


    # ── Actions ──────────────────────────────────────────────────────────────
    async def action_reload_news(self) -> None:
        self._log("Manual news refresh triggered…")
        self.news_events = fetch_high_impact_news()
        self._log(f"News refreshed — {len(self.news_events)} event(s).")
        self.notify("📰  News calendar refreshed.", timeout=3)

    # ── Helper ───────────────────────────────────────────────────────────────
    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        try:
            self.query_one("#syslog", Log).write_line(f"[{timestamp}]  {message}")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = DashboardApp()
    app.run()
