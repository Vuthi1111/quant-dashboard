# Quantitative Volatility Regime Matrix

An institutional-grade Machine Learning pipeline and Terminal Dashboard designed to predict intraday volatility regimes on the Nasdaq 100 (NAS100) and Gold (XAUUSD). By predicting whether the market will be in an expansive (trending) or compressive (range-bound) state, this system acts as a highly accurate quantitative execution gate.

## System Architecture

The pipeline is built with strict defenses against data leakage, utilizing a Walk-Forward Validation engine combined with a real-time MetaTrader 5 (MT5) telemetry bridge.

### 1. Feature Engineering
The model consumes 1-Minute and 1-Hour tick data and engineers features used to measure market turbulence:
* **Garman-Klass Volatility:** Captures intra-bar extreme movements superior to standard deviation.
* **Heterogeneous Autoregressive (HAR) Volatility:** Measures the persistence of volatility across daily, weekly, and monthly horizons.
* **RiskMetrics EWMA:** Exponentially weighted moving average of squared returns to capture volatility clustering.

### 2. Live Inference Engine
Using LightGBM (Gradient Boosted Trees), the model constantly recalculates the probability of entering a high-volatility regime.
* **P > 0.70 (EXPANSIVE):** The market is entering a highly expansive, trending state. The system instructs executing Trend Following strategies.
* **P < 0.30 (COMPRESSIVE):** The market is range-bound. The system instructs executing Mean Reversion strategies and fading extremes.
* **0.30 < P < 0.70 (UNCERTAIN):** No statistical edge. Cash position advised.

---

## Terminal Dashboard Interface

The system features a zero-latency, pure-Python Terminal User Interface (TUI) powered by the `Textual` framework, providing real-time data streaming and advanced visual analytics directly in the console.

### Multi-Asset Telemetry
Real-time tracking of NAS100 and GOLD, displaying the last close, session highs/lows, exact tick volume, and live Bid/Ask spreads directly piped from MetaTrader 5.

### ADR Exhaustion Tracker
Measures the current intraday range against the rolling 20-Day Average Daily Range (ADR). This acts as a secondary filter against "Volatility Budget Exhaustion."
* **< 40% (Green):** Trend continuation is highly likely.
* **40% - 80% (Yellow):** Breakout edge deteriorates.
* **> 80% (Red):** The daily volatility budget is exhausted. Breakout trades are highly likely to fail and mean-revert.

### Execution Matrix & Macro News Integration
The Execution Matrix actively locks or unlocks based on the LightGBM probability engine and the ADR Exhaustion filter. Additionally, the dashboard automatically polls the ForexFactory API for High-Impact USD macroeconomic data, initiating strict "News Blackouts" 2 minutes prior to major releases (e.g., CPI, NFP) to protect the algorithms from spread-widening events.

---

## Backtest & Strategic Impact

We tested the ML Regime filter by wrapping it around two completely different algorithmic strategies:

### 1. Trend Following (ORB)
* **Without Filter:** The baseline strategy suffers a -14.5% Maximum Drawdown.
* **With ML Filter:** By skipping the compressive days where breakouts fail, the Maximum Drawdown was slashed to -7.9%, effectively cutting risk in half while preserving the edge.

### 2. Mean Reversion
* **Without Filter:** Standard band-fade strategies suffer massive drawdowns (-70% to -95%) during aggressive trend days.
* **With ML Filter:** By strictly limiting mean-reversion trades to the predicted low volatility days, drawdowns were mathematically halved across all variants tested.

---

## Project Structure

```text
volatility_regime_model/
├── src/
│   ├── dashboard.py               # Main Terminal Dashboard Application
│   ├── live_inference.py          # LightGBM Production Model Wrapper
│   ├── feature_engineering.py     # Volatility Math (GK, HAR, EWMA)
│   ├── macro_data.py              # ForexFactory API Integration
│   ├── run_pipeline.py            # Walk-forward train/test engine
│   └── walk_forward.py            # ML Cross-Validation engine
├── mt5_bridge/
│   ├── ExportLiveEA.mq5           # Real-time MT5 tick extractor
│   └── ExportLive1H.mq5           # Historical data extractor
└── README.md
```

## Usage

This engine is designed to be run locally in an Anaconda environment.

1. Attach the `ExportLiveEA.mq5` to your MetaTrader 5 charts for NAS100 and GOLD.
2. Launch the terminal dashboard:

```bash
/opt/anaconda3/bin/python src/dashboard.py
```
