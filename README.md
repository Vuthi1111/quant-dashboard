# Quantitative Volatility Regime Matrix v4.0

An institutional-grade Machine Learning pipeline and Terminal Dashboard designed to predict intraday volatility regimes on the Nasdaq 100 (NAS100) and Gold (XAUUSD). By predicting whether the market will be in an expansive (trending) or compressive (range-bound) state, this system acts as a highly accurate quantitative execution gate.

## System Architecture

The pipeline is built with strict defenses against data leakage, utilizing a Walk-Forward Validation engine combined with a real-time MetaTrader 5 (MT5) telemetry bridge.

### 5 Inference Cores
The v4.0 engine runs five distinct LightGBM inference cores simultaneously across different timeframes to provide a complete picture of market microstructure:

1. **Vol Regime 1H Core:** The primary expansive/compressive state predictor.
2. **Vol Regime 4H Core:** The structural higher-timeframe trend context.
3. **Speed of Tape (1H→4H):** Uses raw 1-Minute tick flow to measure the rate at which the market prints new prices, detecting order flow acceleration and burstiness.
4. **Micro-Regime (1M→15M):** Extremely fast short-term tape prediction designed to pause execution algorithms during liquidity vacuums and chop.
5. **VWAP Copilot (15M - GOLD Only):** Identifies extreme statistical deviations from the Anchored Daily VWAP to signal high-probability mean-reversion setups.

### Feature Engineering
The model consumes 1-Minute and 1-Hour tick data and engineers features used to measure market turbulence:
* **Garman-Klass Volatility:** Captures intra-bar extreme movements superior to standard deviation.
* **Heterogeneous Autoregressive (HAR) Volatility:** Measures the persistence of volatility across daily, weekly, and monthly horizons.
* **RiskMetrics EWMA:** Exponentially weighted moving average of squared returns to capture volatility clustering.
* **Tick Volume Acceleration & Active Ratio:** Used by the Speed of Tape cores to measure real market flow vs. empty prints.

---

## Terminal Dashboard Interface

The system features a zero-latency, pure-Python Terminal User Interface (TUI) powered by the `Textual` framework, providing real-time data streaming and advanced visual analytics directly in the console.

### Multi-Asset Telemetry
Real-time tracking of NAS100 and GOLD, displaying the last close, session highs/lows, exact tick volume, and live Bid/Ask spreads directly piped from MetaTrader 5.

### Macro Environment (T-1)
Live tracking of systemic macro drivers that dictate overall asset flows:
* **VIX:** Systemic Equity Risk Premium
* **DXY:** US Dollar Index
* **TNX:** US 10-Year Treasury Yield
* **HYG:** High-Yield Corporate Credit

### Execution Matrix & Macro News Integration
The Execution Matrix actively locks or unlocks based on the LightGBM probability engine and the ADR Exhaustion filter. Additionally, the dashboard automatically polls the ForexFactory API for High-Impact USD macroeconomic data, initiating strict "News Blackouts" 2 minutes prior to major releases (e.g., CPI, NFP) to protect the algorithms from spread-widening events.

---

## Project Structure

```text
volatility_regime_model/
├── src/
│   ├── dashboard.py               # Main v4.0 Terminal Dashboard Application
│   ├── live_inference.py          # 5-Core LightGBM Production Wrapper
│   ├── feature_engineering.py     # Volatility Math (GK, HAR, EWMA)
│   ├── news_fetcher.py            # ForexFactory API Integration
│   └── decision_logger.py         # SQLite trade decision logging
├── mt5_bridge/
│   ├── ExportLiveEA.mq5           # Real-time MT5 1H tick extractor
│   └── ExportLiveEA_1M.mq5        # Real-time MT5 1M tick extractor
├── tape_speed_features.py         # Speed of Tape feature pipeline
├── micro_regime_features.py       # Micro-Regime feature pipeline
└── README.md
```

## Usage

This engine is designed to be run locally in an Anaconda environment.

1. Attach `ExportLiveEA.mq5` to your MetaTrader 5 charts for 1H data (NAS100 and GOLD).
2. Attach `ExportLiveEA_1M.mq5` to your MetaTrader 5 charts for 1M data (NAS100 and GOLD). Note: You must change the `InpFileName` parameter for the GOLD chart to `xauusd_live_1m.csv`.
3. Launch the terminal dashboard:

```bash
python3 src/dashboard.py
```
