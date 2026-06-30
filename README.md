# 🤖 Machine Learning Volatility Regime Filter

An institutional-grade Machine Learning pipeline designed to predict intraday volatility regimes on the **Nasdaq 100 (NAS100)**. By predicting whether the next 4 hours will be **HIGH VOL** (trending) or **LOW VOL** (range-bound), this model acts as a highly accurate quality gate for quantitative trading strategies.

## 🎯 The Core Concept
Different trading strategies require different market conditions to survive:
* **Opening Range Breakouts (ORB)** and Trend Following strategies thrive in HIGH VOL conditions.
* **Mean Reversion** (VWAP/Bollinger Band fades) thrive in LOW VOL conditions and get destroyed on trend days.

Instead of trying to build one magic strategy that works in all markets, this project uses a **LightGBM Classifier** to identify the *environment*, allowing you to toggle your strategies on or off before the New York session even begins.

---

## 🔬 System Architecture

The pipeline is built with strict defenses against data leakage (lookahead bias), utilizing a Walk-Forward Validation engine.

### 1. Feature Engineering
The model is fed 1-Hour candlestick data and engineered features that institutions use to measure market turbulence:
* **Garman-Klass Volatility:** Captures intra-bar extreme movements better than standard standard deviation.
* **Heterogeneous Autoregressive (HAR) Volatility:** Measures the persistence of volatility across daily, weekly, and monthly horizons.
* **RiskMetrics EWMA:** Exponentially weighted moving average of squared returns to capture volatility clustering (the tendency for calm days to be followed by calm days, and crazy days by crazy days).

### 2. The Target Variable (Leakage Prevention)
The model attempts to predict the realized variance of the *next* 4 hours. 
> **Critical Engineering Detail:** To prevent same-bar leakage (where the Open/Close of the 1H prediction bar overlaps with the Open/Close of the target measurement), the pipeline enforces a **4-hour embargo** (`bar_offset=4`). The model is strictly blind to the future.

### 3. The Inference Engine
Using `LightGBM` (Gradient Boosted Trees), the model outputs a probability score from 0.0 to 1.0.
* **`P > 0.70` (HIGH VOL):** The market is entering a highly expansive, trending state. (Green light for ORB, Red light for Mean Reversion).
* **`P < 0.30` (LOW VOL):** The market is choppy, compressing, or range-bound. (Red light for ORB, Green light for Mean Reversion).
* **`0.30 < P < 0.70` (UNCERTAIN):** No statistical edge.

---

## 📊 Backtest Results & Strategy Impact

We tested the ML Regime filter by wrapping it around two completely different algorithmic strategies:

### 1. The ORB Strategy (Opening Range Breakout)
* **Without Filter:** The baseline ORB strategy suffers a **-14.5%** Maximum Drawdown.
* **With ML Filter:** By skipping the choppy LOW VOL days where breakouts fail, the Maximum Drawdown was slashed to **-7.9%**, effectively cutting risk in half while preserving the edge.

### 2. Mean Reversion Strategies
* **Without Filter:** Standard VWAP band-fade strategies blow up their accounts (-70% to -95% drawdowns) because they try to fade aggressive trend days.
* **With ML Filter:** By strictly limiting mean-reversion trades to the predicted LOW VOL days, the drawdowns were mathematically halved across all variants (Session VWAP, Overnight VWAP, and Bollinger Bands).

---

## 🛠️ Project Structure
```text
volatility_regime_model/
├── src/
│   ├── feature_engineering.py     # Volatility math (GK, HAR, EWMA)
│   ├── run_pipeline.py            # Walk-forward train/test engine
│   ├── model_stack.py             # Model definitions (LightGBM)
│   ├── visualization.py           # AUC, Calibration, and Equity charting
│   ├── plot_regime_overlay.py     # Plots NQ price action vs Model probabilities
│   └── mean_reversion_backtest.py # Dual-backtest engine for Mean Reversion testing
└── README.md
```

## 🚀 Usage
This engine is designed to be run locally in an Anaconda environment (`/opt/anaconda3/bin/python`).

To run the pipeline and generate fresh predictions:
```bash
python src/run_pipeline.py
```
