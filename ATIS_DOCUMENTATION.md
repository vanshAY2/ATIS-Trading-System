# ATIS v4.0 — Full System Documentation
## Autonomous Trading Intelligence System for NIFTY 50 Options

---

## 1. Project Overview

ATIS v4.0 is an **8-model ensemble AI system** that analyzes 10 years of NIFTY 50 minute-level data to generate high-confidence trading signals for NIFTY Options (CE/PE). It uses a 3-layer architecture (L1 → L2 → L3) with a Supervisor meta-learner that weighs all model votes to produce a final BUY/SELL/HOLD decision.

### Key Metrics
| Metric | Value |
|--------|-------|
| Training Data | 1,035,257 bars (10 years, 1-min OHLCV) |
| Features | 256 engineered alpha features |
| Models | 8 (5 tree + 1 LSTM + 1 FinBERT + 1 Supervisor) |
| Supervisor Holdout F1 | 68.28% |
| Supervisor Validation Accuracy | 78.49% |
| Inference Latency | < 2 seconds |
| Target | NIFTY 50 Options (CE/PE) |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                    DATA LAYER                       │
│  Raw CSV → Clean Parquet → 256 Features → Target    │
└───────────────────────┬─────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   L1: BASE   │ │  L2: NEURAL  │ │ L2: SENTIMENT│
│              │ │              │ │              │
│ Trend(CB)    │ │ LSTM (Keras) │ │ FinBERT      │
│ Fibo(XGB)    │ │ 60-bar seq   │ │ News headlines│
│ Candle(CB)   │ │ 128 units    │ │ Sentiment ±1 │
│ Trap(XGB)    │ │              │ │              │
│ LGBM         │ │              │ │              │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        ▼
              ┌─────────────────┐
              │  L3: SUPERVISOR │
              │  LogReg Meta    │
              │  Learner        │
              │  Confidence %   │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ STRIKE SELECTOR │
              │ ATM/OTM Strike  │
              │ Entry/SL/Targets│
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ TRADE MANAGER   │
              │ Lock → Track →  │
              │ WIN/LOSS/TIMEOUT│
              └─────────────────┘
```

---

## 3. Directory Structure

```
C:\ATIS\
├── app.py                          # Dash Dashboard (TradingView-style)
├── live_engine_atis.py             # Full 8-model inference engine
├── run_backtest.py                 # Ensemble backtesting
├── requirements.txt                # Python dependencies
├── nifty_1min_raw.csv              # Raw 10-year OHLCV (64 MB)
│
├── config/
│   └── settings.py                 # Central config (API keys, params, paths)
│
├── data/
│   ├── raw/                        # Raw CSV data
│   ├── processed/                  # Clean parquets + feature matrices
│   │   ├── nifty_1min_clean.parquet
│   │   ├── nifty_5min_clean.parquet
│   │   ├── nifty_features.parquet  # 256-column feature matrix
│   │   └── feature_list.txt        # Ordered feature names
│   └── trades/
│       ├── trade_log.csv           # Trade journal (CSV)
│       └── engine_state.json       # Live engine → dashboard sync
│
├── models/
│   └── saved/
│       ├── trend_catboost.joblib   # L1 model
│       ├── fibo_xgboost.joblib     # L1 model
│       ├── candle_catboost.joblib  # L1 model
│       ├── trap_xgboost.joblib     # L1 model
│       ├── lgbm.joblib             # L1 model
│       ├── lstm.keras              # L2 neural model
│       ├── lstm_scaler.joblib      # LSTM feature scaler
│       ├── supervisor.joblib       # L3 meta-learner
│       ├── supervisor_scaler.joblib
│       ├── selected_features.txt   # RFE 128-feature subset
│       ├── training_report.json    # Full tuning history
│       └── final_report.json       # Final holdout scores
│
└── src/
    ├── data/
    │   ├── clean_data.py           # Raw CSV → clean parquet
    │   └── resample_data.py        # 1min → 5min, 15min, 1H
    ├── features/
    │   └── build_features.py       # 256 alpha feature builder
    ├── models/
    │   ├── train_models.py         # Autonomous training loop
    │   ├── final_resync.py         # Supervisor re-sync tool
    │   └── hyperopt_config.py      # Hyperparameter mutation
    ├── news/
    │   ├── news_fetcher.py         # 4-API + RSS news aggregator
    │   └── finbert_agent.py        # FinBERT sentiment agent
    ├── signals/
    │   ├── strike_selector.py      # ATM strike + Entry/SL/Targets
    │   └── trade_manager.py        # Trade locking + journal
    └── utils/
        └── market_simulator.py     # Dry-run simulation engine
```

---

## 4. Data Pipeline

### 4.1 Raw Data
- **Source**: `nifty_1min_raw.csv` — 10 years of NIFTY 50 index 1-minute OHLCV
- **Columns**: `timestamp, open, high, low, close, volume`
- **Size**: ~64 MB, 1,035,257 rows

### 4.2 Cleaning (`src/data/clean_data.py`)
- Parses timestamp, removes duplicates and null rows
- Validates OHLC integrity (high ≥ open/close ≥ low)
- Outputs: `data/processed/nifty_1min_clean.parquet`

### 4.3 Resampling (`src/data/resample_data.py`)
- Creates 5min, 15min, 1H bar aggregations
- Outputs: `nifty_5min_clean.parquet`, `nifty_15min_clean.parquet`, `nifty_1H_clean.parquet`

---

## 5. Feature Engineering (256 Features)

**File**: `src/features/build_features.py`

The system computes **256 alpha features** from raw OHLCV data, grouped into categories:

### 5.1 Trend Features (~40)
| Feature | Description |
|---------|-------------|
| `ema_9/13/21/50/200` | Exponential Moving Averages |
| `ema_cross_*` | EMA crossover signals (binary) |
| `supertrend_*` | SuperTrend indicator (multiple ATR multipliers) |
| `adx_14/21` | Average Directional Index (trend strength) |
| `macd/macd_signal/macd_hist` | MACD oscillator |

### 5.2 Fibonacci Features (~25)
| Feature | Description |
|---------|-------------|
| `fib_*_dist` | Distance to Fibonacci levels (23.6%, 38.2%, 50%, 61.8%) |
| `pivot_*` | Pivot point levels (R1/R2/R3, S1/S2/S3) |
| `fib_zone` | Current Fibonacci zone classification |

### 5.3 Candlestick Features (~30)
| Feature | Description |
|---------|-------------|
| `body_pct` | Body size as % of total range |
| `upper_wick/lower_wick` | Wick ratios |
| `doji/hammer/engulfing` | Pattern detection (binary) |
| `inside_bar/outside_bar` | Range compression/expansion |

### 5.4 Trap Detection Features (~20)
| Feature | Description |
|---------|-------------|
| `bull_trap/bear_trap` | False breakout detection |
| `fake_breakout_*` | Volume-confirmed breakout failures |
| `gap_up/gap_down` | Opening gap analysis |

### 5.5 Volume Features (~30)
| Feature | Description |
|---------|-------------|
| `volume_delta` | Buy vs sell volume proxy |
| `vpoc` | Volume Point of Control |
| `vwap` | Volume-Weighted Average Price |
| `obv` | On-Balance Volume |
| `vol_ma_ratio` | Volume relative to average |

### 5.6 Volatility Features (~25)
| Feature | Description |
|---------|-------------|
| `atr_14/21` | Average True Range |
| `bb_upper/bb_lower/bb_width` | Bollinger Bands |
| `keltner_*` | Keltner Channel |
| `vix_proxy` | Implied volatility proxy |
| `iv_rank` | IV percentile rank |

### 5.7 Momentum Features (~30)
| Feature | Description |
|---------|-------------|
| `rsi_14/21` | Relative Strength Index |
| `stoch_k/stoch_d` | Stochastic Oscillator |
| `williams_r` | Williams %R |
| `cci_14/20` | Commodity Channel Index |
| `roc_*` | Rate of Change (multiple periods) |

### 5.8 Statistical Features (~30)
| Feature | Description |
|---------|-------------|
| `returns_*` | Multi-period returns |
| `volatility_*` | Rolling standard deviation |
| `skew/kurtosis` | Distribution shape |
| `z_score_*` | Standardized price position |
| `hurst_exponent` | Trend persistence |

### 5.9 Sentiment Feature (~1)
| Feature | Description |
|---------|-------------|
| `sentiment_score` | Offline FinBERT proxy (-1 to +1) |

### 5.10 Target Variable
- **`target`**: 3-class classification
  - `0` = BEARISH (price drops > 0.1% in next 5 bars)
  - `1` = SIDEWAYS (price change < 0.1%)
  - `2` = BULLISH (price rises > 0.1% in next 5 bars)

---

## 6. Model Architecture

### 6.1 L1: Base Models (Tree Ensembles)

| Model | Algorithm | Specialty | Best F1 |
|-------|-----------|-----------|---------|
| `trend_catboost` | CatBoost | EMA/ADX/SuperTrend features | 69.0% |
| `fibo_xgboost` | XGBoost | Fibonacci/Pivot features | 74.3% |
| `candle_catboost` | CatBoost | Candlestick pattern features | 68.8% |
| `trap_xgboost` | XGBoost | Trap detection features | 75.9% |
| `lgbm` | LightGBM | Full 256-feature ensemble | 71.2% |

**Training**: Parallelized via `multiprocessing.Pool` (Windows-safe) with `n_jobs=-1`.

### 6.2 L2: Advanced Models

| Model | Type | Input | Best F1 |
|-------|------|-------|---------|
| `lstm` | Keras LSTM | 60-bar sequences, 128-unit, 2-layer | 57.8% |
| `finbert` | FinBERT Transformer | News headlines → sentiment | N/A (proxy) |

**LSTM**: Uses `float32` precision, `batch_size=1024`, with `StandardScaler` preprocessing.

### 6.3 L3: Supervisor Meta-Learner

| Model | Type | Input | Holdout F1 |
|-------|------|-------|------------|
| `supervisor` | Logistic Regression | All L1/L2 predictions | 68.3% |

**How it works**: Takes the 7 model predictions as input features, applies `StandardScaler`, then uses a balanced Logistic Regression to produce a final 3-class prediction with calibrated confidence probabilities.

### 6.4 RFE Feature Selection
- Recursive Feature Elimination reduces 256 → 128 features
- Selected features stored in `models/saved/selected_features.txt`
- Models trained on the RFE subset have better generalization

---

## 7. Training Pipeline

**File**: `src/models/train_models.py`

### 7.1 Data Split
| Split | Date Range | Rows | Purpose |
|-------|------------|------|---------|
| Training | Pre-2024 | 830,629 | Model fitting |
| Validation | 15% of training | ~124,595 | Hyperparameter tuning |
| Holdout | 2024-2026 | 204,628 | Final unbiased evaluation |

### 7.2 Autonomous Tuning Loop
```
for iteration in range(1, MAX_TUNING_ITERATIONS + 1):
    1. Mutate hyperparameters (perturbation-based)
    2. Train all unlocked models in parallel
    3. Evaluate on validation set
    4. If model F1 >= 70% → LOCK (skip in future iterations)
    5. Train L2 (LSTM, FinBERT)
    6. Train L3 (Supervisor on L1/L2 predictions)
    7. Save training_report.json
```

### 7.3 Resume & Model Locking
- Models already achieving ≥70% F1 are **locked** (skipped in future iterations)
- Resume mode: loads existing `.joblib` files and verifies feature shape compatibility
- Shape mismatch (256 vs 128 features) triggers automatic retrain

### 7.4 Running Training
```powershell
# Full autonomous training (10 iterations)
python src/models/train_models.py

# Supervisor re-sync (after RFE mismatch)
python src/models/final_resync.py
```

---

## 8. Signal Generation

### 8.1 Strike Selector (`src/signals/strike_selector.py`)

**ATM Calculation**: `ATM = round(spot / 50) * 50`

| Confidence | Strike Selection |
|------------|-----------------|
| ≥ 85% | ATM (At The Money) |
| ≥ 75% | 1 OTM (One strike Out of The Money) |
| < 75% | 2 OTM |

**Premium Estimation** (Offline): Simplified Black-Scholes proxy using:
- Intrinsic value + time value × OTM decay

**Risk Management**:
| Parameter | Value |
|-----------|-------|
| SL | 30% of premium |
| T1 | 1:1 Risk:Reward |
| T2 | 1:2 Risk:Reward |
| T3 | 1:3 Risk:Reward |
| Min Confidence | 60% |
| Lot Size | 25 |

### 8.2 Trade Manager (`src/signals/trade_manager.py`)

**Trade Lifecycle**:
```
Signal Fires → LOCK (freeze entry/SL/targets)
    ↓
Every tick: check premium vs SL and T1/T2/T3
    ↓
If premium ≤ SL → LOSS (auto-close, log to CSV)
If premium ≥ T3 → WIN (auto-close, log to CSV)
If EOD → TIMEOUT (force close)
    ↓
While LOCKED: no new signals accepted
```

**Journal**: All trades logged to `data/trades/trade_log.csv` with entry, exit, P&L, and confidence.

---

## 9. Dashboard (`app.py`)

### 9.1 Technology Stack
- **Plotly Dash** + **Dash Bootstrap Components** (DARKLY theme)
- **Plotly.js** for TradingView-style candlestick charts
- **Inter** font (Google Fonts) for premium typography

### 9.2 Features
| Component | Description |
|-----------|-------------|
| **Candlestick Chart** | OHLCV with EMA/BB overlays, no weekend/overnight gaps |
| **Signal Card** | Locked trade with live T1/T2/T3 hit tracking |
| **VIX Regime** | Color-coded volatility gauge (Low/Med/High) |
| **Model Consensus** | Per-model accuracy bars with color thresholds |
| **Equity Curve** | Cumulative P&L chart from trade journal |
| **Trade Journal** | Last 10 trades in a scrollable table |
| **News Sentiment** | FinBERT-scored headlines with impact indicators |
| **🧠 Model Analysis** | Modal showing all 7 model predictions + Supervisor reasoning |

### 9.3 Refresh Behavior
- **Chart + Signal Card**: 1-second silent refresh (`uirevision` preserves zoom/pan)
- **Model Gauges + News**: 30-second background refresh
- **No flicker**: All updates are diff-based, no full-page reload

---

## 10. Live Engine (`live_engine_atis.py`)

### 10.1 Inference Pipeline (< 2 seconds)
```
1. Get latest 300 bars (Angel One API or offline replay)
2. Build 256 features from bars
3. L1: Run 5 tree models in sequence
4. L2: Run LSTM (60-bar sequence) + FinBERT (news sentiment)
5. L3: Supervisor meta-prediction + confidence
6. If confidence ≥ 60% → Generate signal
7. If signal valid → Lock trade via TradeManager
8. Export state to engine_state.json (dashboard reads this)
```

### 10.2 Running Modes
```powershell
# Live mode (Angel One API required)
python live_engine_atis.py

# Simulation mode (replays historical data)
python live_engine_atis.py --simulate --interval 1 --duration 300

# Custom interval
python live_engine_atis.py --interval 30
```

### 10.3 Angel One Integration (Ready to Activate)
- API credentials in `config/settings.py`
- SmartConnect SDK integration placeholder in `connect_angel_one()`
- WebSocket (SmartStream) stubs for real-time data push

---

## 11. Market Simulator (`src/utils/market_simulator.py`)

**Purpose**: End-to-end pipeline testing without live API.

### How It Works
1. Loads last 2,000 historical bars
2. Replays them one-by-one with slight noise
3. Generates simulated model predictions (biased by recent trend)
4. Uses **real TradeManager** for trade locking
5. Writes `engine_state.json` every tick
6. Dashboard reads this → displays locked trades in real-time

### Testing Procedure
```powershell
# Terminal 1: Start simulator
python live_engine_atis.py --simulate --interval 1 --duration 300

# Terminal 2: Start dashboard
python app.py

# Open browser: http://127.0.0.1:8050
```

---

## 12. News & Sentiment

### 12.1 News Fetcher (`src/news/news_fetcher.py`)
**4 API Sources + RSS** with rate-limit rotation:

| Source | Daily Limit | Key Required |
|--------|-------------|-------------|
| NewsAPI | 500 | Yes |
| GNews | 100 | Yes |
| NewsData | 200 | Yes |
| MarketAux | 100 | Yes |
| RSS Feeds | Unlimited | No |

**RSS Feeds** (always active):
- Economic Times Markets
- MoneyControl Market Reports
- LiveMint Markets

### 12.2 FinBERT Agent (`src/news/finbert_agent.py`)
- Runs `ProsusAI/finbert` transformer for financial sentiment
- Scores each headline: -1 (bearish) to +1 (bullish)
- Aggregates into a single `finbert_sentiment_score`
- High-impact keyword detection (RBI, Fed, crash, etc.)
- **News Override**: If high-impact negative news + bullish model → confidence dampened by 40%

---

## 13. Configuration Reference

**File**: `config/settings.py`

### Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `HOLDOUT_START` | `2024-01-01` | Start of holdout test period |
| `TARGET_ACCURACY` | `0.70` | Minimum F1 to lock a model |
| `MAX_TUNING_ITERATIONS` | `10` | Max hyperparameter search rounds |
| `CONFIDENCE_GATE` | `0.60` | Min confidence to generate signal |
| `SL_PERCENT` | `0.30` | Stop-loss at 30% of premium |
| `STRIKE_INTERVAL` | `50` | NIFTY strike price granularity |
| `NIFTY_LOT_SIZE` | `25` | Contracts per lot |
| `RANDOM_STATE` | `42` | Reproducibility seed |

---

## 14. Dependencies

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
catboost>=1.2
lightgbm>=4.0
tensorflow>=2.15
transformers>=4.35
torch>=2.0
plotly>=5.18
dash>=2.14
dash-bootstrap-components>=1.5
feedparser>=6.0
requests>=2.31
joblib>=1.3
```

---

## 15. Quick Start

```powershell
# 1. Setup
cd C:\ATIS
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Prepare data
python src/data/clean_data.py
python src/data/resample_data.py
python src/features/build_features.py

# 3. Train models (autonomous, 10 iterations)
python src/models/train_models.py

# 4. Resync supervisor (if needed)
python src/models/final_resync.py

# 5. Backtest
python run_backtest.py

# 6. Launch dashboard
python app.py

# 7. Test with simulator
python live_engine_atis.py --simulate --interval 1 --duration 300
```

---

> **ATIS v4.0** — Built for autonomous NIFTY 50 Options intelligence.
> System designed as a modular, production-ready trading intelligence platform.
