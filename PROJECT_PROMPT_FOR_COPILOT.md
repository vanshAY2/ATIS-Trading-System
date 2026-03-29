# 🤖 ATIS v4.0 - Complete Project Prompt for GitHub Copilot

## PROJECT OVERVIEW

**Project Name**: ATIS v4.0 (Autonomous Trading Intelligence System)  
**Environment**: GitHub Codespaces (Ubuntu Linux, Python 3.11, 4-core CPU, 16GB RAM)  
**Objective**: Train an 8-model ensemble ML system for NIFTY 50 Options trading signals  
**Training Data**: 10 years of 1-minute OHLCV bars (~1M bars, 64 MB CSV)  
**Output**: Fully trained production-ready trading system with dashboard

---

## PROJECT ARCHITECTURE

### 🏗️ **System Architecture - 3 Layers**

```
┌─────────────────────────────────────┐
│     DATA LAYER (Raw → Features)     │
│  CSV → Parquet → 256 Alpha Features │
└────────────────┬────────────────────┘
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
   ┌────────┐┌────────┐┌────────┐
   │ L1:    ││ L2:    ││ L2:    │
   │BASE   ││NEURAL ││SENTIMENT
   │MODELS ││       ││        │
   │       ││LSTM   ││FinBERT│
   │5 Tree ││128u   ││News   │
   │Models ││60 seq ││Senti  │
   └───┬───┘└───┬───┘└───┬───┘
       │        │        │
       └────────┼────────┘
                ▼
        ┌───────────────┐
        │ L3: SUPERVISOR│
        │ LogRegression │
        │ Meta-Learner  │
        │ Final Vote    │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │STRIKE SELECTOR│
        │Entry/SL/Gain  │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │TRADE MANAGER  │
        │Track & Journal│
        └───────────────┘
```

---

## 📁 COMPLETE FOLDER STRUCTURE

```
/workspaces/ATIS-Trading-System/
│
├── 📄 .devcontainer/
│   ├── devcontainer.json          # Codespaces environment config
│   └── post-create.sh             # Auto-setup script (already ran)
│
├── 📄 .gitignore                  # Excludes .csv, .parquet, .joblib, .keras
├── 📄 requirements.txt            # All Python dependencies
├── 📄 ATIS_DOCUMENTATION.md       # Full project docs
├── 📄 CODESPACES_SETUP.md        # Codespaces guide
├── 📄 CODESPACES_QUICK_START.md  # Quick reference
├── 📄 CODESPACES_ACTION_PLAN.md  # Action checklist
│
├── 📁 config/
│   ├── __init__.py               # Package initializer
│   └── settings.py               # ⭐ CENTRAL CONFIG
│       ├── API Keys (Angel One, Alpha Vantage, Finnhub)
│       ├── Model Hyperparameters
│       ├── Paths & Directories
│       ├── Thresholds (confidence gates, target accuracy)
│       └── Global Symbols & Timeframes
│
├── 📁 data/
│   ├── raw/
│   │   └── nifty_1min_raw.csv    # ⭐ INPUT: 10 years 1-min OHLCV (~64 MB)
│   │
│   ├── processed/
│   │   ├── nifty_1min_clean.parquet    # Cleaned 1-min bars
│   │   ├── nifty_5min_clean.parquet    # Resampled 5-min bars
│   │   ├── nifty_15min_clean.parquet   # Resampled 15-min bars
│   │   ├── nifty_1H_clean.parquet      # Resampled 1-hour bars
│   │   ├── nifty_features.parquet      # ⭐ 256-column feature matrix
│   │   ├── feature_list.txt            # Ordered feature names
│   │   └── feature_selection.txt       # RFE-selected 128 features
│   │
│   └── trades/
│       ├── trade_log.csv         # Trade journal (entry/exit/P&L)
│       └── engine_state.json     # Live engine state snapshot
│
├── 📁 models/
│   └── saved/
│       ├── trend_catboost.joblib        # ⭐ L1 Model: Trend detection
│       ├── fibo_xgboost.joblib         # ⭐ L1 Model: Fibonacci levels
│       ├── candle_catboost.joblib      # ⭐ L1 Model: Candlestick patterns
│       ├── trap_xgboost.joblib         # ⭐ L1 Model: False breakout traps
│       ├── lgbm.joblib                 # ⭐ L1 Model: LightGBM ensemble
│       │
│       ├── lstm.keras                  # ⭐ L2 Model: LSTM neural network
│       ├── lstm_scaler.joblib          # LSTM feature scaler
│       │
│       ├── supervisor.joblib           # ⭐ L3 Model: Meta-learner
│       ├── supervisor_scaler.joblib    # Supervisor feature scaler
│       │
│       ├── final_report.json           # ⭐ OUTPUT: Final holdout scores
│       ├── training_report.json        # Training history & metrics
│       ├── backtest_report.json        # Backtesting results (if run)
│       └── selected_features.txt       # Final feature subset
│
├── 📁 src/
│   │
│   ├── 📁 data/
│   │   ├── __init__.py
│   │   ├── clean_data.py               # ⭐ STEP 1: Raw CSV → Parquet
│   │   │   └── Validates OHLC integrity
│   │   │   └── Removes duplicates/nulls
│   │   │   └── Creates clean_parquet
│   │   │
│   │   ├── resample_data.py            # Create multi-timeframe data
│   │   │   └── 1-min → 5-min, 15-min, 1H
│   │   │
│   │   └── fetch_global_data.py        # Fetch macro data
│   │       └── S&P 500, Nasdaq, DXY, Forex
│   │
│   ├── 📁 features/
│   │   ├── __init__.py
│   │   ├── build_features.py           # ⭐ STEP 2: Compute 256 features
│   │   │   ├── Trend Features (~40): EMA, SuperTrend, ADX, MACD
│   │   │   ├── Fibonacci Features (~25): Levels, Pivots, Zones
│   │   │   ├── Candlestick Features (~30): Patterns, Wicks, Body%
│   │   │   ├── Trap Detection (~20): Bull/Bear traps, Fake breakouts
│   │   │   ├── Volume Features (~30): Volume delta, VPOC, VWAP, OBV
│   │   │   ├── Volatility Features (~25): ATR, BB, Keltner, IV
│   │   │   ├── Momentum Features (~40): RSI, Stoch, CCI, Williams %R
│   │   │   ├── Correlation Features (~15): Inter-timeframe correlation
│   │   │   └── Global Macro Features (~30): SPX, NDX, DXY, USD/INR
│   │   │
│   │   ├── correlation_audit.py        # Find feature correlations
│   │   └── [OUTPUT: nifty_features.parquet]
│   │
│   ├── 📁 models/
│   │   ├── __init__.py
│   │   ├── train_models.py             # ⭐ STEP 3: Train all 8 models
│   │   │   ├── Load features from parquet
│   │   │   ├── Split train/holdout (2024-01-01)
│   │   │   ├── Train L1 Base Models (1 hour)
│   │   │   │   ├── Trend CatBoost
│   │   │   │   ├── Fibo XGBoost
│   │   │   │   ├── Candle CatBoost
│   │   │   │   ├── Trap XGBoost
│   │   │   │   └── LightGBM
│   │   │   ├── Train L2 LSTM (30 min) with sequences
│   │   │   ├── Train L2 FinBERT (if sentiment available)
│   │   │   ├── Generate L1 predictions on full dataset
│   │   │   ├── Train L3 Supervisor meta-learner (5 min)
│   │   │   └── Evaluate on holdout & save all models
│   │   │
│   │   ├── hyperopt_config.py          # Hyperparameter tuning config
│   │   ├── final_resync.py             # Sync models after tuning
│   │   └── [OUTPUT: .joblib & .keras files + reports]
│   │
│   ├── 📁 news/
│   │   ├── __init__.py
│   │   ├── news_fetcher.py             # Fetch news from 4 APIs + RSS
│   │   │   └── NewsAPI, GNews, NewsData, MarketAux, RSS feeds
│   │   └── finbert_agent.py            # FinBERT sentiment analysis
│   │       └── Convert news → sentiment scores (-1, 0, +1)
│   │
│   ├── 📁 signals/
│   │   ├── __init__.py
│   │   ├── strike_selector.py          # Select ATM/OTM strikes
│   │   │   ├── Compute entry price
│   │   │   ├── Set stop-loss (SL)
│   │   │   ├── Compute profit targets (1:1, 1:2, 1:3 RR)
│   │   │   └── Validate against Greeks
│   │   │
│   │   └── trade_manager.py            # Manage trade lifecycle
│   │       ├── Lock trades at market open
│   │       ├── Track P&L in real-time
│   │       ├── Close on target/SL/timeout
│   │       └── Log all trades to trade_log.csv
│   │
│   └── 📁 utils/
│       ├── __init__.py
│       └── market_simulator.py         # Dry-run market simulation
│
├── 📁 static/                         # (For dashboard assets if needed)
│
├── 📁 logs/                           # Training & runtime logs
│   └── {YYYY-MM-DD}/
│       └── training.log
│
├── 📄 app.py                          # ⭐ STEP 5: Dash Dashboard
│   ├── Professional TradingView-style UI
│   ├── Candlestick charts + Volume + RSI
│   ├── Model consensus panel
│   ├── Trade signal display
│   ├── News sentiment feed
│   ├── Trade journal table
│   ├── Live port forwarding: http://localhost:8050
│   └── Auto-refresh every 30 seconds
│
├── 📄 live_engine_atis.py             # ⭐ LIVE INFERENCE ENGINE
│   ├── Load all 8 trained models
│   ├── Compute 256 features per new bar
│   ├── Generate L1 → L2 → L3 predictions
│   ├── Confidence scoring
│   ├── Strike selection
│   ├── Angel One API integration (ready to activate)
│   └── < 5 seconds inference latency
│
├── 📄 run_backtest.py                 # ⭐ STEP 4: Backtest on holdout
│   ├── Load all 8 models
│   ├── Run walk-forward validation on 2024-2026 data
│   ├── Simulate strike selection + entry/SL/target validation
│   ├── Compute metrics (accuracy, F1, precision, recall, Sharpe)
│   ├── Generate backtest_report.json
│   └── Validate system before live deployment
│
├── 📄 test_angel_api.py               # Test Angel One API connection
├── 📄 validate_boost.py               # Validate model boost
└── 📄 README.md                       # Project readme

```

---

## 🔄 COMPLETE TRAINING PIPELINE (WHAT NEEDS TO HAPPEN)

### **STEP 1: Data Cleaning (5-10 minutes)**
```bash
python src/data/clean_data.py
```
**Input**: `data/raw/nifty_1min_raw.csv`  
**Process**:
- Parse timestamps
- Remove duplicates/nulls
- Validate OHLC relationship (high ≥ open/close ≥ low)
- Handle gaps

**Output**: `data/processed/nifty_1min_clean.parquet`

---

### **STEP 2: Feature Engineering (5-10 minutes)**
```bash
python src/features/build_features.py
```
**Input**: `data/processed/nifty_1min_clean.parquet`  
**Process**:
- Compute 256 alpha features from raw OHLCV
  - Trend: EMA, SuperTrend, ADX, MACD, RSI, Stoch
  - Fibonacci: Levels, Pivots, Zones
  - Candlesticks: Pattern recognition (doji, hammer, engulfing)
  - Traps: Bull/bear trap detection
  - Volume: VPOC, VWAP, OBV, Volume delta
  - Volatility: ATR, Bollinger Bands, Keltner
  - Global: SPX correlation, DXY, USD/INR
- Create feature_list.txt with ordered names
- Handle NaN with forward-fill and mean imputation

**Output**: `data/processed/nifty_features.parquet` (256 columns)

---

### **STEP 3: Model Training (1-2 HOURS) ⏳**
```bash
python src/models/train_models.py
```

**Input**: `data/processed/nifty_features.parquet`  

**Process**:

#### **Split Data**
- Train: 2016-2023 (80% of data)
- Holdout: 2024-2026 (20% unseen data for validation)

#### **Layer 1 - Base Models (5 parallel jobs, ~1 hour total)**
| Model | Algorithm | Hyperparams | Purpose |
|-------|-----------|------------|---------|
| trend_catboost | CatBoost (1200 iterations) | depth=7, lr=0.03 | Trend strength & direction |
| fibo_xgboost | XGBoost (1200 estimators) | depth=7, lr=0.03 | Support/resistance zones |
| candle_catboost | CatBoost (1000 iterations) | depth=7, lr=0.03 | Candlestick patterns |
| trap_xgboost | XGBoost (1000 estimators) | depth=7, lr=0.03 | False breakout detection |
| lgbm | LightGBM (1200 estimators) | depth=8, lr=0.03 | General ensemble |

**Training**:
- 3-fold cross-validation on train set
- Hyperparameter tuning with Optuna
- Early stopping on holdout validation accuracy

**Output**: `models/saved/{trend,fibo,candle,trap}_*.joblib` + `lgbm.joblib`

#### **Layer 2 - Neural & Sentiment (parallel, ~30 min)**
| Model | Type | Config | Purpose |
|-------|------|--------|---------|
| lstm.keras | LSTM | 128 units, 2 layers, 60-bar sequence | Temporal patterns |
| finbert | FinBERT | Pre-trained | News sentiment (+1, 0, -1) |

**LSTM Training**:
- Reshape features into 60-bar sequences
- StandardScaler normalization
- 50 epochs, batch_size=512, dropout=0.3
- Save scaler as lstm_scaler.joblib
- Output shape: (batch, sequence_length, features)

**Output**: `models/saved/lstm.keras` + `lstm_scaler.joblib`, sentiment scores

#### **Layer 3 - Supervisor Meta-Learner (~5 min)**
- Load all L1+L2 model predictions
- Train LogisticRegression on L1+L2 outputs
- Learns optimal voting weights
- Outputs confidence % for each signal

**Output**: `models/saved/supervisor.joblib` + `supervisor_scaler.joblib`

#### **Evaluation**
- Holdout accuracy, F1, precision, recall for each model
- Save final scores to `models/saved/final_report.json`
- Print summary metrics to console

---

### **STEP 4: Backtesting (10-20 minutes)**
```bash
python run_backtest.py
```

**Input**: All trained models + 2024-2026 holdout data  

**Process**:
- Walk-forward validation on holdout set
- For each bar:
  1. Compute 256 features
  2. Run full 8-model ensemble (L1 → L2 → L3)
  3. Get supervisor confidence score
  4. If confidence > 60% → Generate trade signal
  5. Select strike (ATM/OTM)
  6. Set entry, SL, target prices
  7. Simulate trade: did it hit SL or target?
  8. Log to trade journal

**Output**: `models/saved/backtest_report.json` with:
- Total trades
- Win rate %
- Profit factor
- Max drawdown
- Sharpe ratio
- Average R:R

---

### **STEP 5: Launch Dashboard (Ongoing)**
```bash
python app.py
```

**Input**: Latest models + trade_log.csv + engine_state.json  

**Output**: 
- Live dashboard on `http://localhost:8050`
- TradingView-style candlestick charts
- Model consensus panel
- Trade signals table
- News sentiment feed
- Trade journal
- Real-time refresh every 30 seconds

---

## 📊 KEY METRICS & EXPECTED OUTPUT

After complete training, you should have:

```
ATIS v4.0 - FINAL METRICS
═══════════════════════════════════════════

L1 Base Models (Individual Performance):
  ├─ Trend CatBoost        → Accuracy: ~72%, F1: ~0.68
  ├─ Fibo XGBoost          → Accuracy: ~71%, F1: ~0.67
  ├─ Candle CatBoost       → Accuracy: ~70%, F1: ~0.66
  ├─ Trap XGBoost          → Accuracy: ~69%, F1: ~0.65
  └─ LightGBM              → Accuracy: ~73%, F1: ~0.69

L2 Neural Models:
  ├─ LSTM                  → Accuracy: ~68%, F1: ~0.63
  └─ FinBERT               → (If news available) Sentiment accuracy

L3 Supervisor (Ensemble):
  └─ LogRegression         → Holdout F1: 68-72% ⭐
                             Holdout Accuracy: 76-80% ⭐

Backtest Results (2024-2026 Holdout):
  ├─ Total Trades: ~500-1000
  ├─ Win Rate: 55-65%
  ├─ Profit Factor: 1.2-1.5
  ├─ Sharpe Ratio: 1.0-1.5
  └─ Max Drawdown: 15-25%

Average Inference Latency: < 2 seconds ⚡
Dashboard Response Time: < 100ms
```

---

## 🎯 YOUR OBJECTIVES

### **During Codespaces Training**:
1. ✅ Clean raw CSV data → Parquet
2. ✅ Engineer 256 alpha features
3. ✅ Train 8-model ensemble (patience for 1-2 hours)
4. ✅ Validate on 2024-2026 holdout
5. ✅ Generate backtesting report
6. ✅ Launch interactive dashboard
7. ✅ Push all models to GitHub

### **After Training**:
- Monitor model performance metrics
- Adjust hyperparameters if needed
- Integrate Angel One API for live trading
- Deploy dashboard to production
- Keep models in GitHub for version control

---

## 🔄 MIGRATION STRATEGY (IMPORTANT)

### **Local ↔ Codespaces ↔ Production**

```
┌─────────────────────┐
│   LOCAL MACHINE     │
│ (C:\ATIS_updated)   │
│ ├─ Code             │
│ ├─ Requirements     │
│ └─ .devcontainer/   │
└──────────┬──────────┘
           │ git push
           ▼
┌─────────────────────┐
│   GITHUB REPO       │ ⭐
│ vanshAY2/           │
│ ATIS-Trading-System │
│ ├─ Source code      │
│ ├─ .devcontainer/   │
│ └─ (NO large files) │
└──────────┬──────────┘
           │ Codespaces
           ▼
┌─────────────────────┐
│  CODESPACES ENV     │ 🚀 (YOU ARE HERE)
│ /workspaces/        │
│ ├─ Auto-clone repo  │
│ ├─ Auto-install deps│
│ ├─ Upload CSV here  │
│ ├─ TRAIN MODELS ⏳ │
│ ├─ Generate outputs │
│ └─ Push to GitHub   │
└──────────┬──────────┘
           │ git push
           ▼
┌─────────────────────┐
│   GITHUB ARTIFACTS  │
│ (Releases/Tags)     │
│ ├─ Trained models   │
│ ├─ Reports (JSON)   │
│ └─ Trade journal    │
└─────────────────────┘
           │
           ▼ Deploy
    ┌────────────────┐
    │PRODUCTION ENV  │
    │ (Live Trading)│
    └────────────────┘
```

### **Key Points**:
- **Local machine**: Stores code only (Git repo)
- **GitHub**: Central repository (code + versions)
- **Codespaces**: Training environment (where models are trained)
- **Trained models**: Saved as artifacts/releases on GitHub
- **Production**: Only needs to load pre-trained models (not retrain)

### **What STAYS in Codespaces**:
- Training logs and reports
- Raw CSV (don't push to Git)
- Trained .joblib & .keras files
- Trade journals

### **What GOES to GitHub**:
- Source code (src/, config/)
- Requirements.txt
- Documentation
- .devcontainer/ (for reproducibility)
- (Optional) Tagged releases with model artifacts

---

## ⚙️ ENVIRONMENT DETAILS

```
Environment: GitHub Codespaces
OS: Ubuntu Linux (Debian Bullseye)
Python: 3.11.x
RAM: 16 GB
CPU: 4-core
Storage: 32 GB
GPU: None (but CPU sufficient for training)

Pre-installed (via .devcontainer):
├─ Python 3.11
├─ pip (latest)
├─ All packages from requirements.txt
│  ├─ pandas 3.0+
│  ├─ numpy 1.24+
│  ├─ scikit-learn 1.3+
│  ├─ xgboost 2.0+
│  ├─ catboost 1.2+
│  ├─ lightgbm 4.0+
│  ├─ tensorflow 2.15+
│  ├─ torch 2.1+
│  ├─ transformers 4.36+ (FinBERT)
│  ├─ dash 2.14+ (Dashboard)
│  ├─ plotly 5.18+ (Charts)
│  └─ ... (30+ packages total)
├─ VS Code + Extensions (Python, Pylance, Jupyter, Copilot)
├─ Git + GitHub CLI
└─ Bash shell

Port Forwarding:
├─ Port 8050 → Dash dashboard
└─ Port 8888 → Jupyter notebooks
```

---

## 📋 COMPLETE EXECUTION CHECKLIST

```
☐ Verify CSV is uploaded to data/raw/
☐ Verify Python 3.11 is working
☐ Verify all dependencies installed
☐ Step 1: Run src/data/clean_data.py → Check for nifty_1min_clean.parquet
☐ Step 2: Run src/features/build_features.py → Check for nifty_features.parquet (256 columns)
☐ Step 3: Run src/models/train_models.py → Train all 8 models (1-2 hours ⏳)
        → Check models/saved/ for .joblib and .keras files
        → Check final_report.json for metrics
☐ Step 4: Run run_backtest.py → Generate backtest_report.json
☐ Step 5: Run app.py → Dashboard live on port 8050
☐ Verify dashboard is accessible
☐ Monitor training logs in logs/YYYY-MM-DD/training.log
☐ Once complete: git add . && git commit && git push
☐ Verify all files pushed to GitHub
☐ Archive models as GitHub Release (optional)
```

---

## 🎯 FINAL DELIVERABLES

Once training completes, you will have:

**1. Trained Models**:
- ✅ 5 tree-based models (.joblib)
- ✅ 1 LSTM neural network (.keras)
- ✅ 1 FinBERT sentiment model
- ✅ 1 Supervisor meta-learner (.joblib)

**2. Data Artifacts**:
- ✅ Clean parquet files (multi-timeframe)
- ✅ 256 engineered features
- ✅ Feature selection list

**3. Reports**:
- ✅ Training report (hyperparameters, CV scores)
- ✅ Final report (holdout accuracy, F1)
- ✅ Backtest report (win rate, Sharpe, drawdown)
- ✅ Trade journal (entry/exit/P&L)

**4. Dashboard**:
- ✅ Live interactive dashboard
- ✅ Candlestick charts with indicators
- ✅ Model consensus panel
- ✅ Trade signals in real-time
- ✅ News sentiment feed

**5. Code Ready for Production**:
- ✅ live_engine_atis.py (inference engine)
- ✅ Angel API integration ready
- ✅ All models serialized and loadable
- ✅ Full version control on GitHub

---

## 🚀 NOW YOU'RE READY!

**Your next step**:
1. Upload `nifty_1min_raw.csv` to Codespaces `data/raw/`
2. Run the 5-step pipeline
3. Monitor training progress
4. Access dashboard on port 8050
5. Push final results to GitHub

**Expected timeline**: ~2 hours total (most spent on model training)

**Questions?** I'm here to help debug or optimize any step!
