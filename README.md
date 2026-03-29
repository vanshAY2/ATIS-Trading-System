# ATIS v5.0 — Multi-Agent NIFTY 50 Options Intelligence System

**A sophisticated multi-agent architecture for real-time market analysis, model training, safety validation, and code deployment.**

---

## 🏗️ Project Architecture

### Phase 1: Multi-Agent Foundation ✅ (COMPLETE)

Directory structure:

```
ATIS-Trading-System/
├── config/
│   ├── __init__.py
│   └── settings.py              # APIs, hyperparameters, global symbols
│
├── src/
│   ├── __init__.py
│   ├── agents/                  # Intelligent agent modules
│   │   ├── __init__.py
│   │   ├── orchestrator.py     # Main controller
│   │   ├── global_observer.py  # Market data & alignment
│   │   ├── planner.py          # (Stub for Phase 2)
│   │   ├── coder.py            # (Stub for Phase 2)
│   │   ├── guardian.py         # (Stub for Phase 2)
│   │   └── tester.py           # (Stub for Phase 2)
│   │
│   ├── data/                    # Data layer
│   │   ├── __init__.py
│   │   ├── fetchers.py         # (Stub) Fetch SPY, QQQ, USD/INR, VIX
│   │   ├── cleaners.py         # (Stub) Data validation
│   │   └── feature_engine.py   # (Stub) Alpha feature generation
│   │
│   ├── models/                  # Model training & inference
│   │   ├── __init__.py
│   │   ├── l1_base.py          # (Stub) L1 base model training
│   │   ├── l2_neural.py        # (Stub) L2 LSTM training
│   │   └── l3_supervisor.py    # (Stub) L3 meta-learner
│   │
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── safety_config.py    # Guardian Protocol ✅
│       ├── circuit_breaker.py  # (Stub) VIX/SL enforcement
│       └── backtest.py         # (Stub) Walk-forward validation
│
├── dev_state.json              # Real-time development state ✅
├── agent_monitor.py            # Phase 2 monitoring script ✅
├── requirements.txt            # Dependencies ✅
└── README.md                   # This file
```

---

## 🎯 Phase 1: Project Structure Creation

### ✅ Completed

1. **Directory Skeleton**
   - `/src/agents/` — 5 agent modules (Orchestrator, GlobalObserver, Planner, Coder, Guardian, Tester)
   - `/src/data/` — Data fetching, cleaning, feature engineering stubs
   - `/src/models/` — L1-L3 model training stubs
   - `/src/utils/` — Safety config, circuit breakers, backtesting

2. **Guardian Protocol** (`src/utils/safety_config.py`)
   - ✅ F1 score baseline comparison logic
   - ✅ Global VIX circuit breaker (VIX > 25 → NO TRADE)
   - ✅ 30% stop loss enforcement (from settings.py)
   - ✅ Trade authorization gate
   - ✅ US gap analysis (-1% overnight drop = bearish signal)
   - ✅ Forex strength check (USD/INR > 5-day EMA = caution)

3. **Global Observer Agent** (`src/agents/global_observer.py`)
   - ✅ US Market close (SPY/QQQ) ↔ NIFTY open (9:15 IST) alignment
   - ✅ Overnight gap detection (if SPY > -1%, signals)
   - ✅ USD/INR forex correlation to 5-day EMA
   - ✅ Composite global signal generator
   - ✅ Market time synchronization logic

4. **Development State Tracking** (`dev_state.json`)
   - ✅ Real-time agent status (idle/active)
   - ✅ Task progress (0-100%)
   - ✅ Baseline vs current F1 metrics
   - ✅ Global VIX, US Gap, Forex levels
   - ✅ Circuit breaker status

---

## 🚀 How to Run

### Phase 1: Initialize Foundation

```bash
# Run orchestrator to initialize Phase 1
python src/agents/orchestrator.py
```

**Output:**
```
🚀 ATIS v5.0 — PHASE 1: MULTI-AGENT FOUNDATION
============================================================
🤖 [Planner] Creating execution plan
   ✅ Plan created with 6 tasks
🤖 [GlobalObserver] Fetching global market data
   ✅ Market data synced
🤖 [Guardian] Running safety checks
   ✅ Safety checks passed: 4/4
🤖 [Coder] Reviewing model architecture
   ✅ Models ready for training
🤖 [Tester] Running system tests
   ✅ All system tests passed

✨ Phase 1 Complete! Ready for Phase 2: Real-Time Monitoring
```

### Phase 2: Real-Time Monitoring

In one terminal:

```bash
# Initialize Phase 1
python src/agents/orchestrator.py
```

In another terminal:

```bash
# Watch agent status live
python agent_monitor.py
```

**Live Dashboard Output:**

```
════════════════════════════════════════════════════════════
ATIS v5.0 — MULTI-AGENT DEVELOPMENT MONITOR
════════════════════════════════════════════════════════════

AGENT STATUS:
Orchestrator    [ACTIVE] [100%] Initializing Phase 1
Planner         [ACTIVE] [100%] Plan created
GlobalObserver  [ACTIVE] [100%] Global market analysis complete
Guardian        [ACTIVE] [100%] Guardian checks complete
Coder           [ACTIVE] [100%] Model training preparation
Tester          [ACTIVE] [100%] Tests complete - system ready

REAL-TIME METRICS:
  Active Agent: Tester
  Current Task: Tests complete - system ready
  Phase: 2

  PERFORMANCE:
    Baseline F1: 0.6500
    Current F1:  0.0000

  GLOBAL SIGNALS:
    Data Alignment: synced
    Global VIX:     N/A
    US Gap:         N/A
    Forex Strength: N/A

  SAFETY:
    Circuit Breaker: ENABLED
    VIX Threshold:   25
    SL Max:          30.0%
```

---

## 🛡️ Guardian Protocol Details

### F1 Score Baseline Comparison

```python
from src.utils.safety_config import GuardianProtocol

guardian = GuardianProtocol(baseline_f1=0.65)

# Check if model improved over baseline
approved, msg = guardian.check_f1_improvement("trend_catboost", 0.68)
# Output: ✅ trend_catboost: F1 0.6800 >= baseline 0.6500 (+4.6%)
```

### VIX Circuit Breaker

```python
allowed, msg = guardian.check_vix_circuit(vix_value=28.5)
# Output: 🛑 VIX CIRCUIT BREAKER TRIGGERED: VIX 28.50 > 25
# Result: Trading halted
```

### Stop Loss Enforcement

```python
valid, msg = guardian.validate_sl(entry_price=100, sl_price=71)
# Output: ✅ SL VALID: 29.00% <= 30.0%
```

### Trade Authorization Gate

```python
checks = {
    'f1_approved': True,      # F1 >= baseline
    'vix_ok': False,          # VIX <= 25
    'sl_valid': True,         # SL <= 30%
    'no_us_gap': True         # US not down >1%
}

approved, reason = guardian.authorize_trade(checks)
# Output: ❌ TRADE REJECTED: vix_ok
```

---

## 🌍 Global Observer Enhancements

### US Gap Analysis

```python
from src.agents.global_observer import GlobalObserver
import pandas as pd

observer = GlobalObserver()

spy_data = pd.DataFrame({
    'timestamp': [...],
    'open': [...],
    'close': [...]
})

gap_info = observer.analyze_us_gap(spy_data)
# Output: {
#     'gap_percent': -1.5,
#     'signal': 'bearish',          # SPY down >1%
#     'pressure': 'bearish_pressure',
#     'recommendation': 'Apply bearish bias to NIFTY predictions'
# }
```

### Forex Strength Check

```python
usdinr_data = pd.DataFrame({
    'timestamp': [...],
    'close': [...]  # at least 5 rows
})

forex_info = observer.analyze_forex_strength(usdinr_data)
# Output: {
#     'usdinr_current': 82.45,
#     'usdinr_ema5': 82.15,
#     'signal': 'strong',                    # Above EMA5
#     'caution': True,                       # ⚠️ CAUTION for bulls
#     'interpretation': 'USD/INR is strong. ⚠️ CAUTION for NIFTY bulls'
# }
```

### Composite Global Signal

```python
global_signal = observer.generate_global_signal(
    spy_df=spy_data,
    usdinr_df=usdinr_data,
    vix_value=22.5
)

# Output: {
#     'global_bias': 'bearish',
#     'confidence': 0.67,
#     'score': -1.0,
#     'trading_allowed': True,
#     'factors': {
#         'us_gap': {...},
#         'forex': {...},
#         'vix': {...}
#     }
# }
```

---

## 📊 Real-Time Development State

The `dev_state.json` file tracks:

- **Active Agent**: Which agent is currently executing
- **Agent Status**: Individual status for each agent (idle/active)
- **Progress**: Task completion percentage (0-100)
- **Metrics**: F1 scores, data alignment, market signals
- **Circuit Breakers**: Safety gate status and thresholds

Example structure:

```json
{
  "active_agent": "GlobalObserver",
  "current_task": "Analyzing US gaps & forex",
  "phase": 1,
  "timestamp": "2026-03-29T11:00:00Z",
  "agents": {
    "Planner": {"status": "idle", "last_task": "Plan created", "progress": 100},
    "GlobalObserver": {"status": "active", "last_task": "Analyzing US gaps & forex", "progress": 70}
  },
  "metrics": {
    "baseline_f1": 0.65,
    "current_f1": 0.0,
    "data_alignment": "synced",
    "global_vix": 22.5,
    "us_gap": -1.2,
    "forex_strength": 82.45
  }
}
```

---

## 📋 Next Steps (Phase 2+)

- [ ] Implement data fetchers for SPY, QQQ, USD/INR, VIX
- [ ] Data cleaning and validation
- [ ] Feature engineering (Alpha generation)
- [ ] L1-L3 model training pipeline
- [ ] Backtesting with walk-forward validation
- [ ] Live trading interface
- [ ] Dashboard  visualization

---

## 📞 Support

For questions about the multi-agent architecture or Guardian Protocol, check the source code:
- `src/agents/orchestrator.py` — Main orchestrator
- `src/agents/global_observer.py` — Market analysis
- `src/utils/safety_config.py` — Safety gates
