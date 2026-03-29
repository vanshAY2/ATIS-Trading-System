"""
ATIS v5.0 — Central Configuration (Alpha Boost)
All paths, API credentials, model hyperparameters, and thresholds.
"""
import os
from pathlib import Path

# ─── Project Root ───────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TRADES_DIR = DATA_DIR / "trades"
MODELS_DIR = ROOT_DIR / "models" / "saved"
STATIC_DIR = ROOT_DIR / "static"

for d in [RAW_DIR, PROCESSED_DIR, TRADES_DIR, MODELS_DIR, STATIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_CSV = RAW_DIR / "nifty_1min_raw.csv"

# ─── Angel One API ──────────────────────────────────────────────────
ANGEL_API_KEY      = "jbTc0qFF"
ANGEL_CLIENT_ID    = "R340624"
ANGEL_PASSWORD     = "5415"
ANGEL_TOTP_SECRET  = "3RJ7DRO6OOQNY65AXOVQRWFRBY"

# ─── Alpha Vantage (Global Macro Historical Data) ───────────────────
ALPHA_VANTAGE_KEY  = "D65V8SYN1OA3CXH8"

# ─── Finnhub (Live Streaming Data) ─────────────────────────────────
FINNHUB_KEY        = "d71dl0hr01qot5jd1ds0d71dl0hr01qot5jd1dsg"

# ─── News APIs ──────────────────────────────────────────────────────
NEWSAPI_KEY        = ""
GNEWS_API_KEY      = ""
NEWSDATA_API_KEY   = ""
MARKETAUX_API_KEY  = ""

# ─── RSS Feeds ──────────────────────────────────────────────────────
RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.livemint.com/rss/markets",
]

# ─── FinBERT High-Impact Keywords ───────────────────────────────────
HIGH_IMPACT_KEYWORDS = [
    "RBI", "Fed", "rate cut", "rate hike", "war", "crash",
    "sanctions", "GDP", "inflation", "recession", "default",
    "SEBI", "circuit breaker", "black swan", "emergency",
    "nuclear", "election", "budget", "tariff", "pandemic",
]

# ─── Global Macro Symbols (Alpha Vantage) ───────────────────────────
GLOBAL_SYMBOLS = {
    "SPX":    "SPY",        # S&P 500 ETF proxy
    "NDX":    "QQQ",        # Nasdaq 100 ETF proxy
    "DXY":    "UUP",        # Dollar Index ETF proxy
    "USDINR": "USD/INR",    # Forex pair (FX_DAILY endpoint)
}

# ─── Timeframes ─────────────────────────────────────────────────────
TIMEFRAMES = {
    "1min":  "1min",
    "5min":  "5min",
    "15min": "15min",
    "1H":    "1h",
}

# ─── Strike Price Config ────────────────────────────────────────────
NIFTY_LOT_SIZE     = 25
NIFTY_TICK_SIZE    = 0.05
STRIKE_INTERVAL    = 50
SL_PERCENT         = 0.30
TARGET_RR_RATIOS   = [1.0, 2.0, 3.0]
ENTRY_TOLERANCE    = 0.05
CONFIDENCE_GATE    = 0.60

# ─── Model Hyperparameters (v5.0 Alpha Boost) ──────────────────────
MODEL_PARAMS = {
    "trend_catboost": {
        "iterations": 1200, "depth": 7, "learning_rate": 0.03,
        "l2_leaf_reg": 5, "verbose": 0, "thread_count": -1,
        "auto_class_weights": "Balanced",
    },
    "fibo_xgboost": {
        "n_estimators": 1200, "max_depth": 7, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "tree_method": "hist", "verbosity": 0, "n_jobs": -1,
    },
    "candle_catboost": {
        "iterations": 1000, "depth": 7, "learning_rate": 0.03,
        "l2_leaf_reg": 5, "verbose": 0, "thread_count": -1,
        "auto_class_weights": "Balanced",
    },
    "trap_xgboost": {
        "n_estimators": 1000, "max_depth": 7, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "tree_method": "hist", "verbosity": 0, "n_jobs": -1,
    },
    "lgbm": {
        "n_estimators": 1200, "max_depth": 8, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "verbose": -1, "n_jobs": -1, "class_weight": "balanced",
    },
    "lstm": {
        "units": 128, "layers": 2, "dropout": 0.3,
        "sequence_length": 60, "epochs": 50, "batch_size": 512,
        "attention_heads": 4,
    },
    "supervisor": {
        "C": 1.0, "max_iter": 1000, "class_weight": "balanced",
    },
}

# ─── Training Config ────────────────────────────────────────────────
HOLDOUT_START = "2024-01-01"
MAX_TUNING_ITERATIONS = 10
TARGET_ACCURACY = 0.72    # v5.0: raised from 0.70
RANDOM_STATE = 42

# ─── Global Features (protected from RFE if corr > threshold) ──────
GLOBAL_FEATURE_PREFIX = ["gap_", "forex_", "us_vix_", "dxy_", "spx_", "ndx_", "global_"]
GLOBAL_RFE_CORR_THRESHOLD = 0.40

# ─── Dashboard ──────────────────────────────────────────────────────
DASH_HOST = "127.0.0.1"
DASH_PORT = 8050
DASH_DEBUG = True
