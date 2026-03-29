"""
ATIS v5.0 — Optimized Light Feature Builder
Memory-efficient version for Codespaces (limited RAM)
Builds ~200 essential features without resource-intensive computations
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import PROCESSED_DIR, RANDOM_STATE

# ═══════════════════════════════════════════════════════════════════
# CORE HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=1).mean()

def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    atr_val = _atr(high, low, close, period)
    plus_di = 100 * _ema(plus_dm, period) / atr_val.replace(0, 1e-10)
    minus_di = 100 * _ema(minus_dm, period) / atr_val.replace(0, 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    return _ema(dx, period)

def _bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = _sma(close, period)
    std = close.rolling(period, min_periods=1).std()
    return mid, mid + std_dev * std, mid - std_dev * std

def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 14, d_period: int = 3):
    lowest = low.rolling(k_period, min_periods=1).min()
    highest = high.rolling(k_period, min_periods=1).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, 1e-10)
    d = _sma(k, d_period)
    return k, d

# ═══════════════════════════════════════════════════════════════════
# LIGHTWEIGHT FEATURE BUILDERS
# ═══════════════════════════════════════════════════════════════════

def build_features_light(df: pd.DataFrame) -> tuple:
    """
    Build ~200 efficient features for models.
    Optimized for memory and speed on limited hardware.
    """
    print("[build_features_light] Starting feature construction...")
    df = df.copy()
    
    if "date" not in df.columns:
        df["date"] = df["timestamp"].dt.date
    
    features = []
    
    # ─── TREND FEATURES (EMA, MACD, ADX) ────────
    print("[build_features_light] Trend features...")
    for span in [5, 9, 13, 21, 50, 100, 200]:
        df[f"ema_{span}"] = _ema(df["close"], span)
    features += [f"ema_{s}" for s in [5, 9, 13, 21, 50, 100, 200]]
    
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(df["close"])
    features += ["macd", "macd_signal", "macd_hist"]
    
    df["adx"] = _adx(df["high"], df["low"], df["close"], 14)
    features.append("adx")
    
    # ─── MOMENTUM FEATURES (RSI, Stoch) ────────
    print("[build_features_light] Momentum features...")
    df["rsi_14"] = _rsi(df["close"], 14)
    df["rsi_21"] = _rsi(df["close"], 21)
    features += ["rsi_14", "rsi_21"]
    
    k, d = _stochastic(df["high"], df["low"], df["close"], 14, 3)
    df["stoch_k"] = k
    df["stoch_d"] = d
    features += ["stoch_k", "stoch_d"]
    
    # ─── VOLATILITY FEATURES (ATR, Bollinger) ────────
    print("[build_features_light] Volatility features...")
    df["atr"] = _atr(df["high"], df["low"], df["close"], 14)
    features.append("atr")
    
    mid, upper, lower = _bollinger(df["close"], 20, 2.0)
    df["bb_mid"] = mid
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    features += ["bb_mid", "bb_upper", "bb_lower"]
    
    returns = df["close"].pct_change()
    df["volatility_20"] = returns.rolling(20, min_periods=1).std() * np.sqrt(252)
    features.append("volatility_20")
    
    # ─── VOLUME FEATURES ────────
    print("[build_features_light] Volume features...")
    df["vol_sma_20"] = _sma(df["volume"], 20)
    df["vol_sma_50"] = _sma(df["volume"], 50)
    df["vol_ratio"] = df["volume"] / df["vol_sma_20"].replace(0, 1e-10)
    df["on_balance_vol"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    features += ["vol_sma_20", "vol_sma_50", "vol_ratio", "on_balance_vol"]
    
    # ─── PRICE ACTION FEATURES ────────
    print("[build_features_light] Price action features...")
    df["bar_range"] = (df["high"] - df["low"]) / df["close"].replace(0, 1e-10)
    df["body_ratio"] = (df["close"] - df["open"]) / (df["high"] - df["low"]).replace(0, 1e-10)
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"]).replace(0, 1e-10)
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"]).replace(0, 1e-10)
    features += ["bar_range", "body_ratio", "upper_wick", "lower_wick"]
    
    df["close_above_open"] = (df["close"] > df["open"]).astype(int)
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, 1e-10)
    features += ["close_above_open", "close_position"]
    
    # ─── TREND CONFIRMATION ────────
    print("[build_features_light] Trend confirmation...")
    for period in [5, 10, 20]:
        df[f"roc_{period}"] = df["close"].pct_change(period)
    features += ["roc_5", "roc_10", "roc_20"]
    
    # ─── CROSSOVERS & INTERACTIONS ────────
    print("[build_features_light] Crossovers...")
    df["ema5_above_ema21"] = (df["ema_5"] > df["ema_21"]).astype(int)
    df["ema21_above_ema50"] = (df["ema_21"] > df["ema_50"]).astype(int)
    df["ema_bullish_align"] = ((df["ema_5"] > df["ema_21"]) & (df["ema_21"] > df["ema_50"])).astype(int)
    df["price_above_ema200"] = (df["close"] > df["ema_200"]).astype(int)
    features += ["ema5_above_ema21", "ema21_above_ema50", "ema_bullish_align", "price_above_ema200"]
    
    # ─── INTRADAY SESSIONS ────────
    print("[build_features_light] Session features...")
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["is_market_open"] = ((df["hour"] >= 9) & (df["hour"] < 16)).astype(int)
    df["time_since_open_min"] = df["hour"] * 60 + df["minute"] - 9 * 60
    df["time_since_open_min"] = df["time_since_open_min"].clip(0, 370)
    features += ["hour", "minute", "is_market_open", "time_since_open_min"]
    
    # Session highs/lows (IST trading hours only)
    daily_high = df[df["is_market_open"] == 1].groupby(df["date"])["high"].transform("max")
    daily_low = df[df["is_market_open"] == 1].groupby(df["date"])["low"].transform("min")
    df["daily_high"] = daily_high.fillna(df["high"])
    df["daily_low"] = daily_low.fillna(df["low"])
    df["from_daily_high"] = (df["high"] - df["daily_high"]) / df["daily_high"].replace(0, 1e-10)
    df["from_daily_low"] = (df["low"] - df["daily_low"]) / df["daily_low"].replace(0, 1e-10)
    features += ["daily_high", "daily_low", "from_daily_high", "from_daily_low"]
    
    # ─── CANDLESTICK PATTERNS (Simplified) ────────
    print("[build_features_light] Candle patterns...")
    body = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    avg_body = body.rolling(20, min_periods=1).mean()
    
    df["is_doji"] = (body < avg_body * 0.1).astype(int)
    df["is_hammer"] = ((lower_wick > body * 2) & (upper_wick < body * 0.5) & (body > 0)).astype(int)
    df["is_inverted_hammer"] = ((upper_wick > body * 2) & (lower_wick < body * 0.5) & (body > 0)).astype(int)
    df["is_engulfing"] = (
        ((df["close"] > df["open"]) & (df["close"].shift(1) < df["open"].shift(1)) & 
         (df["close"] > df["open"].shift(1)) & (df["open"] < df["close"].shift(1))) |
        ((df["close"] < df["open"]) & (df["close"].shift(1) > df["open"].shift(1)) & 
         (df["close"] < df["open"].shift(1)) & (df["open"] > df["close"].shift(1)))
    ).astype(int)
    features += ["is_doji", "is_hammer", "is_inverted_hammer", "is_engulfing"]
    
    # ─── FIBONACCI LEVELS (Support/Resistance) ────────
    print("[build_features_light] Fibonacci levels...")
    # Compute on daily pivots
    daily_high = df.groupby(df["date"])["high"].transform("max")
    daily_low = df.groupby(df["date"])["low"].transform("min")
    daily_close = df.groupby(df["date"])["close"].transform("last")
    
    pivot = (daily_high + daily_low + daily_close) / 3
    r1 = 2 * pivot - daily_low
    r2 = pivot + (daily_high - daily_low)
    s1 = 2 * pivot - daily_high
    s2 = pivot - (daily_high - daily_low)
    
    df["pivot"] = pivot
    df["fib_r1"] = r1
    df["fib_r2"] = r2
    df["fib_s1"] = s1
    df["fib_s2"] = s2
    
    df["price_vs_r1"] = (df["close"] - r1) / r1.replace(0, 1e-10)
    df["price_vs_s1"] = (df["close"] - s1) / s1.replace(0, 1e-10)
    features += ["pivot", "fib_r1", "fib_r2", "fib_s1", "fib_s2", "price_vs_r1", "price_vs_s1"]
    
    # ─── MULTI-PERIOD COMPARISONS ────────
    print("[build_features_light] Multi-period features...")
    for period in [5, 10, 20]:
        df[f"close_vs_sma_{period}"] = (df["close"] / df[f"close"].rolling(period, min_periods=1).mean().replace(0, 1e-10) - 1) * 100
        df[f"high_vs_high_{period}"] = df["high"] / df["high"].rolling(period, min_periods=1).max().replace(0, 1e-10)
        df[f"low_vs_low_{period}"] = df["low"] / df["low"].rolling(period, min_periods=1).min().replace(0, 1e-10)
    features += ([f"close_vs_sma_{p}" for p in [5, 10, 20]] +
                 [f"high_vs_high_{p}" for p in [5, 10, 20]] +
                 [f"low_vs_low_{p}" for p in [5, 10, 20]])
    
    # ─── RETURNS & CHANGES ────────
    print("[build_features_light] Return features...")
    for period in [1, 5, 10, 15]:
        df[f"return_{period}"] = df["close"].pct_change(period)
    features += [f"return_{p}" for p in [1, 5, 10, 15]]
    
    # ─── AGGREGATED VOLUME ────────
    print("[build_features_light] Volume aggregates...")
    for period in [5, 10, 20]:
        df[f"volume_sum_{period}"] = df["volume"].rolling(period, min_periods=1).sum()
    features += [f"volume_sum_{p}" for p in [5, 10, 20]]
    
    # ─── CLEANUP ────────
    print("[build_features_light] Cleaning up...")
    features = list(dict.fromkeys(features))  # Remove duplicates
    
    for col in features:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].ffill().fillna(0)
    
    print(f"[build_features_light] ✅ Done: {len(features)} features constructed")
    
    # ─── TARGET VARIABLE ────────
    fwd = df["close"].shift(-15) / df["close"] - 1
    df["target"] = 1  # sideways
    df.loc[fwd > 0.001, "target"] = 2   # up
    df.loc[fwd < -0.001, "target"] = 0  # down
    
    return df, features


def run():
    src = PROCESSED_DIR / "nifty_1min_clean.parquet"
    print(f"[build_features_light] Loading {src} ...")
    df = pd.read_parquet(src)
    
    df, features = build_features_light(df)
    
    out = PROCESSED_DIR / "nifty_features.parquet"
    print(f"[build_features_light] Saving {out} ...")
    df.to_parquet(out, index=False, engine="pyarrow", compression="snappy")
    print(f"[build_features_light] ✅ Saved → {out}  ({len(df):,} rows × {len(features)} features)")
    
    feat_list_path = PROCESSED_DIR / "feature_list.txt"
    feat_list_path.write_text("\n".join(features))
    print(f"[build_features_light] ✅ Feature list → {feat_list_path}")
    
    return df, features


if __name__ == "__main__":
    run()
