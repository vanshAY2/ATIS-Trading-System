"""
ATIS v5.0 — Alpha Boost Feature Builder
Fixes data leakage, adds volatility-normalized, Heikin-Ashi, gap-adjusted RSI,
multi-timeframe, global macro, and VIX regime interaction features.

v4.0: 256 features (166 legacy + 90 alpha)  →  v5.0: ~300 features (no pads, no leakage)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import PROCESSED_DIR, RANDOM_STATE

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (unchanged from v4.0)
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

def _vwap_synthetic(high, low, close, volume, date):
    typical = (high + low + close) / 3
    cum_tp_vol = (typical * volume).groupby(date).cumsum()
    cum_vol = volume.groupby(date).cumsum()
    return cum_tp_vol / cum_vol.replace(0, 1e-10)


# ═══════════════════════════════════════════════════════════════════
# CANDLESTICK PATTERN DETECTORS (32 patterns)
# ═══════════════════════════════════════════════════════════════════

def _detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l
    avg_body = body.rolling(20, min_periods=1).mean()
    pats = {}
    pats["pat_doji"] = (body < avg_body * 0.1).astype(int)
    pats["pat_spinning_top"] = ((body < avg_body * 0.3) & (upper_shadow > body) & (lower_shadow > body)).astype(int)
    pats["pat_hammer"] = ((lower_shadow > body * 2) & (upper_shadow < body * 0.5) & (body > 0)).astype(int)
    pats["pat_inv_hammer"] = ((upper_shadow > body * 2) & (lower_shadow < body * 0.5) & (body > 0)).astype(int)
    pats["pat_bull_engulf"] = ((c > o) & (c.shift(1) < o.shift(1)) & (c > o.shift(1)) & (o < c.shift(1))).astype(int)
    pats["pat_bear_engulf"] = ((c < o) & (c.shift(1) > o.shift(1)) & (c < o.shift(1)) & (o > c.shift(1))).astype(int)
    small_body_prev = body.shift(1) < avg_body * 0.3
    pats["pat_morning_star"] = ((c.shift(2) < o.shift(2)) & small_body_prev & (c > o) & (c > (o.shift(2) + c.shift(2)) / 2)).astype(int)
    pats["pat_evening_star"] = ((c.shift(2) > o.shift(2)) & small_body_prev & (c < o) & (c < (o.shift(2) + c.shift(2)) / 2)).astype(int)
    bull_candle = c > o
    bear_candle = c < o
    pats["pat_three_white"] = (bull_candle & bull_candle.shift(1) & bull_candle.shift(2) & (c > c.shift(1)) & (c.shift(1) > c.shift(2))).astype(int)
    pats["pat_three_black"] = (bear_candle & bear_candle.shift(1) & bear_candle.shift(2) & (c < c.shift(1)) & (c.shift(1) < c.shift(2))).astype(int)
    pats["pat_piercing"] = ((c.shift(1) < o.shift(1)) & (c > o) & (o < c.shift(1)) & (c > (o.shift(1) + c.shift(1)) / 2)).astype(int)
    pats["pat_dark_cloud"] = ((c.shift(1) > o.shift(1)) & (c < o) & (o > c.shift(1)) & (c < (o.shift(1) + c.shift(1)) / 2)).astype(int)
    pats["pat_hanging_man"] = ((lower_shadow > body * 2) & (upper_shadow < body * 0.3) & (c < o)).astype(int)
    pats["pat_shooting_star"] = ((upper_shadow > body * 2) & (lower_shadow < body * 0.3) & (c < o)).astype(int)
    pats["pat_harami_bull"] = ((c.shift(1) < o.shift(1)) & (c > o) & (o > c.shift(1)) & (c < o.shift(1))).astype(int)
    pats["pat_harami_bear"] = ((c.shift(1) > o.shift(1)) & (c < o) & (o < c.shift(1)) & (c > o.shift(1))).astype(int)
    pats["pat_tweezer_top"] = ((h == h.shift(1)) & (c.shift(1) > o.shift(1)) & (c < o)).astype(int)
    pats["pat_tweezer_bottom"] = ((l == l.shift(1)) & (c.shift(1) < o.shift(1)) & (c > o)).astype(int)
    pats["pat_marubozu_bull"] = ((c > o) & (upper_shadow < body * 0.05) & (lower_shadow < body * 0.05)).astype(int)
    pats["pat_marubozu_bear"] = ((c < o) & (upper_shadow < body * 0.05) & (lower_shadow < body * 0.05)).astype(int)
    pats["pat_inside_bar"] = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)
    pats["pat_outside_bar"] = ((h > h.shift(1)) & (l < l.shift(1))).astype(int)
    pats["pat_dragonfly_doji"] = ((body < avg_body * 0.1) & (lower_shadow > avg_body) & (upper_shadow < avg_body * 0.1)).astype(int)
    pats["pat_gravestone_doji"] = ((body < avg_body * 0.1) & (upper_shadow > avg_body) & (lower_shadow < avg_body * 0.1)).astype(int)
    pats["pat_rising_three"] = (bull_candle & bear_candle.shift(1) & bear_candle.shift(2) & bear_candle.shift(3) & bull_candle.shift(4) & (c > h.shift(4))).astype(int)
    pats["pat_falling_three"] = (bear_candle & bull_candle.shift(1) & bull_candle.shift(2) & bull_candle.shift(3) & bear_candle.shift(4) & (c < l.shift(4))).astype(int)
    pats["pat_belt_hold_bull"] = ((o == l) & (c > o) & (body > avg_body * 1.5)).astype(int)
    pats["pat_belt_hold_bear"] = ((o == h) & (c < o) & (body > avg_body * 1.5)).astype(int)
    pats["pat_kicker_bull"] = ((c.shift(1) < o.shift(1)) & (o > o.shift(1)) & (c > o) & (body > avg_body * 1.5)).astype(int)
    pats["pat_kicker_bear"] = ((c.shift(1) > o.shift(1)) & (o < o.shift(1)) & (c < o) & (body > avg_body * 1.5)).astype(int)
    pats["pat_gap_up"] = (o > h.shift(1)).astype(int)
    pats["pat_gap_down"] = (o < l.shift(1)).astype(int)
    return pd.DataFrame(pats, index=df.index)


# ═══════════════════════════════════════════════════════════════════
# FIBONACCI LEVELS (27)
# ═══════════════════════════════════════════════════════════════════

FIBO_RATIOS = [0.0, 0.114, 0.236, 0.382, 0.5, 0.618, 0.786, 0.886, 1.0,
               1.114, 1.272, 1.414, 1.618, 2.0, 2.272, 2.414, 2.618,
               -0.114, -0.236, -0.382, -0.5, -0.618, -0.786, -0.886,
               -1.272, -1.618, -2.618]

def _fibonacci_levels(high, low, close, date):
    day_high = high.groupby(date).transform("max")
    day_low = low.groupby(date).transform("min")
    swing = day_high - day_low
    fibo = {}
    for i, r in enumerate(FIBO_RATIOS):
        level = day_low + swing * r
        fibo[f"fibo_{i:02d}_dist"] = (close - level) / close.replace(0, 1e-10)
    return pd.DataFrame(fibo, index=close.index)


# ═══════════════════════════════════════════════════════════════════
# SESSION FEATURES (8)
# ═══════════════════════════════════════════════════════════════════

def _session_features(ts: pd.Series) -> pd.DataFrame:
    minutes = ts.dt.hour * 60 + ts.dt.minute
    sess = {}
    sess["sess_preopen"] = ((minutes >= 540) & (minutes < 555)).astype(int)
    sess["sess_open_30"] = ((minutes >= 555) & (minutes < 585)).astype(int)
    sess["sess_morning"] = ((minutes >= 585) & (minutes < 720)).astype(int)
    sess["sess_midday"] = ((minutes >= 720) & (minutes < 840)).astype(int)
    sess["sess_afternoon"] = ((minutes >= 840) & (minutes < 900)).astype(int)
    sess["sess_close_30"] = ((minutes >= 900) & (minutes < 930)).astype(int)
    sess["sess_minute_of_day"] = minutes - 555
    sess["sess_day_of_week"] = ts.dt.dayofweek
    return pd.DataFrame(sess, index=ts.index)


# ═══════════════════════════════════════════════════════════════════
# ALPHA FEATURES (existing from v4.0)
# ═══════════════════════════════════════════════════════════════════

def _volume_delta_features(df):
    alpha = {}
    tick_dir = np.sign(df["close"] - df["close"].shift(1)).fillna(0)
    alpha["vol_delta"] = tick_dir * df["volume"]
    alpha["vol_delta_cum"] = alpha["vol_delta"].cumsum()
    alpha["vol_delta_cum_intra"] = alpha["vol_delta"].groupby(df["date"]).cumsum()
    alpha["vol_delta_5"] = alpha["vol_delta"].rolling(5, min_periods=1).sum()
    alpha["vol_delta_20"] = alpha["vol_delta"].rolling(20, min_periods=1).sum()
    alpha["vol_delta_ratio"] = alpha["vol_delta_5"] / alpha["vol_delta_20"].replace(0, 1e-10)
    alpha["vol_aggressive"] = (alpha["vol_delta"].clip(lower=0)).rolling(20, min_periods=1).sum()
    alpha["vol_passive"] = (-alpha["vol_delta"].clip(upper=0)).rolling(20, min_periods=1).sum()
    alpha["vol_agg_ratio"] = alpha["vol_aggressive"] / alpha["vol_passive"].replace(0, 1e-10)
    alpha["vol_bar_intensity"] = df["volume"] / df["volume"].rolling(50, min_periods=1).mean().replace(0, 1e-10)
    return pd.DataFrame(alpha, index=df.index)

def _vpoc_features_fast(df):
    alpha = {}
    typical = (df["high"] + df["low"] + df["close"]) / 3
    for window in [50, 100, 200]:
        cum_tp_vol = (typical * df["volume"]).rolling(window, min_periods=1).sum()
        cum_vol = df["volume"].rolling(window, min_periods=1).sum()
        vpoc = cum_tp_vol / cum_vol.replace(0, 1e-10)
        alpha[f"vpoc_{window}_dist"] = (df["close"] - vpoc) / df["close"].replace(0, 1e-10)
        alpha[f"vpoc_{window}_above"] = (df["close"] > vpoc).astype(int)
    return pd.DataFrame(alpha, index=df.index)

def _volatility_features(df):
    alpha = {}
    returns = np.log(df["close"] / df["close"].shift(1))
    for w in [20, 50, 100]:
        rv = returns.rolling(w, min_periods=1).std() * np.sqrt(252 * 375)
        alpha[f"realized_vol_{w}"] = rv
    rv_20 = alpha["realized_vol_20"]
    alpha["vix_proxy"] = rv_20
    alpha["vix_proxy_pct"] = rv_20.rolling(252 * 375, min_periods=375).rank(pct=True)
    alpha["vix_regime"] = pd.cut(alpha["vix_proxy_pct"].fillna(0.5),
                                  bins=[0, 0.3, 0.7, 1.0], labels=[0, 1, 2]).astype(float)
    roll_max = rv_20.rolling(252 * 375, min_periods=375).max()
    roll_min = rv_20.rolling(252 * 375, min_periods=375).min()
    alpha["iv_rank_proxy"] = (rv_20 - roll_min) / (roll_max - roll_min).replace(0, 1e-10)
    alpha["vol_ratio_20_50"] = alpha["realized_vol_20"] / alpha["realized_vol_50"].replace(0, 1e-10)
    alpha["parkinson_vol"] = np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(df["high"] / df["low"].replace(0, 1e-10)) ** 2)
    ).rolling(20, min_periods=1).mean()
    return pd.DataFrame(alpha, index=df.index)

def _dte_features(ts):
    alpha = {}
    days_in_month = ts.dt.days_in_month
    alpha["dte_approx"] = (days_in_month - ts.dt.day).clip(0, 30)
    alpha["dte_normalized"] = alpha["dte_approx"] / 30.0
    alpha["dte_squared"] = alpha["dte_normalized"] ** 2
    alpha["dte_bucket"] = pd.cut(alpha["dte_approx"], bins=[-1, 3, 7, 14, 30], labels=[0, 1, 2, 3]).astype(float)
    return pd.DataFrame(alpha, index=ts.index)

def _pcr_proxy_features(df):
    alpha = {}
    ret = df["close"].pct_change()
    momentum_20 = ret.rolling(20, min_periods=1).sum()
    alpha["pcr_proxy"] = 1.0 - (momentum_20 / momentum_20.rolling(100, min_periods=1).std().replace(0, 1e-10)).clip(-2, 2) / 4 + 0.5
    alpha["pcr_proxy_5"] = alpha["pcr_proxy"].rolling(5, min_periods=1).mean()
    alpha["pcr_proxy_delta"] = alpha["pcr_proxy"] - alpha["pcr_proxy_5"]
    return pd.DataFrame(alpha, index=df.index)

def _flow_proxy_features(df):
    alpha = {}
    for days in [5, 10, 20]:
        n_bars = days * 375
        drift = df["close"].pct_change(n_bars)
        alpha[f"flow_proxy_{days}d"] = drift
        alpha[f"flow_proxy_{days}d_z"] = (drift - drift.rolling(n_bars, min_periods=1).mean()) / drift.rolling(n_bars, min_periods=1).std().replace(0, 1e-10)
    return pd.DataFrame(alpha, index=df.index)

def _momentum_features(df):
    alpha = {}
    c = df["close"]
    for periods in [5, 10, 20, 50, 100]:
        alpha[f"roc_{periods}"] = c.pct_change(periods)
    for w in [20, 50, 100]:
        m = c.rolling(w, min_periods=1).mean()
        s = c.rolling(w, min_periods=1).std().replace(0, 1e-10)
        alpha[f"zscore_{w}"] = (c - m) / s
    w52 = 252 * 375
    alpha["dist_52w_high"] = (c - c.rolling(w52, min_periods=375).max()) / c.replace(0, 1e-10)
    alpha["dist_52w_low"] = (c - c.rolling(w52, min_periods=375).min()) / c.replace(0, 1e-10)
    mom_20 = c.pct_change(20)
    alpha["momentum_accel"] = mom_20 - mom_20.shift(20)
    return pd.DataFrame(alpha, index=df.index)

def _cross_features(df):
    alpha = {}
    ema_pairs = [(9, 21), (13, 50), (21, 50), (50, 200)]
    c = df["close"]
    for fast, slow in ema_pairs:
        ema_f = _ema(c, fast)
        ema_s = _ema(c, slow)
        alpha[f"ema_cross_{fast}_{slow}"] = (ema_f > ema_s).astype(int)
        alpha[f"ema_dist_{fast}_{slow}"] = (ema_f - ema_s) / c.replace(0, 1e-10)
    return pd.DataFrame(alpha, index=df.index)


# ═══════════════════════════════════════════════════════════════════
# NEW v5.0: SENTIMENT PROXY (NO LEAKAGE)
# ═══════════════════════════════════════════════════════════════════

def _sentiment_proxy(df):
    """
    Legitimate sentiment proxy using ONLY past data.
    Replaces the leaky _get_offline_sentiment that used future returns.
    Composite of: lagged momentum + volume trend + volatility signal.
    """
    mom_5 = df["close"].pct_change(5).fillna(0)
    mom_20 = df["close"].pct_change(20).fillna(0)
    vol_trend = (df["volume"] / df["volume"].rolling(50, min_periods=1).mean().replace(0, 1e-10) - 1).clip(-1, 1).fillna(0)
    composite = mom_5 * 0.4 + mom_20 * 0.3 + vol_trend * 0.3
    return composite.clip(-1, 1)


# ═══════════════════════════════════════════════════════════════════
# NEW v5.0: HEIKIN-ASHI FEATURES (6)
# ═══════════════════════════════════════════════════════════════════

def _heikin_ashi_features(df):
    """Heikin-Ashi smoothed candles — reduces gap-up noise for Candle Agent."""
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)

    ha_close = (o + h + l + c) / 4.0
    # Vectorized ha_open via exponential smoothing (alpha=0.5)
    ha_close_s = pd.Series(ha_close, index=df.index).shift(1)
    ha_close_s.iloc[0] = (o[0] + c[0]) / 2.0
    ha_open = ha_close_s.ewm(alpha=0.5, adjust=False).mean().values

    ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(l, np.minimum(ha_open, ha_close))
    ha_body = ha_close - ha_open
    ha_trend = pd.Series(np.sign(ha_body), index=df.index).rolling(5, min_periods=1).sum()

    return pd.DataFrame({
        "ha_close": ha_close, "ha_open": ha_open,
        "ha_high": ha_high, "ha_low": ha_low,
        "ha_body": ha_body, "ha_trend": ha_trend.values,
    }, index=df.index)


# ═══════════════════════════════════════════════════════════════════
# NEW v5.0: GAP-ADJUSTED RSI (4)
# ═══════════════════════════════════════════════════════════════════

def _gap_adjusted_rsi(df):
    """RSI on intraday-only movement, ignoring overnight gaps at 9:15 AM."""
    date_col = df["timestamp"].dt.date
    is_first_bar = date_col != date_col.shift(1)
    prev_close = df["close"].shift(1)

    overnight_gap = np.where(is_first_bar, (df["open"] - prev_close) / prev_close.replace(0, 1e-10), 0.0)
    gap_adj = np.where(is_first_bar, df["open"] - prev_close, 0.0)
    cum_gap = pd.Series(gap_adj, index=df.index).cumsum()
    gap_adj_close = df["close"] - cum_gap

    alpha = {}
    alpha["overnight_gap"] = overnight_gap
    alpha["gap_magnitude"] = np.abs(overnight_gap)
    alpha["gap_adj_rsi_14"] = _rsi(gap_adj_close, 14).values
    alpha["gap_adj_rsi_diff"] = alpha["gap_adj_rsi_14"] - _rsi(df["close"], 14).values
    return pd.DataFrame(alpha, index=df.index)


# ═══════════════════════════════════════════════════════════════════
# NEW v5.0: OPENING RANGE BREAKOUT (ORB) FEATURES (4)
# ═══════════════════════════════════════════════════════════════════

def _orb_features(df):
    """Opening Range Breakout: distance from 9:15-9:30 high/low.
    Critical for NIFTY's opening drive phase."""
    date_col = df["timestamp"].dt.date
    minutes = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    is_orb_window = (minutes >= 555) & (minutes < 570)  # 9:15-9:30

    # Compute ORB high/low for each day
    orb_high = df.loc[is_orb_window].groupby(date_col[is_orb_window])["high"].transform("max")
    orb_low = df.loc[is_orb_window].groupby(date_col[is_orb_window])["low"].transform("min")

    # Map ORB levels to all bars of the day
    daily_orb_high = df.loc[is_orb_window].groupby(date_col[is_orb_window])["high"].max()
    daily_orb_low = df.loc[is_orb_window].groupby(date_col[is_orb_window])["low"].min()

    alpha = {}
    mapped_high = date_col.map(daily_orb_high)
    mapped_low = date_col.map(daily_orb_low)
    orb_range = (mapped_high - mapped_low).replace(0, 1e-10)

    alpha["orb_dist_high"] = (df["close"] - mapped_high) / orb_range
    alpha["orb_dist_low"] = (df["close"] - mapped_low) / orb_range
    alpha["orb_breakout_up"] = (df["close"] > mapped_high).astype(int)
    alpha["orb_breakout_down"] = (df["close"] < mapped_low).astype(int)
    return pd.DataFrame(alpha, index=df.index)


# ═══════════════════════════════════════════════════════════════════
# NEW v5.0: GAP-ADJUSTED STOCHASTIC (2)
# ═══════════════════════════════════════════════════════════════════

def _gap_adjusted_stochastic(df):
    """Stochastic oscillator on gap-adjusted prices for session continuity."""
    date_col = df["timestamp"].dt.date
    is_first_bar = date_col != date_col.shift(1)
    gap_adj = np.where(is_first_bar, df["open"] - df["close"].shift(1), 0.0)
    cum_gap = pd.Series(gap_adj, index=df.index).cumsum()

    adj_high = df["high"] - cum_gap
    adj_low = df["low"] - cum_gap
    adj_close = df["close"] - cum_gap

    k, d = _stochastic(adj_high, adj_low, adj_close)
    return pd.DataFrame({
        "gap_adj_stoch_k": k,
        "gap_adj_stoch_d": d,
    }, index=df.index)


# ═══════════════════════════════════════════════════════════════════
# NEW v5.0: VOLATILITY-NORMALIZED FEATURES (4)
# ═══════════════════════════════════════════════════════════════════

def _volatility_normalized_features(df):
    """RSI/VIX, ATR/StdDev, MACD/VIX — regime-aware indicators."""
    vix = df.get("vix_proxy", df["close"].rolling(20, min_periods=1).std()).replace(0, 1e-10)
    price_std = df["close"].rolling(20, min_periods=1).std().replace(0, 1e-10)
    alpha = {}
    alpha["rsi_14_vix_norm"] = df["rsi_14"] / (vix * 100 + 1e-10)
    alpha["atr_price_std"] = df["atr_14"] / price_std
    alpha["macd_vix_norm"] = df["macd"] / vix
    alpha["bb_width_vix_norm"] = df["bb_width"] / vix
    return pd.DataFrame(alpha, index=df.index)


# ═══════════════════════════════════════════════════════════════════
# NEW v5.0: HIGH-VOLUME ZONE (3)
# ═══════════════════════════════════════════════════════════════════

def _high_volume_zone_features(ts):
    """Flags for market open/close sessions with highest real volume."""
    minutes = ts.dt.hour * 60 + ts.dt.minute
    alpha = {}
    alpha["is_high_volume_zone"] = (((minutes >= 555) & (minutes < 600)) | ((minutes >= 885) & (minutes < 930))).astype(int)
    alpha["is_opening_auction"] = ((minutes >= 555) & (minutes < 570)).astype(int)
    alpha["is_closing_auction"] = ((minutes >= 900) & (minutes < 930)).astype(int)
    return pd.DataFrame(alpha, index=ts.index)


# ═══════════════════════════════════════════════════════════════════
# NEW v5.0: VIX REGIME INTERACTION (4)
# ═══════════════════════════════════════════════════════════════════

def _vix_regime_interaction_features(df):
    """VIX regime × momentum/trend/RSI/volume cross-features."""
    regime = df.get("vix_regime", pd.Series(1.0, index=df.index)).fillna(1.0)
    alpha = {}
    alpha["vix_x_momentum"] = regime * df["close"].pct_change(20).fillna(0)
    alpha["vix_x_trend"] = regime * ((df.get("ema_cross_9_21", pd.Series(0, index=df.index))).astype(float) - 0.5) * 2
    alpha["vix_x_rsi"] = regime * (df.get("rsi_14", pd.Series(50, index=df.index)) - 50) / 50
    alpha["vix_x_volume"] = regime * (df["volume"] / df["volume"].rolling(50, min_periods=1).mean().replace(0, 1e-10) - 1).clip(-2, 2).fillna(0)
    return pd.DataFrame(alpha, index=df.index)


# ═══════════════════════════════════════════════════════════════════
# NEW v5.0: MULTI-TIMEFRAME FEATURES (8)
# ═══════════════════════════════════════════════════════════════════

def _multi_timeframe_features(df, processed_dir):
    """Load 15m/1H data, compute features, and forward-fill to 1-min."""
    all_feats = {}
    ts_1min = df["timestamp"]

    for tf_label, tf_file in [("15m", "nifty_15min_clean.parquet"), ("1h", "nifty_1H_clean.parquet")]:
        tf_path = processed_dir / tf_file
        if not tf_path.exists():
            print(f"[build_features]   {tf_label} data not found, skipping MTF features")
            continue

        df_tf = pd.read_parquet(tf_path)
        df_tf["timestamp"] = pd.to_datetime(df_tf["timestamp"])
        # Remove timezone for merge compatibility
        if df_tf["timestamp"].dt.tz is not None:
            df_tf["timestamp"] = df_tf["timestamp"].dt.tz_localize(None)

        # Compute features on higher TF
        ema_f = _ema(df_tf["close"], 9)
        ema_s = _ema(df_tf["close"], 21)
        rsi_val = _rsi(df_tf["close"], 14)
        _, _, macd_h = _macd(df_tf["close"])
        atr_val = _atr(df_tf["high"], df_tf["low"], df_tf["close"], 14)

        tf_feats = pd.DataFrame({
            "timestamp": df_tf["timestamp"],
            f"mtf_{tf_label}_trend": (ema_f > ema_s).astype(int),
            f"mtf_{tf_label}_rsi": rsi_val,
            f"mtf_{tf_label}_macd_sign": np.sign(macd_h),
            f"mtf_{tf_label}_atr": atr_val,
        })

        # Merge to 1-min via merge_asof (backward = use most recent completed bar)
        ts_clean = ts_1min.dt.tz_localize(None) if ts_1min.dt.tz is not None else ts_1min
        df_merge = pd.DataFrame({"timestamp": ts_clean})
        merged = pd.merge_asof(
            df_merge.sort_values("timestamp"),
            tf_feats.sort_values("timestamp"),
            on="timestamp", direction="backward"
        )
        for col in [f"mtf_{tf_label}_trend", f"mtf_{tf_label}_rsi",
                     f"mtf_{tf_label}_macd_sign", f"mtf_{tf_label}_atr"]:
            all_feats[col] = merged[col].values

    return pd.DataFrame(all_feats, index=df.index)


# ═══════════════════════════════════════════════════════════════════
# NEW v5.0: GLOBAL MACRO FEATURES (12)
# ═══════════════════════════════════════════════════════════════════

def _global_macro_features(df, processed_dir):
    """
    Load global daily data (SPY, QQQ, UUP, USD/INR) and create leading indicators.
    US Close Day T → India Open Day T+1 (1 bday shift to prevent look-ahead).
    """
    global_path = processed_dir / "global_daily.parquet"
    if not global_path.exists():
        print("[build_features]   Global data not found — run fetch_global_data.py first")
        return pd.DataFrame(index=df.index)

    gdf = pd.read_parquet(global_path)
    gdf["date"] = pd.to_datetime(gdf["date"])

    # CRITICAL: shift US data by 1 day (US Day T → India Day T+1)
    for col in gdf.columns:
        if col != "date":
            gdf[col] = gdf[col].shift(1)
    gdf = gdf.dropna()

    # Compute daily features on global data
    alpha = {}
    if "SPY_close" in gdf.columns:
        gdf["spx_return"] = gdf["SPY_close"].pct_change()
        gdf["spx_rsi_14"] = _rsi(gdf["SPY_close"], 14)
        gdf["spx_vol_20"] = gdf["spx_return"].rolling(20, min_periods=1).std() * np.sqrt(252)
        gdf["us_vix_level"] = gdf["spx_vol_20"] * 100  # VIX proxy from realized vol
        gdf["us_vix_regime"] = pd.cut(gdf["us_vix_level"].fillna(15), bins=[-1, 15, 25, 100], labels=[0, 1, 2]).astype(float)
        gdf["us_vix_spike"] = (gdf["us_vix_level"].diff().abs() > 3).astype(int)
        alpha_cols = ["spx_return", "spx_rsi_14", "us_vix_level", "us_vix_regime", "us_vix_spike"]
    else:
        alpha_cols = []

    if "QQQ_close" in gdf.columns and "SPY_close" in gdf.columns:
        gdf["ndx_vs_spx"] = gdf["QQQ_close"].pct_change() - gdf["SPY_close"].pct_change()
        alpha_cols.append("ndx_vs_spx")

    if "USDINR_close" in gdf.columns:
        gdf["forex_momentum_5d"] = gdf["USDINR_close"].pct_change(5)
        gdf["forex_momentum_20d"] = gdf["USDINR_close"].pct_change(20)
        alpha_cols += ["forex_momentum_5d", "forex_momentum_20d"]

    if "UUP_close" in gdf.columns:
        gdf["dxy_trend"] = _ema(gdf["UUP_close"], 20).pct_change(5)
        dup_mean = gdf["UUP_close"].rolling(20, min_periods=1).mean()
        dup_std = gdf["UUP_close"].rolling(20, min_periods=1).std().replace(0, 1e-10)
        gdf["dxy_zscore"] = (gdf["UUP_close"] - dup_mean) / dup_std
        alpha_cols += ["dxy_trend", "dxy_zscore"]

    # Gap predictor: US close return vs previous India close return
    if "SPY_close" in gdf.columns:
        gdf["us_overnight_return"] = (gdf["SPY_close"] - gdf["SPY_open"]) / gdf["SPY_open"].replace(0, 1e-10)
        alpha_cols.append("us_overnight_return")

    # Global risk composite
    risk_components = []
    if "us_vix_regime" in gdf.columns:
        risk_components.append(gdf["us_vix_regime"].fillna(1) / 2)
    if "dxy_zscore" in gdf.columns:
        risk_components.append(gdf["dxy_zscore"].clip(-2, 2).fillna(0) / 4 + 0.5)
    if "forex_momentum_20d" in gdf.columns:
        risk_components.append(gdf["forex_momentum_20d"].clip(-0.1, 0.1).fillna(0) * 5 + 0.5)
    if risk_components:
        gdf["global_risk_score"] = sum(risk_components) / len(risk_components)
        alpha_cols.append("global_risk_score")

    if not alpha_cols:
        return pd.DataFrame(index=df.index)

    # Merge to 1-min on date
    gdf_feats = gdf[["date"] + alpha_cols].copy()
    india_date = df["timestamp"].dt.tz_localize(None).dt.date if df["timestamp"].dt.tz is not None else df["timestamp"].dt.date
    date_map = gdf_feats.set_index(gdf_feats["date"].dt.date).drop(columns=["date"])

    result = pd.DataFrame(index=df.index)
    for col in alpha_cols:
        mapped = india_date.map(date_map[col].to_dict() if col in date_map.columns else {})
        result[f"global_{col}" if not col.startswith(("us_", "spx_", "ndx_", "forex_", "dxy_", "global_")) else col] = mapped

    return result.ffill().fillna(0)


# ═══════════════════════════════════════════════════════════════════
# MASTER BUILDER
# ═══════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame, fast_vpoc: bool = True) -> tuple:
    """
    Build v5.0 Alpha Boost feature matrix.
    Returns: (DataFrame with features + target, feature_name_list)
    """
    print("[build_features] Starting v5.0 Alpha Boost feature construction ...")
    df = df.copy()

    if "date" not in df.columns:
        df["date"] = df["timestamp"].dt.date

    features = []

    # ─── LEGACY BLOCK (same as v4.0) ────────────────────────
    print("[build_features]   EMAs ...")
    for span in [5, 9, 13, 21, 50, 100, 200]:
        df[f"ema_{span}"] = _ema(df["close"], span)
    features += [f"ema_{s}" for s in [5, 9, 13, 21, 50, 100, 200]]

    print("[build_features]   RSI ...")
    df["rsi_14"] = _rsi(df["close"], 14)
    df["rsi_21"] = _rsi(df["close"], 21)
    features += ["rsi_14", "rsi_21"]

    print("[build_features]   MACD ...")
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(df["close"])
    features += ["macd", "macd_signal", "macd_hist"]

    print("[build_features]   Stochastic ...")
    df["stoch_k"], df["stoch_d"] = _stochastic(df["high"], df["low"], df["close"])
    features += ["stoch_k", "stoch_d"]

    print("[build_features]   ATR ...")
    for p in [14, 21, 50]:
        df[f"atr_{p}"] = _atr(df["high"], df["low"], df["close"], p)
    features += [f"atr_{p}" for p in [14, 21, 50]]

    print("[build_features]   ADX ...")
    df["adx_14"] = _adx(df["high"], df["low"], df["close"], 14)
    features += ["adx_14"]

    print("[build_features]   Bollinger ...")
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = _bollinger(df["close"])
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, 1e-10)
    features += ["bb_mid", "bb_upper", "bb_lower", "bb_width"]

    print("[build_features]   VWAP ...")
    df["vwap"] = _vwap_synthetic(df["high"], df["low"], df["close"], df["volume"], df["date"])
    df["vwap_dist"] = (df["close"] - df["vwap"]) / df["close"].replace(0, 1e-10)
    features += ["vwap", "vwap_dist"]

    print("[build_features]   Supertrend ...")
    atr_st = _atr(df["high"], df["low"], df["close"], 10)
    hl2 = (df["high"] + df["low"]) / 2
    df["supertrend_upper"] = hl2 + 3 * atr_st
    df["supertrend_lower"] = hl2 - 3 * atr_st
    df["supertrend_dir"] = (df["close"] > df["supertrend_upper"].shift(1)).astype(int) - \
                            (df["close"] < df["supertrend_lower"].shift(1)).astype(int)
    features += ["supertrend_upper", "supertrend_lower", "supertrend_dir"]

    print("[build_features]   Fibonacci levels ...")
    fibo_df = _fibonacci_levels(df["high"], df["low"], df["close"], df["date"])
    df = pd.concat([df, fibo_df], axis=1)
    features += list(fibo_df.columns)

    print("[build_features]   Candlestick patterns ...")
    pat_df = _detect_patterns(df)
    df = pd.concat([df, pat_df], axis=1)
    features += list(pat_df.columns)

    print("[build_features]   Session features ...")
    sess_df = _session_features(df["timestamp"])
    df = pd.concat([df, sess_df], axis=1)
    features += list(sess_df.columns)

    print("[build_features]   EMA crossovers ...")
    cross_df = _cross_features(df)
    df = pd.concat([df, cross_df], axis=1)
    features += list(cross_df.columns)

    df["bar_range"] = (df["high"] - df["low"]) / df["close"].replace(0, 1e-10)
    df["body_ratio"] = (df["close"] - df["open"]) / (df["high"] - df["low"]).replace(0, 1e-10)
    df["upper_wick_ratio"] = (df["high"] - pd.concat([df["open"], df["close"]], axis=1).max(axis=1)) / (df["high"] - df["low"]).replace(0, 1e-10)
    features += ["bar_range", "body_ratio", "upper_wick_ratio"]

    df["vol_sma_20"] = _sma(df["volume"], 20)
    df["vol_sma_50"] = _sma(df["volume"], 50)
    df["vol_ratio"] = df["volume"] / df["vol_sma_20"].replace(0, 1e-10)
    features += ["vol_sma_20", "vol_sma_50", "vol_ratio"]

    # ─── ALPHA BLOCK (same as v4.0) ────────────────────────
    print("[build_features]   Volume delta ...")
    vd_df = _volume_delta_features(df)
    df = pd.concat([df, vd_df], axis=1)
    features += list(vd_df.columns)

    print("[build_features]   VPOC ...")
    vpoc_df = _vpoc_features_fast(df)
    df = pd.concat([df, vpoc_df], axis=1)
    features += list(vpoc_df.columns)

    print("[build_features]   Volatility ...")
    vol_df = _volatility_features(df)
    df = pd.concat([df, vol_df], axis=1)
    features += list(vol_df.columns)

    print("[build_features]   DTE ...")
    dte_df = _dte_features(df["timestamp"])
    df = pd.concat([df, dte_df], axis=1)
    features += list(dte_df.columns)

    print("[build_features]   PCR proxy ...")
    pcr_df = _pcr_proxy_features(df)
    df = pd.concat([df, pcr_df], axis=1)
    features += list(pcr_df.columns)

    print("[build_features]   FII/DII flow proxy ...")
    flow_df = _flow_proxy_features(df)
    df = pd.concat([df, flow_df], axis=1)
    features += list(flow_df.columns)

    print("[build_features]   Momentum ...")
    mom_df = _momentum_features(df)
    df = pd.concat([df, mom_df], axis=1)
    features += list(mom_df.columns)

    # ─── NEW v5.0 ALPHA BOOST FEATURES ─────────────────────
    print("[build_features]   [v5.0] Sentiment proxy (no leakage) ...")
    df["sentiment_score"] = _sentiment_proxy(df)
    features.append("sentiment_score")

    print("[build_features]   [v5.0] Heikin-Ashi ...")
    ha_df = _heikin_ashi_features(df)
    df = pd.concat([df, ha_df], axis=1)
    features += list(ha_df.columns)

    print("[build_features]   [v5.0] Gap-Adjusted RSI ...")
    ga_df = _gap_adjusted_rsi(df)
    df = pd.concat([df, ga_df], axis=1)
    features += list(ga_df.columns)

    print("[build_features]   [v5.0] Gap-Adjusted Stochastic ...")
    gs_df = _gap_adjusted_stochastic(df)
    df = pd.concat([df, gs_df], axis=1)
    features += list(gs_df.columns)

    print("[build_features]   [v5.0] Opening Range Breakout (ORB) ...")
    orb_df = _orb_features(df)
    df = pd.concat([df, orb_df], axis=1)
    features += list(orb_df.columns)

    print("[build_features]   [v5.0] Volatility-Normalized ...")
    vn_df = _volatility_normalized_features(df)
    df = pd.concat([df, vn_df], axis=1)
    features += list(vn_df.columns)

    print("[build_features]   [v5.0] High-Volume Zone ...")
    hvz_df = _high_volume_zone_features(df["timestamp"])
    df = pd.concat([df, hvz_df], axis=1)
    features += list(hvz_df.columns)

    print("[build_features]   [v5.0] VIX Regime Interactions ...")
    vri_df = _vix_regime_interaction_features(df)
    df = pd.concat([df, vri_df], axis=1)
    features += list(vri_df.columns)

    print("[build_features]   [v5.0] Multi-Timeframe (15m/1H) ...")
    mtf_df = _multi_timeframe_features(df, PROCESSED_DIR)
    if not mtf_df.empty:
        df = pd.concat([df, mtf_df], axis=1)
        features += list(mtf_df.columns)

    print("[build_features]   [v5.0] Global Macro ...")
    gm_df = _global_macro_features(df, PROCESSED_DIR)
    if not gm_df.empty:
        df = pd.concat([df, gm_df], axis=1)
        features += list(gm_df.columns)

    # ─── CLEAN UP (NO MORE PADDING) ────────────────────────
    features = list(dict.fromkeys(features))

    for col in features:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].ffill().fillna(0)

    n_feats = len(features)
    print(f"[build_features] Done: {n_feats} features constructed (v5.0 — no zero pads).")

    # ─── TARGET VARIABLE ────────────────────────────────────
    fwd = df["close"].shift(-15) / df["close"] - 1
    df["target"] = 1  # sideways
    df.loc[fwd > 0.001, "target"] = 2   # up
    df.loc[fwd < -0.001, "target"] = 0  # down

    return df, features


def run():
    src = PROCESSED_DIR / "nifty_1min_clean.parquet"
    print(f"[build_features] Loading {src} ...")
    df = pd.read_parquet(src)
    df, features = build_features(df)

    out = PROCESSED_DIR / "nifty_features.parquet"
    df.to_parquet(out, index=False, engine="pyarrow")
    print(f"[build_features] Saved → {out}  ({len(df):,} rows × {len(features)} features)")

    feat_list_path = PROCESSED_DIR / "feature_list.txt"
    feat_list_path.write_text("\n".join(features))
    print(f"[build_features] Feature list → {feat_list_path}")
    return df, features


if __name__ == "__main__":
    run()
