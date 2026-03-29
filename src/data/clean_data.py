"""
ATIS v4.0 — Data Cleaning Pipeline
Parses raw 1-min CSV, handles timezone, fills gaps, removes noise.
Output: data/processed/nifty_1min_clean.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as script or module
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import RAW_CSV, PROCESSED_DIR


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load raw CSV and parse timestamps."""
    print(f"[clean_data] Loading {path} ...")
    df = pd.read_csv(
        path,
        parse_dates=["timestamp"],
        dtype={"open": "float64", "high": "float64",
               "low": "float64", "close": "float64", "volume": "float64"},
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"[clean_data]   Loaded {len(df):,} rows  "
          f"[{df['timestamp'].min()} → {df['timestamp'].max()}]")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Core cleaning logic."""
    initial = len(df)

    # 1. Drop exact duplicate timestamps
    df = df.drop_duplicates(subset="timestamp", keep="first")

    # 2. Filter to market hours only (09:15 – 15:29 IST)
    t = df["timestamp"]
    market_mask = (t.dt.hour * 60 + t.dt.minute >= 9 * 60 + 15) & \
                  (t.dt.hour * 60 + t.dt.minute <= 15 * 60 + 29)
    df = df[market_mask].copy()

    # 3. Remove weekends
    df = df[df["timestamp"].dt.dayofweek < 5].copy()

    # 4. Sanity: high >= low, all prices > 0
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0)].copy()

    # 5. Forward-fill small price gaps (max 5 consecutive NaN)
    df = df.set_index("timestamp")
    df = df.asfreq("min")  # explicit 1-min frequency
    # only fill within market hours
    mkt = (df.index.hour * 60 + df.index.minute >= 9 * 60 + 15) & \
          (df.index.hour * 60 + df.index.minute <= 15 * 60 + 29) & \
          (df.index.dayofweek < 5)
    df = df[mkt]
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill(limit=5)
    df["volume"] = df["volume"].fillna(0)
    df = df.dropna(subset=["close"])
    df = df.reset_index()

    final = len(df)
    print(f"[clean_data]   Cleaned: {initial:,} → {final:,} rows  "
          f"(dropped {initial - final:,})")
    return df


def synthesize_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesize futures-style volume from price action using tick-rule heuristic.
    Volume proxy = |close - open| * (1 + normalized range) * base_multiplier
    """
    body = (df["close"] - df["open"]).abs()
    rng = df["high"] - df["low"]
    avg_rng = rng.rolling(100, min_periods=1).mean()
    norm_rng = (rng / avg_rng.replace(0, 1)).clip(0.1, 10)

    base = 5000  # base contracts per bar
    df["volume"] = (body / df["close"].rolling(100, min_periods=1).mean().replace(0, 1)
                    * norm_rng * base * 1000).round().astype("int64")
    df["volume"] = df["volume"].clip(lower=100)
    print(f"[clean_data]   Synthesized volume: mean={df['volume'].mean():.0f}, "
          f"max={df['volume'].max():,}")
    return df


def run():
    df = load_raw()
    df = clean(df)
    df = synthesize_volume(df)

    out = PROCESSED_DIR / "nifty_1min_clean.parquet"
    df.to_parquet(out, index=False, engine="pyarrow")
    print(f"[clean_data]   Saved → {out}  ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    run()
