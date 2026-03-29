"""
ATIS v4.0 — Data Resampling
Resamples clean 1-min data into 5min, 15min, 1H OHLCV bars.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import PROCESSED_DIR


RESAMPLE_MAP = {
    "5min":  "5min",
    "15min": "15min",
    "1H":    "1h",
}


def resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample OHLCV to the given frequency."""
    df = df.set_index("timestamp")
    agg = df.resample(freq, label="left", closed="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["close"])
    return agg.reset_index()


def run():
    src = PROCESSED_DIR / "nifty_1min_clean.parquet"
    print(f"[resample] Loading {src} ...")
    df = pd.read_parquet(src)
    print(f"[resample]   {len(df):,} 1-min bars")

    for label, freq in RESAMPLE_MAP.items():
        resampled = resample(df, freq)
        out = PROCESSED_DIR / f"nifty_{label}_clean.parquet"
        resampled.to_parquet(out, index=False, engine="pyarrow")
        print(f"[resample]   {label}: {len(resampled):,} bars → {out}")

    print("[resample] Done.")


if __name__ == "__main__":
    run()
