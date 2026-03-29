"""
ATIS v5.0 — Global Macro Data Fetcher
Uses yfinance (free, no API key, no rate limits) for historical daily data.
Alpha Vantage kept for forex fallback. Finnhub reserved for live streaming.

Symbols: SPY (S&P500), QQQ (Nasdaq), UUP (DXY proxy), USD/INR, ^VIX
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import PROCESSED_DIR

# ── yfinance tickers ────────────────────────────────────────────
TICKERS = {
    "SPY":    "SPY",      # S&P 500 ETF
    "QQQ":    "QQQ",      # Nasdaq 100 ETF
    "UUP":    "UUP",      # US Dollar Index ETF (DXY proxy)
    "USDINR": "INR=X",    # USD/INR Forex
    "VIX":    "^VIX",     # CBOE Volatility Index
}

START_DATE = "2016-01-01"


def _fetch_yfinance(name: str, ticker: str) -> pd.DataFrame:
    """Fetch daily OHLCV via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print(f"[global_data] yfinance not installed. Run: pip install yfinance")
        return pd.DataFrame()

    print(f"[global_data] Fetching {name} ({ticker}) via yfinance ...")
    try:
        data = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)
        if data.empty:
            print(f"[global_data]   WARNING: No data for {name}")
            return pd.DataFrame()

        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()
        data = data.rename(columns={
            "Date": "date",
            "Open": f"{name}_open",
            "High": f"{name}_high",
            "Low": f"{name}_low",
            "Close": f"{name}_close",
            "Volume": f"{name}_volume",
        })

        # Keep only what we need
        cols = ["date"] + [c for c in data.columns if c.startswith(name)]
        data = data[cols].copy()
        data["date"] = pd.to_datetime(data["date"])

        print(f"[global_data]   {name}: {len(data)} daily bars "
              f"[{data['date'].min().date()} → {data['date'].max().date()}]")
        return data

    except Exception as e:
        print(f"[global_data]   ERROR fetching {name}: {e}")
        return pd.DataFrame()


def fetch_all():
    """Fetch all global tickers, merge on date, save parquet."""
    dfs = []

    for name, ticker in TICKERS.items():
        df = _fetch_yfinance(name, ticker)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("[global_data] ERROR: No data fetched at all!")
        return None

    # Merge all on date
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged.ffill().dropna(subset=["date"])

    out = PROCESSED_DIR / "global_daily.parquet"
    merged.to_parquet(out, index=False, engine="pyarrow")

    print(f"\n[global_data] ✅ Saved → {out}  ({len(merged)} rows)")
    print(f"[global_data] Columns: {list(merged.columns)}")
    print(f"[global_data] Date range: {merged['date'].min().date()} → {merged['date'].max().date()}")
    return merged


if __name__ == "__main__":
    fetch_all()
