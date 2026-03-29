"""
ATIS v4.0 — Market Simulator
Generates simulated real-time OHLCV bars and news headlines for dry-run testing.
Mimics Angel One API responses so the full pipeline can be validated end-to-end
without a live API connection.
"""
import sys
import json
import time
import random
import threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import PROCESSED_DIR


class MarketSimulator:
    """
    Simulates a live data feed by replaying historical bars with slight noise.
    Outputs:
      - data/trades/engine_state.json (fake engine state for dashboard testing)
      - Provides get_bars() method compatible with LiveEngine
    """

    def __init__(self, speed_multiplier: float = 1.0):
        self.speed = speed_multiplier
        self._history = None
        self._cursor = 0
        self._running = False
        self._load_history()

    def _load_history(self):
        """Load historical data as the replay source."""
        path = PROCESSED_DIR / "nifty_1min_clean.parquet"
        if not path.exists():
            path = PROCESSED_DIR / "nifty_features.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            # Use last 2000 bars for simulation
            self._history = df.tail(2000).reset_index(drop=True)
            self._cursor = 0
            print(f"[Simulator] Loaded {len(self._history)} bars for replay")
        else:
            print("[Simulator] ⚠️ No historical data found. Using synthetic data.")
            self._history = self._generate_synthetic()

    def _generate_synthetic(self, n_bars=2000):
        """Generate synthetic OHLCV when no historical data exists."""
        base = 23000
        timestamps = pd.date_range(end=datetime.now(), periods=n_bars, freq="1min")
        data = []
        price = base
        for ts in timestamps:
            change = random.gauss(0, 15)
            o = price
            h = o + abs(random.gauss(0, 20))
            l = o - abs(random.gauss(0, 20))
            c = o + change
            v = random.randint(50000, 500000)
            data.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
            price = c
        return pd.DataFrame(data)

    def get_bars(self, n_bars: int = 300) -> pd.DataFrame:
        """Get the next chunk of bars (simulating a rolling window)."""
        if self._history is None:
            return pd.DataFrame()
        end = min(self._cursor + n_bars, len(self._history))
        chunk = self._history.iloc[max(0, end - n_bars):end].copy()

        # Add slight noise to simulate live variation
        noise = np.random.normal(0, 0.5, len(chunk))
        chunk["close"] = chunk["close"] + noise
        chunk["high"] = chunk[["high", "close"]].max(axis=1)
        chunk["low"] = chunk[["low", "close"]].min(axis=1)

        return chunk

    def advance(self, bars: int = 1):
        """Advance the cursor (time passes)."""
        self._cursor = min(self._cursor + bars, len(self._history) - 1)

    def get_simulated_engine_state(self) -> dict:
        """Generate a fake engine_state for dashboard testing."""
        bars = self.get_bars(300)
        if bars.empty:
            return {}

        spot = float(bars["close"].iloc[-1])

        # Simulate model predictions (random but biased toward recent trend)
        recent_change = bars["close"].iloc[-1] - bars["close"].iloc[-10]
        bias = 2 if recent_change > 20 else 0 if recent_change < -20 else 1

        predictions = {
            "trend_catboost": bias,
            "fibo_xgboost": random.choice([bias, 1, bias]),
            "candle_catboost": random.choice([0, 1, 2]),
            "trap_xgboost": bias,
            "lgbm": bias,
            "lstm": random.choice([bias, 1]),
            "finbert": 1,  # Neutral in simulation
        }

        # Majority vote simulation
        votes = list(predictions.values())
        final = max(set(votes), key=votes.count)
        confidence = votes.count(final) / len(votes)

        direction = {0: "BEARISH", 1: "SIDEWAYS", 2: "BULLISH"}.get(final, "SIDEWAYS")

        signal = None
        if direction != "SIDEWAYS" and confidence >= 0.6:
            from src.signals.strike_selector import StrikeSelector
            selector = StrikeSelector()
            atr = float(bars["high"].iloc[-20:].mean() - bars["low"].iloc[-20:].mean())
            signal = selector.compute_entry_sl_targets(
                spot=spot, direction=direction, confidence=confidence,
                atr=atr, vix_proxy=0.15,
            )

        return {
            "timestamp": datetime.now().isoformat(),
            "spot": spot,
            "predictions": predictions,
            "final_prediction": final,
            "direction": direction,
            "confidence": round(confidence, 4),
            "signal": signal,
            "news": {"finbert_sentiment_score": random.uniform(-0.3, 0.3),
                     "news_override": False, "headlines_count": 5},
            "active_trade": None,
            "trade_stats": {"total": 0, "wins": 0, "losses": 0},
            "latency_seconds": round(random.uniform(0.1, 0.5), 3),
        }

    def run_simulation(self, interval: float = 1.0, duration_seconds: int = 300):
        """
        Run the simulation with REAL trade locking.
        - Locks trades when signals fire
        - Tracks premium (simulated) against SL/T1/T2/T3 each tick
        - Only unlocks when SL hit or T3 hit (or timeout at EOD)
        - No new signals while trade is locked
        """
        from src.signals.trade_manager import TradeManager

        state_path = ROOT / "data" / "trades" / "engine_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        trade_mgr = TradeManager()

        print(f"\n{'='*60}")
        print(f"  ATIS v4.0 — Market Simulator (Dry Run + Trade Locking)")
        print(f"  Duration: {duration_seconds}s | Interval: {interval}s")
        print(f"  State output: {state_path}")
        print(f"{'='*60}\n")

        self._running = True
        start = time.time()

        while self._running and (time.time() - start) < duration_seconds:
            state = self.get_simulated_engine_state()
            self.advance(1)

            spot = state.get("spot", 0)
            signal = state.get("signal")

            # ── TRADE LOCKING LOGIC ─────────────────────────
            # If a trade is locked, track the price (simulate premium movement)
            active = trade_mgr.get_active_trade()
            if active and active.get("status") == "LOCKED":
                # Simulate option premium movement based on spot change
                entry = active["entry"]
                direction = active.get("direction", "")
                # Estimate premium delta from spot movement
                if direction == "BULLISH":
                    delta = (spot - active.get("spot_at_signal", spot)) * 0.4
                else:
                    delta = (active.get("spot_at_signal", spot) - spot) * 0.4

                simulated_premium = max(1.0, entry + delta + random.gauss(0, 2))
                status = trade_mgr.update_price(simulated_premium)

                if status in ("WIN", "LOSS", "TIMEOUT"):
                    final_trade = trade_mgr.get_active_trade()
                    emoji = {"WIN": "✅", "LOSS": "❌", "TIMEOUT": "⏰"}.get(status, "")
                    pnl = final_trade.get("pnl", 0) if final_trade else 0
                    print(f"           {emoji} TRADE {status}! P&L: ₹{pnl:.2f}")

                # Don't generate new signals while locked
                state["signal"] = None

            elif signal and trade_mgr.can_lock():
                # New signal — LOCK IT
                # Attach spot_at_signal for premium tracking
                signal["spot_at_signal"] = spot
                trade = trade_mgr.lock_trade(signal)
                if trade:
                    print(f"           🔒 LOCKED: {signal['symbol']} @ ₹{signal['entry']:.2f}")

            # Update state with active trade info
            active_trade = trade_mgr.get_active_trade()
            state["active_trade"] = active_trade
            state["trade_stats"] = trade_mgr.get_statistics()

            # Write to JSON for dashboard
            try:
                with open(state_path, "w") as f:
                    json.dump(state, f, indent=2, default=str)
            except Exception:
                pass

            elapsed = time.time() - start
            direction = state.get("direction", "?")
            conf = state.get("confidence", 0)

            # Status line
            lock_indicator = ""
            if active_trade and active_trade.get("status") == "LOCKED":
                hits = []
                if active_trade.get("t1_hit"): hits.append("T1✅")
                if active_trade.get("t2_hit"): hits.append("T2✅")
                if active_trade.get("t3_hit"): hits.append("T3✅")
                hit_str = " ".join(hits) if hits else "tracking..."
                lock_indicator = f" | 🔒 {active_trade.get('symbol','')} [{hit_str}]"

            print(f"  [{elapsed:6.1f}s] Spot: ₹{spot:,.2f} | {direction} | Conf: {conf:.0%}{lock_indicator}")

            time.sleep(interval / self.speed)

        # Force close any remaining trade
        if trade_mgr.get_active_trade():
            trade_mgr.force_close_eod(spot)
            print("  [EOD] Force closed remaining trade.")

        self._running = False
        stats = trade_mgr.get_statistics()
        print(f"\n{'='*60}")
        print(f"  SIMULATION COMPLETE")
        print(f"  Total: {stats['total']} | Wins: {stats['wins']} | Losses: {stats['losses']}")
        print(f"  Win Rate: {stats['win_rate']}% | Total P&L: ₹{stats['total_pnl']:.2f}")
        print(f"{'='*60}")


    def stop(self):
        self._running = False


if __name__ == "__main__":
    sim = MarketSimulator(speed_multiplier=1.0)
    # Run for 5 minutes, updating every 1 second
    sim.run_simulation(interval=1.0, duration_seconds=300)
