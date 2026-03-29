"""
ATIS v4.0 — Live Inference Engine
Loads all 8 models, computes features, generates signals with entry/SL/targets.
Angel One API ready (activated when credentials go live).
"""
import sys
import time
import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import (
    MODELS_DIR, PROCESSED_DIR, CONFIDENCE_GATE,
    ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET,
)
from src.features.build_features import build_features
from src.signals.strike_selector import StrikeSelector
from src.signals.trade_manager import TradeManager
from src.news.finbert_agent import FinBERTAgent


class LiveEngine:
    """
    ATIS v4.0 Live Inference Engine.
    - Loads 8 models (7 ML + FinBERT)
    - Angel One API integration (ready to activate)
    - Computes 256 features per bar
    - Generates validated signals with entry/SL/targets
    - Total inference < 5s
    """

    def __init__(self):
        self.models = {}
        self.features = []
        self.scaler = None
        self.supervisor = None
        self.supervisor_scaler = None
        self.strike_selector = StrikeSelector()
        self.trade_manager = TradeManager()
        self.finbert = FinBERTAgent()
        self._angel_connected = False
        self._running = False
        self._history_df = None  # rolling bar history

    def load_models(self):
        """Load all saved models."""
        print("[LiveEngine] Loading models ...")

        # Load feature list
        feat_path = PROCESSED_DIR / "feature_list.txt"
        if feat_path.exists():
            self.features = feat_path.read_text().strip().split("\n")
        print(f"[LiveEngine]   {len(self.features)} features")

        # Load tree models
        for name in ["trend_catboost", "fibo_xgboost", "candle_catboost",
                      "trap_xgboost", "lgbm"]:
            path = MODELS_DIR / f"{name}.joblib"
            if path.exists():
                self.models[name] = joblib.load(path)
                print(f"[LiveEngine]   ✅ {name}")
            else:
                print(f"[LiveEngine]   ⚠️ {name} not found")

        # Load LSTM
        lstm_path = MODELS_DIR / "lstm.keras"
        if lstm_path.exists():
            try:
                import tensorflow as tf
                self.models["lstm"] = tf.keras.models.load_model(str(lstm_path))
                print("[LiveEngine]   ✅ lstm")
            except Exception as e:
                print(f"[LiveEngine]   ⚠️ lstm: {e}")

        # Load LSTM scaler
        scaler_path = MODELS_DIR / "lstm_scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

        # Load Supervisor
        sup_path = MODELS_DIR / "supervisor.joblib"
        if sup_path.exists():
            self.supervisor = joblib.load(sup_path)
            print("[LiveEngine]   ✅ supervisor")

        sup_scaler_path = MODELS_DIR / "supervisor_scaler.joblib"
        if sup_scaler_path.exists():
            self.supervisor_scaler = joblib.load(sup_scaler_path)

        # Load FinBERT
        print("[LiveEngine]   Loading FinBERT ...")
        self.finbert.load_model()
        print("[LiveEngine] All models loaded.")

    def connect_angel_one(self) -> bool:
        """
        Connect to Angel One API.
        Currently inactive — returns False.
        """
        if not ANGEL_API_KEY or ANGEL_API_KEY == "":
            print("[LiveEngine] Angel One: credentials not configured")
            return False

        try:
            # Placeholder — activate when API goes live
            # from SmartApi import SmartConnect
            # obj = SmartConnect(api_key=ANGEL_API_KEY)
            # data = obj.generateSession(ANGEL_CLIENT_ID, ANGEL_PASSWORD, totp)
            print("[LiveEngine] Angel One: API inactive (offline mode)")
            self._angel_connected = False
            return False
        except Exception as e:
            print(f"[LiveEngine] Angel One connection failed: {e}")
            return False

    def get_latest_bars(self, n_bars: int = 300) -> pd.DataFrame:
        """
        Get latest bars from Angel One API or historical CSV.
        Returns DataFrame with OHLCV columns.
        """
        if self._angel_connected:
            # TODO: Fetch real-time bars from Angel One
            pass

        # Offline: use last N bars from historical data
        if self._history_df is None:
            path = PROCESSED_DIR / "nifty_1min_clean.parquet"
            if path.exists():
                self._history_df = pd.read_parquet(path)

        if self._history_df is not None:
            return self._history_df.tail(n_bars).copy()
        return pd.DataFrame()

    def infer(self) -> dict:
        """
        Run full inference pipeline.
        Returns signal dict or None.
        Latency target: < 5 seconds.
        """
        t0 = time.time()

        # 1. Get latest bars
        bars = self.get_latest_bars(300)
        if len(bars) < 100:
            return {"error": "Insufficient data", "latency": time.time() - t0}

        # 2. Build features
        featured_df, feat_list = build_features(bars, fast_vpoc=True)
        feat_cols = [f for f in feat_list if f in featured_df.columns and f in self.features]
        if not feat_cols:
            feat_cols = [f for f in feat_list if f in featured_df.columns]

        last_row = featured_df.iloc[-1:]
        X = last_row[feat_cols].values.astype(np.float64)
        np.nan_to_num(X, copy=False)

        spot = float(bars["close"].iloc[-1])
        atr = float(last_row.get("atr_14", spot * 0.005).iloc[0]) if "atr_14" in last_row.columns else spot * 0.005
        vix = float(last_row.get("vix_proxy", 0.15).iloc[0]) if "vix_proxy" in last_row.columns else 0.15

        # 3. L1 Predictions (Tree models)
        predictions = {}
        for name in ["trend_catboost", "fibo_xgboost", "candle_catboost", "trap_xgboost", "lgbm"]:
            if name in self.models:
                predictions[name] = int(self.models[name].predict(X)[0])

        # 4. L2: LSTM Prediction (Sequence based)
        if "lstm" in self.models and self.scaler:
            X_all = featured_df[feat_cols].values.astype(np.float32)
            if len(X_all) >= 60:
                X_s = self.scaler.transform(X_all[-60:])
                X_seq = X_s.reshape(1, 60, X_s.shape[1])
                lstm_pred = self.models["lstm"].predict(X_seq, verbose=0).argmax(axis=1)[0]
                predictions["lstm"] = int(lstm_pred)

        # 5. L2: FinBERT sentiment
        news_state = self.finbert.get_aggregate()
        f_score = news_state["finbert_sentiment_score"]
        predictions["finbert"] = 2 if f_score > 0.2 else 0 if f_score < -0.2 else 1

        # 6. L3: Supervisor meta-prediction
        if self.supervisor and len(predictions) >= 3:
            # Order must match training: trend, fibo, candle, trap, lgbm, finbert, lstm (if exists)
            meta_keys = ["trend_catboost", "fibo_xgboost", "candle_catboost", "trap_xgboost", "lgbm", "finbert"]
            if "lstm" in predictions: meta_keys.append("lstm")
            
            meta_input = pd.DataFrame([{k: predictions.get(k, 1) for k in meta_keys}])
            X_meta = self.supervisor_scaler.transform(meta_input.fillna(1))
            final_pred = int(self.supervisor.predict(X_meta)[0])
            probs = self.supervisor.predict_proba(X_meta)[0]
            confidence = float(probs[final_pred])
        else:
            votes = list(predictions.values())
            final_pred = max(set(votes), key=votes.count) if votes else 1
            confidence = votes.count(final_pred) / max(len(votes), 1)

        # 7. News override: boost/dampen
        if news_state["news_override"]:
            confidence = min(confidence * 1.1, 0.99)
            if (f_score < -0.2 and final_pred == 2) or (f_score > 0.2 and final_pred == 0):
                confidence *= 0.6  # penalize conflict

        # 7. Generate signal
        direction = {0: "BEARISH", 2: "BULLISH"}.get(final_pred)
        signal = None

        if direction and confidence >= CONFIDENCE_GATE:
            signal = self.strike_selector.compute_entry_sl_targets(
                spot=spot, direction=direction, confidence=confidence,
                atr=atr, vix_proxy=vix,
            )

            # 8. Lock trade if valid
            if signal and self.trade_manager.can_lock():
                self.trade_manager.lock_trade(signal)

        latency = time.time() - t0

        result = {
            "timestamp": datetime.now().isoformat(),
            "spot": spot,
            "predictions": predictions,
            "final_prediction": final_pred,
            "direction": direction or "SIDEWAYS",
            "confidence": round(confidence, 4),
            "signal": signal,
            "news": news_state,
            "active_trade": self.trade_manager.get_active_trade(),
            "trade_stats": self.trade_manager.get_statistics(),
            "latency_seconds": round(latency, 3),
        }

        return result

    def start(self, interval: int = 60):
        """Start live inference loop."""
        self.load_models()
        self.connect_angel_one()
        self.finbert.start_background_polling(interval=60)
        self._running = True

        print(f"\n[LiveEngine] ▶️  Live engine started (interval={interval}s)")
        while self._running:
            try:
                result = self.infer()
                self._print_status(result)
            except Exception as e:
                print(f"[LiveEngine] Error: {e}")
            time.sleep(interval)

    def stop(self):
        self._running = False
        self.finbert.stop()

    def _print_status(self, result: dict):
        print(f"\n{'─'*50}")
        print(f"  {result['timestamp']}  |  Spot: {result['spot']:.2f}")
        print(f"  Direction: {result['direction']}  |  Confidence: {result['confidence']:.1%}")
        print(f"  Latency: {result['latency_seconds']:.3f}s")
        if result.get("signal"):
            print(f"  Signal: {result['signal']['symbol']} @ ₹{result['signal']['entry']:.2f}")
        if result.get("active_trade"):
            t = result["active_trade"]
            print(f"  Active Trade: {t['id']} — {t['status']}")
        print(f"{'─'*50}")
        
        # EXPORT STATE TO DASHBOARD
        state_path = ROOT / "data" / "trades" / "engine_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(state_path, "w") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ATIS v4.0 Live Engine")
    parser.add_argument("--simulate", action="store_true",
                        help="Run in simulation mode (no real API, replays historical data)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Inference interval in seconds (default: 60)")
    parser.add_argument("--duration", type=int, default=300,
                        help="Simulation duration in seconds (default: 300)")
    args = parser.parse_args()

    if args.simulate:
        print("=" * 50)
        print("  ATIS v4.0 — SIMULATION MODE")
        print("  Dashboard: python app.py")
        print("=" * 50)
        from src.utils.market_simulator import MarketSimulator
        sim = MarketSimulator()
        sim.run_simulation(interval=max(1, args.interval), duration_seconds=args.duration)
    else:
        engine = LiveEngine()
        engine.start(interval=args.interval)
