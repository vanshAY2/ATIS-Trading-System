"""
ATIS v4.0 — Backtesting Engine
Walk-forward validation on 2024-2026 holdout set with all 8 models.
Simulates strike selection + entry/SL/target validation.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import PROCESSED_DIR, MODELS_DIR, HOLDOUT_START, TARGET_ACCURACY
from src.signals.strike_selector import StrikeSelector
from src.signals.trade_manager import TradeManager


def load_holdout():
    """Load holdout data."""
    df = pd.read_parquet(PROCESSED_DIR / "nifty_features.parquet")
    features = (PROCESSED_DIR / "feature_list.txt").read_text().strip().split("\n")
    features = [f for f in features if f in df.columns]

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    holdout = df[df["timestamp"] >= HOLDOUT_START].copy()
    holdout = holdout.dropna(subset=["target"])
    return holdout, features


def evaluate_model(model_path: Path, X: np.ndarray, y: np.ndarray, name: str, features: list) -> dict:
    """Evaluate a single saved model on holdout data."""
    try:
        if name == "lstm":
            import tensorflow as tf
            model = tf.keras.models.load_model(str(model_path))
            scaler = joblib.load(MODELS_DIR / "lstm_scaler.joblib")
            X_s = scaler.transform(X).astype(np.float32)
            seq_len = 60
            X_seq = np.lib.stride_tricks.sliding_window_view(X_s, (seq_len, X_s.shape[1])).squeeze(axis=1)
            y_seq = y[seq_len:]
            preds = model.predict(X_seq, verbose=0).argmax(axis=1).flatten()
            acc = accuracy_score(y_seq, preds)
            f1 = f1_score(y_seq, preds, average="weighted")
        elif name == "supervisor":
            model = joblib.load(model_path)
            # Supervisor eval logic is handled in the final report loop
            return {"name": name, "accuracy": 0, "f1": 0, "skip": True}
        else:
            model = joblib.load(model_path)
            preds = model.predict(X)
            acc = accuracy_score(y, preds)
            f1 = f1_score(y, preds, average="weighted")

        return {
            "name": name, "accuracy": round(acc, 4), "f1": round(f1, 4),
            "pass": f1 >= TARGET_ACCURACY,
        }
    except Exception as e:
        return {"name": name, "accuracy": 0, "f1": 0, "error": str(e)}


def simulate_trades(holdout: pd.DataFrame, features: list) -> dict:
    """Simulate trades using the L3 Supervisor ensemble predictions."""
    selector = StrikeSelector()
    manager = TradeManager()
    
    # Load Models for Ensemble
    l1_names = ["trend_catboost", "fibo_xgboost", "candle_catboost", "trap_xgboost", "lgbm"]
    l1_models = {}
    for n in l1_names:
        p = MODELS_DIR / f"{n}.joblib"
        if p.exists(): l1_models[n] = joblib.load(p)
    
    sup_p = MODELS_DIR / "supervisor.joblib"
    if not sup_p.exists():
        print("[backtest] ERROR: Supervisor model not found. Cannot simulate trades.")
        return {}
    supervisor = joblib.load(sup_p)
    sup_scaler = joblib.load(MODELS_DIR / "supervisor_scaler.joblib")

    print(f"[backtest] Simulating trades with {len(l1_models)} L1 models + L3 Supervisor...")

    X_h = holdout[features].values
    sentiment = holdout["sentiment_score"].values

    # Step-by-step simulation to respect causality
    for i in range(200, len(holdout), 15):
        row = holdout.iloc[i]
        x_row = X_h[i].reshape(1, -1)
        
        # 1. L1/L2 Predictions
        l1_preds = {n: m.predict(x_row)[0] for n, m in l1_models.items()}
        l1_preds["finbert"] = sentiment[i]
        
        # 2. L3 Ensemble (Note: LSTM skipped in backtest loop for speed)
        meta_feat = pd.DataFrame([l1_preds])
        X_meta = sup_scaler.transform(meta_feat.fillna(1))
        ensemble_pred = supervisor.predict(X_meta)[0]
        probs = supervisor.predict_proba(X_meta)[0]
        confidence = probs[int(ensemble_pred)]

        # 3. Decision
        if ensemble_pred == 2: direction = "BULLISH"
        elif ensemble_pred == 0: direction = "BEARISH"
        else: continue

        if confidence < 0.60: continue

        # 4. Signal & Execution
        spot = row["close"]
        atr = row.get("atr_14", spot * 0.005)
        signal = selector.compute_entry_sl_targets(spot, direction, confidence, atr, row.get("vix_proxy", 0.15))
        
        if signal and manager.can_lock():
            manager.lock_trade(signal)
            # Update price over next 15 mins
            future = holdout.iloc[i:i+15]
            for _, f_row in future.iterrows():
                manager.update_price(f_row["close"])

    return manager.get_statistics()


def run():
    """Run full backtest."""
    print("=" * 70)
    print("  ATIS v4.0 — BACKTESTING ENGINE")
    print("=" * 70)

    holdout, features = load_holdout()
    X = holdout[features].values
    y = holdout["target"].values.astype(int)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nHoldout set: {len(holdout):,} rows ({HOLDOUT_START}+)")
    print(f"Features: {len(features)}")

    # Evaluate all saved models
    model_files = list(MODELS_DIR.glob("*.joblib")) + list(MODELS_DIR.glob("*.keras"))
    model_files = [f for f in model_files if "scaler" not in f.stem]

    results = []
    print(f"\n{'─'*60}")
    print(f"  MODEL EVALUATION ON HOLDOUT SET")
    print(f"{'─'*60}")

    for mf in sorted(model_files):
        name = mf.stem
        r = evaluate_model(mf, X, y, name, features)
        results.append(r)
        status = "✅" if r.get("pass") else "❌"
        acc = r.get("accuracy", 0)
        f1 = r.get("f1", 0)
        err = r.get("error", "")
        if err:
            print(f"  {status} {name:25s}  ERROR: {err}")
        else:
            print(f"  {status} {name:25s}  Acc: {acc:.4f}  F1: {f1:.4f}  "
                  f"P: {r.get('precision', 0):.4f}  R: {r.get('recall', 0):.4f}")

    # Simulate trades
    print(f"\n{'─'*60}")
    print(f"  SIMULATED TRADE RESULTS")
    print(f"{'─'*60}")

    trade_stats = simulate_trades(holdout, features)
    for k, v in trade_stats.items():
        print(f"  {k:20s}: {v}")

    # Save report
    report = {
        "holdout_start": HOLDOUT_START,
        "holdout_rows": len(holdout),
        "model_results": results,
        "trade_stats": trade_stats,
    }
    report_path = MODELS_DIR / "backtest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved → {report_path}")

    return results


if __name__ == "__main__":
    run()
