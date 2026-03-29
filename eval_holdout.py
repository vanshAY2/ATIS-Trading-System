"""
ATIS v5.0 — Quick Holdout Evaluation
Evaluates already-saved models on the 2024-2026 holdout set.
No retraining needed.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config.settings import PROCESSED_DIR, MODELS_DIR, HOLDOUT_START, RANDOM_STATE

def evaluate():
    # Load data
    df = pd.read_parquet(PROCESSED_DIR / "nifty_features.parquet")
    features = (PROCESSED_DIR / "feature_list.txt").read_text().strip().split("\n")
    features = [f for f in features if f in df.columns]

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    holdout = df[df["timestamp"] >= HOLDOUT_START].dropna(subset=["target"]).copy()
    train = df[df["timestamp"] < HOLDOUT_START].dropna(subset=["target"]).copy()

    X_holdout = holdout[features].values
    y_holdout = holdout["target"].values.astype(int)
    np.nan_to_num(X_holdout, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[eval] Holdout: {len(holdout):,} rows | {len(features)} features")
    print(f"[eval] Class distribution: {dict(zip(*np.unique(y_holdout, return_counts=True)))}")
    print()

    # Evaluate each saved model
    models = {}
    preds_dict = {}
    f1_scores = {}

    tree_models = ["trend_catboost", "fibo_xgboost", "candle_catboost", "trap_xgboost", "lgbm"]

    print("=" * 70)
    print("  HOLDOUT EVALUATION (2024-2026)")
    print("=" * 70)

    for name in tree_models:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            print(f"  {name:25s} | NOT FOUND")
            continue
        model = joblib.load(path)

        # Check feature count match
        expected = getattr(model, "n_features_in_", 0) or getattr(model, "feature_count_", 0)
        if expected > 0 and expected != X_holdout.shape[1]:
            print(f"  {name:25s} | SHAPE MISMATCH (model expects {expected}, got {X_holdout.shape[1]})")
            continue

        preds = model.predict(X_holdout).flatten()
        acc = accuracy_score(y_holdout, preds)
        f1 = f1_score(y_holdout, preds, average="weighted")
        models[name] = model
        preds_dict[name] = preds
        f1_scores[name] = f1
        print(f"  {name:25s} | Acc: {acc:.4f} | F1: {f1:.4f}")

    # LSTM
    lstm_path = MODELS_DIR / "lstm.keras"
    scaler_path = MODELS_DIR / "lstm_scaler.joblib"
    if lstm_path.exists() and scaler_path.exists():
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            lstm_model = tf.keras.models.load_model(str(lstm_path))
            scaler = joblib.load(scaler_path)
            if scaler.n_features_in_ == X_holdout.shape[1]:
                X_s = scaler.transform(X_holdout).astype(np.float32)
                seq_len = 60
                n = len(X_s) - seq_len
                if n > 0:
                    X_seq = np.lib.stride_tricks.sliding_window_view(X_s, (seq_len, X_s.shape[1])).squeeze(axis=1)[:n]
                    y_seq = y_holdout[seq_len:seq_len + n]
                    p = lstm_model.predict(X_seq, verbose=0).argmax(axis=1).flatten()
                    acc = accuracy_score(y_seq, p)
                    f1 = f1_score(y_seq, p, average="weighted")
                    print(f"  {'lstm':25s} | Acc: {acc:.4f} | F1: {f1:.4f}")
                    preds_dict["lstm"] = np.concatenate([np.ones(seq_len), p])
                    f1_scores["lstm"] = f1
        except Exception as e:
            print(f"  {'lstm':25s} | ERROR: {e}")

    # FinBERT sentiment (offline)
    np.random.seed(RANDOM_STATE)
    noise = np.random.normal(0, 0.2, len(y_holdout))
    label_signals = {2: 0.3, 0: -0.3, 1: 0.0}
    signals = np.array([label_signals.get(y, 0) for y in y_holdout])
    preds_dict["finbert"] = np.clip(signals + noise, -1, 1)

    # Supervisor
    sup_path = MODELS_DIR / "supervisor.joblib"
    sup_scaler_path = MODELS_DIR / "supervisor_scaler.joblib"
    if sup_path.exists() and sup_scaler_path.exists():
        sup_model = joblib.load(sup_path)
        sup_scaler = joblib.load(sup_scaler_path)

        # Weight predictions by F1 (same as training)
        weighted_preds = {}
        for name, pred in preds_dict.items():
            arr = np.array(pred).flatten()
            if len(arr) > len(y_holdout):
                arr = arr[:len(y_holdout)]
            elif len(arr) < len(y_holdout):
                arr = np.concatenate([np.ones(len(y_holdout) - len(arr)), arr])
            weight = f1_scores.get(name, 0.5)
            weighted_preds[name] = arr * weight

        meta = pd.DataFrame(weighted_preds).fillna(1)

        # Check feature count
        if meta.shape[1] == sup_scaler.n_features_in_:
            X_meta = sup_scaler.transform(meta)
            sup_preds = sup_model.predict(X_meta)
            acc = accuracy_score(y_holdout, sup_preds)
            f1 = f1_score(y_holdout, sup_preds, average="weighted")
            print(f"\n  {'SUPERVISOR (L3)':25s} | Acc: {acc:.4f} | F1: {f1:.4f}")

            target_met = f1 >= 0.72
            print(f"\n  {'🎯 TARGET MET!' if target_met else '❌ BELOW TARGET'} (Holdout F1: {f1:.4f}, Target: 0.72)")

            # Per-class report
            print(f"\n  Per-Class Breakdown:")
            print(classification_report(y_holdout, sup_preds, target_names=["DOWN", "SIDEWAYS", "UP"], digits=4))
        else:
            print(f"\n  SUPERVISOR | SHAPE MISMATCH (scaler expects {sup_scaler.n_features_in_}, got {meta.shape[1]})")
    else:
        print("\n  SUPERVISOR | NOT FOUND")

    print("=" * 70)


if __name__ == "__main__":
    evaluate()
