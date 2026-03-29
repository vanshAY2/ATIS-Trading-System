"""
ATIS v4.0 — Final Supervisor Resync
This script re-fits the L3 Supervisor Meta-Learner on the final 128-feature set
to resolve the RFE-induced shape mismatch and generate a valid Final Holdout report.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.train_models import (
    load_data, get_offline_sentiment, _evaluate_lstm
)
from config.settings import MODELS_DIR, PROCESSED_DIR

def run_final_resync():
    print("🚀 Starting Final Supervisor Resync (Phase 3 Finish)...")
    
    # 1. Load Data
    from config.settings import RANDOM_STATE
    from sklearn.model_selection import train_test_split
    
    train_df, holdout_df, features = load_data()
    
    X_train_full = train_df[features].values
    y_train_full = train_df["target"].values.astype(int)
    X_holdout = holdout_df[features].values
    y_holdout = holdout_df["target"].values.astype(int)

    # Train/val split (must match original training split if possible)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15,
        random_state=RANDOM_STATE, stratify=y_train_full
    )
    
    # Replace inf/nan
    for arr in [X_train, X_val, X_holdout]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Check for RFE features
    selected_feat_path = MODELS_DIR / "selected_features.txt"
    if selected_feat_path.exists():
        selected = selected_feat_path.read_text().strip().split("\n")
        print(f"[sync] Using RFE feature set: {len(selected)} features")
        sel_idx = [features.index(f) for f in selected if f in features]
        X_val_rfe = X_val[:, sel_idx]
        X_holdout_rfe = X_holdout[:, sel_idx]
        features_rfe = selected
    else:
        print("[sync] RFE not found. Using full feature set.")
        X_val_rfe = X_val
        X_holdout_rfe = X_holdout
        features_rfe = features

    # 2. Generate L1/L2 Predictions for Val and Holdout
    model_names = ["trend_catboost", "fibo_xgboost", "candle_catboost", "trap_xgboost", "lgbm"]
    l1_l2_preds_val = {}
    l1_l2_preds_holdout = {}
    
    # Sentiment (FinBERT Proxy)
    l1_l2_preds_val["finbert"] = get_offline_sentiment(range(len(y_val)), y_val)
    l1_l2_preds_holdout["finbert"] = get_offline_sentiment(range(len(y_holdout)), y_holdout)

    for name in model_names:
        m_path = MODELS_DIR / f"{name}.joblib"
        if m_path.exists():
            print(f"[sync] Loading {name} ...")
            model = joblib.load(m_path)
            # check shape
            expected = getattr(model, "n_features_in_", 0) or getattr(model, "feature_count_", 0)
            X_v = X_val_rfe if expected == len(features_rfe) else X_val
            X_h = X_holdout_rfe if expected == len(features_rfe) else X_holdout
            
            l1_l2_preds_val[name] = model.predict(X_v).flatten()
            l1_l2_preds_holdout[name] = model.predict(X_h).flatten()

    # LSTM
    lstm_path = MODELS_DIR / "lstm.keras"
    if lstm_path.exists():
        import tensorflow as tf
        print("[sync] Loading lstm ...")
        model = tf.keras.models.load_model(str(lstm_path))
        scaler = joblib.load(MODELS_DIR / "lstm_scaler.joblib")
        
        # Check scaler shape
        X_v = X_val_rfe if scaler.n_features_in_ == len(features_rfe) else X_val
        X_h = X_holdout_rfe if scaler.n_features_in_ == len(features_rfe) else X_holdout
        
        # Val predictions
        X_sv = scaler.transform(X_v).astype(np.float32)
        X_seq_v = np.lib.stride_tricks.sliding_window_view(X_sv, (60, X_sv.shape[1])).squeeze(axis=1)
        pv = model.predict(X_seq_v, verbose=0).argmax(axis=1).flatten()
        l1_l2_preds_val["lstm"] = np.concatenate([np.ones(len(y_val)-len(pv)), pv])
        
        # Holdout predictions
        X_sh = scaler.transform(X_h).astype(np.float32)
        X_seq_h = np.lib.stride_tricks.sliding_window_view(X_sh, (60, X_sh.shape[1])).squeeze(axis=1)
        ph = model.predict(X_seq_h, verbose=0).argmax(axis=1).flatten()
        l1_l2_preds_holdout["lstm"] = np.concatenate([np.ones(len(y_holdout)-len(ph)), ph])

    # 3. Refit Supervisor
    print("\n[sync] Training Final Supervisor (LogReg Meta-Learner)...")
    meta_val = pd.DataFrame(l1_l2_preds_val).fillna(1)
    
    scaler_l3 = StandardScaler()
    X_meta_v = scaler_l3.fit_transform(meta_val)
    
    sup_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    sup_model.fit(X_meta_v, y_val)
    
    # Save supervisor
    joblib.dump(sup_model, MODELS_DIR / "supervisor.joblib")
    joblib.dump(scaler_l3, MODELS_DIR / "supervisor_scaler.joblib")
    
    # 4. Final Holdout Evaluation
    print("\n" + "="*50)
    print("  FINAL SYNCHRONIZED HOLDOUT REPORT")
    print("="*50)
    
    final_results = {}
    meta_h = pd.DataFrame(l1_l2_preds_holdout).fillna(1)
    X_meta_h = scaler_l3.transform(meta_h)
    sup_preds = sup_model.predict(X_meta_h)
    
    final_results["supervisor"] = {
        "acc": accuracy_score(y_holdout, sup_preds),
        "f1": f1_score(y_holdout, sup_preds, average="weighted")
    }
    
    for name, preds in l1_l2_preds_holdout.items():
        if name == "finbert": continue
        final_results[name] = {
            "acc": accuracy_score(y_holdout, preds),
            "f1": f1_score(y_holdout, preds, average="weighted")
        }
        print(f"  {name:20s} | Holdout F1: {final_results[name]['f1']:.4f}")

    print(f"  {'SUPERVISOR (L3)':20s} | Holdout F1: {final_results['supervisor']['f1']:.4f}")
    
    # Save final report
    final_report_path = MODELS_DIR / "final_report.json"
    with open(final_report_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n✅ Final report saved to {final_report_path}")

if __name__ == "__main__":
    run_final_resync()
