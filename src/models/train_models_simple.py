"""
ATIS v4.0 — Optimized Training Pipeline (Codespaces)
Simplified trainer for memory-limited environments
Builds 5 core models + Supervisor
"""
import sys
import json
import time
from pathlib import Path

import os
os.environ['CATBOOST_DEV_MODE'] = '0'

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Import CatBoost with widgets disabled
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import PROCESSED_DIR, MODELS_DIR, HOLDOUT_START, RANDOM_STATE

TARGET_F1 = 0.68

def load_data():
    """Load features and split into train/holdout."""
    print(f"[train_simple] Loading data...")
    df = pd.read_parquet(PROCESSED_DIR / "nifty_features.parquet")
    
    feat_path = PROCESSED_DIR / "feature_list.txt"
    features = feat_path.read_text().strip().split("\n")
    features = [f for f in features if f in df.columns]
    
    print(f"[train_simple] Loaded: {len(features)} features, {len(df):,} rows")
    
    # Split on timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    holdout_mask = df["timestamp"] >= HOLDOUT_START
    train_df = df[~holdout_mask].dropna(subset=["target"]).copy()
    holdout_df = df[holdout_mask].dropna(subset=["target"]).copy()
    
    print(f"[train_simple] Train: {len(train_df):,}  |  Holdout: {len(holdout_df):,}")
    
    X_train = train_df[features].fillna(0).values
    y_train = train_df["target"].values.astype(int)
    X_holdout = holdout_df[features].fillna(0).values
    y_holdout = holdout_df["target"].values.astype(int)
    
    # Replace infinities
    for X in [X_train, X_holdout]:
        np.nan_to_num(X, copy=False, nan=0, posinf=0, neginf=0)
    
    return X_train, y_train, X_holdout, y_holdout, features


def train_model(name, X_train, y_train, X_val, y_val, X_holdout, y_holdout):
    """Train a single model and evaluate."""
    print(f"[train_simple] Training {name}...")
    start = time.time()
    
    try:
        if name == "trend_catboost":
            model = CatBoostClassifier(
                iterations=300, depth=5, learning_rate=0.05,
                auto_class_weights="Balanced", verbose=0, random_state=RANDOM_STATE
            )
        elif name == "fibo_xgboost":
            model = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                tree_method="hist", eval_metric="mlogloss", random_state=RANDOM_STATE
            )
        elif name == "candle_catboost":
            model = CatBoostClassifier(
                iterations=250, depth=5, learning_rate=0.04,
                auto_class_weights="Balanced", verbose=0, random_state=RANDOM_STATE
            )
        elif name == "trap_xgboost":
            model = XGBClassifier(
                n_estimators=250, max_depth=5, learning_rate=0.04,
                tree_method="hist", eval_metric="mlogloss", random_state=RANDOM_STATE
            )
        elif name == "lgbm":
            model = LGBMClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.04,
                class_weight="balanced", verbose=-1, random_state=RANDOM_STATE
            )
        else:
            return None, 0, 0
        
        # Train on train data
        if hasattr(model, 'fit'):
            model.fit(X_train, y_train)  # eval_set not used for faster training
        
        # Predict on holdout
        y_pred = model.predict(X_holdout)
        acc = accuracy_score(y_holdout, y_pred)
        f1 = f1_score(y_holdout, y_pred, average="weighted", zero_division=0)
        
        elapsed = time.time() - start
        print(f"  ✅ {name:20s} | Acc: {acc:.4f} | F1: {f1:.4f} | Time: {elapsed:.0f}s")
        
        return model, acc, f1
    except Exception as e:
        print(f"  ❌ {name:20s} | ERROR: {str(e)[:60]}")
        return None, 0, 0


def main():
    """Train all models."""
    print("[train_simple] ═" * 40)
    print("[train_simple] ATIS v4.0 - Optimized Training")
    print("[train_simple] ═" * 40)
    
    # Load data
    X_train_full, y_train_full, X_holdout, y_holdout, features = load_data()
    
    # Split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2,
        random_state=RANDOM_STATE, stratify=y_train_full
    )
    
    print(f"[train_simple] Data split → Train: {len(X_train):,} | Val: {len(X_val):,} | Holdout: {len(X_holdout):,}")
    
    # Train models
    models_dict = {}
    l1_preds_val = []
    l1_preds_holdout = []
    
    model_names = ["trend_catboost", "fibo_xgboost", "candle_catboost", "trap_xgboost", "lgbm"]
    
    print(f"\n[train_simple] Training L1 Base Models...")
    for name in model_names:
        model, acc_h, f1_h = train_model(name, X_train, y_train, X_val, y_val, X_holdout, y_holdout)
        
        if model is not None:
            models_dict[name] = (model, acc_h, f1_h)
            
            # Get predictions for supervisor training
            if hasattr(model, 'predict_proba'):
                l1_preds_val.append(model.predict_proba(X_val))
                l1_preds_holdout.append(model.predict_proba(X_holdout))
            else:
                l1_preds_val.append(model.predict(X_val))
                l1_preds_holdout.append(model.predict(X_holdout))
            
            # Save model
            model_path = MODELS_DIR / f"{name}.joblib"
            joblib.dump(model, model_path)
            print(f"  Saved → {model_path}")
    
    # Train Supervisor (L3)
    print(f"\n[train_simple] Training L3 Supervisor...")
    if l1_preds_val:
        # Stack L1 predictions
        X_val_l3 = np.hstack(l1_preds_val)
        X_holdout_l3 = np.hstack(l1_preds_holdout)
        
        supervisor = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        supervisor.fit(X_val_l3, y_val)
        
        y_pred_supervisor = supervisor.predict(X_holdout_l3)
        acc_sup = accuracy_score(y_holdout, y_pred_supervisor)
        f1_sup = f1_score(y_holdout, y_pred_supervisor, average="weighted", zero_division=0)
        
        print(f"  ✅ {'supervisor':20s} | Acc: {acc_sup:.4f} | F1: {f1_sup:.4f}")
        
        sup_path = MODELS_DIR / "supervisor.joblib"
        joblib.dump(supervisor, sup_path)
        print(f"  Saved → {sup_path}")
    
    # Generate report
    print(f"\n[train_simple] ═" * 40)
    print("[train_simple] FINAL METRICS")
    print("[train_simple] ═" * 40)
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "4.0-optimized",
        "model_count": len(models_dict),
        "target_f1": TARGET_F1,
        "holdout_results": {}
    }
    
    for name, (model, acc, f1) in models_dict.items():
        report["holdout_results"][name] = {"accuracy": float(acc), "f1": float(f1)}
        status = "✅" if f1 >= TARGET_F1 * 0.9 else "⚠️"
        print(f"{status} {name:20s} → F1: {f1:.4f}")
    
    if l1_preds_val:
        report["holdout_results"]["supervisor"] = {"accuracy": float(acc_sup), "f1": float(f1_sup)}
        print(f"✅ {'supervisor':20s} → F1: {f1_sup:.4f}")
    
    report_path = MODELS_DIR / "final_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[train_simple] Report → {report_path}")
    print("[train_simple] ✅ Training complete!")


if __name__ == "__main__":
    main()
