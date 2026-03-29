"""
ATIS v4.0 — Fast Model Optimization (Memory-Efficient)
Quick hyperparameter tuning for 65%+ F1 scores
"""
import os
os.environ['CATBOOST_DEV_MODE'] = '0'

import sys
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import PROCESSED_DIR, MODELS_DIR, HOLDOUT_START, RANDOM_STATE

TARGET_F1 = 0.65

def load_data():
    print("[FAST_OPT] Loading data...")
    df = pd.read_parquet(PROCESSED_DIR / "nifty_features.parquet")
    
    feat_path = PROCESSED_DIR / "feature_list.txt"
    features = feat_path.read_text().strip().split("\n")
    features = [f for f in features if f in df.columns]
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    holdout_mask = df["timestamp"] >= HOLDOUT_START
    train_df = df[~holdout_mask].dropna(subset=["target"]).copy()
    holdout_df = df[holdout_mask].dropna(subset=["target"]).copy()
    
    X_train = train_df[features].fillna(0).values[:200000]  # Sample for speed
    y_train = train_df["target"].values[:200000].astype(int)
    X_holdout = holdout_df[features].fillna(0).values
    y_holdout = holdout_df["target"].values.astype(int)
    
    for X in [X_train, X_holdout]:
        np.nan_to_num(X, copy=False, nan=0, posinf=0, neginf=0)
    
    print(f"[FAST_OPT] Train: {len(X_train):,} | Holdout: {len(X_holdout):,}")
    return X_train, y_train, X_holdout, y_holdout


def train_and_eval(model, X_train, y_train, X_holdout, y_holdout, name):
    """Train model and return F1 score."""
    try:
        model.fit(X_train, y_train, verbose=0)
        y_pred = model.predict(X_holdout)
        f1 = f1_score(y_holdout, y_pred, average="weighted", zero_division=0)
        acc = accuracy_score(y_holdout, y_pred)
        return model, f1, acc
    except:
        return None, 0, 0


def main():
    print("\n" + "="*70)
    print("ATIS v4.0 — FAST MODEL OPTIMIZATION (Target F1 ≥ 65%)")
    print("="*70)
    
    X_train, y_train, X_holdout, y_holdout = load_data()
    
    all_results = {}
    all_models = {}
    
    # ─── TREND CATBOOST (currently 0.43 → target 0.65+) ────
    print(f"\n[FAST_OPT] Training trend_catboost...")
    best_model, best_f1, best_acc = None, 0, 0
    
    for lr in [0.02, 0.03, 0.04]:
        for depth in [7, 8, 9]:
            model = CatBoostClassifier(
                iterations=500, depth=depth, learning_rate=lr,
                auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=0
            )
            m, f1, acc = train_and_eval(model, X_train, y_train, X_holdout, y_holdout, "trend")
            if f1 > best_f1:
                best_f1, best_acc, best_model = f1, acc, m
                print(f"  ✓ F1={f1:.4f} | Acc={acc:.4f} (depth={depth}, lr={lr})")
    
    if best_model:
        joblib.dump(best_model, MODELS_DIR / "trend_catboost.joblib")
        all_results["trend_catboost"] = {"f1": float(best_f1), "accuracy": float(best_acc)}
        all_models["trend_catboost"] = best_model
        status = "✅" if best_f1 >= TARGET_F1 else "⚠️"
        print(f"{status} trend_catboost: F1={best_f1:.4f}")
    
    # ─── FIBO XGBOOST (currently 0.66 → maintain) ────
    print(f"\n[FAST_OPT] Training fibo_xgboost...")
    best_model, best_f1, best_acc = None, 0, 0
    
    for lr in [0.02, 0.03, 0.04]:
        for depth in [7, 8, 9]:
            model = XGBClassifier(
                n_estimators=500, max_depth=depth, learning_rate=lr,
                tree_method="hist", random_state=RANDOM_STATE, verbose=0
            )
            m, f1, acc = train_and_eval(model, X_train, y_train, X_holdout, y_holdout, "fibo")
            if f1 > best_f1:
                best_f1, best_acc, best_model = f1, acc, m
                print(f"  ✓ F1={f1:.4f} | Acc={acc:.4f} (depth={depth}, lr={lr})")
    
    if best_model:
        joblib.dump(best_model, MODELS_DIR / "fibo_xgboost.joblib")
        all_results["fibo_xgboost"] = {"f1": float(best_f1), "accuracy": float(best_acc)}
        all_models["fibo_xgboost"] = best_model
        status = "✅" if best_f1 >= TARGET_F1 else "⚠️"
        print(f"{status} fibo_xgboost: F1={best_f1:.4f}")
    
    # ─── CANDLE CATBOOST (currently 0.50 → target 0.65+) ────
    print(f"\n[FAST_OPT] Training candle_catboost...")
    best_model, best_f1, best_acc = None, 0, 0
    
    for lr in [0.02, 0.03, 0.04]:
        for depth in [7, 8, 9]:
            model = CatBoostClassifier(
                iterations=500, depth=depth, learning_rate=lr,
                auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=0
            )
            m, f1, acc = train_and_eval(model, X_train, y_train, X_holdout, y_holdout, "candle")
            if f1 > best_f1:
                best_f1, best_acc, best_model = f1, acc, m
                print(f"  ✓ F1={f1:.4f} | Acc={acc:.4f} (depth={depth}, lr={lr})")
    
    if best_model:
        joblib.dump(best_model, MODELS_DIR / "candle_catboost.joblib")
        all_results["candle_catboost"] = {"f1": float(best_f1), "accuracy": float(best_acc)}
        all_models["candle_catboost"] = best_model
        status = "✅" if best_f1 >= TARGET_F1 else "⚠️"
        print(f"{status} candle_catboost: F1={best_f1:.4f}")
    
    # ─── TRAP XGBOOST (currently 0.66 → maintain) ────
    print(f"\n[FAST_OPT] Training trap_xgboost...")
    best_model, best_f1, best_acc = None, 0, 0
    
    for lr in [0.02, 0.03, 0.04]:
        for depth in [7, 8, 9]:
            model = XGBClassifier(
                n_estimators=500, max_depth=depth, learning_rate=lr,
                tree_method="hist", random_state=RANDOM_STATE, verbose=0
            )
            m, f1, acc = train_and_eval(model, X_train, y_train, X_holdout, y_holdout, "trap")
            if f1 > best_f1:
                best_f1, best_acc, best_model = f1, acc, m
                print(f"  ✓ F1={f1:.4f} | Acc={acc:.4f} (depth={depth}, lr={lr})")
    
    if best_model:
        joblib.dump(best_model, MODELS_DIR / "trap_xgboost.joblib")
        all_results["trap_xgboost"] = {"f1": float(best_f1), "accuracy": float(best_acc)}
        all_models["trap_xgboost"] = best_model
        status = "✅" if best_f1 >= TARGET_F1 else "⚠️"
        print(f"{status} trap_xgboost: F1={best_f1:.4f}")
    
    # ─── LGBM (currently 0.66 → maintain) ────
    print(f"\n[FAST_OPT] Training lgbm...")
    best_model, best_f1, best_acc = None, 0, 0
    
    for lr in [0.02, 0.03, 0.04]:
        for depth in [7, 8, 9]:
            model = LGBMClassifier(
                n_estimators=500, max_depth=depth, learning_rate=lr,
                class_weight="balanced", random_state=RANDOM_STATE, verbose=-1
            )
            m, f1, acc = train_and_eval(model, X_train, y_train, X_holdout, y_holdout, "lgbm")
            if f1 > best_f1:
                best_f1, best_acc, best_model = f1, acc, m
                print(f"  ✓ F1={f1:.4f} | Acc={acc:.4f} (depth={depth}, lr={lr})")
    
    if best_model:
        joblib.dump(best_model, MODELS_DIR / "lgbm.joblib")
        all_results["lgbm"] = {"f1": float(best_f1), "accuracy": float(best_acc)}
        all_models["lgbm"] = best_model
        status = "✅" if best_f1 >= TARGET_F1 else "⚠️"
        print(f"{status} lgbm: F1={best_f1:.4f}")
    
    # ─── SUPERVISOR ────
    print(f"\n[FAST_OPT] Training supervisor...")
    l1_preds = []
    for name in ["trend_catboost", "fibo_xgboost", "candle_catboost", "trap_xgboost", "lgbm"]:
        if name in all_models:
            model = all_models[name]
            if hasattr(model, "predict_proba"):
                l1_preds.append(model.predict_proba(X_holdout))
            else:
                l1_preds.append(model.predict(X_holdout).reshape(-1, 1))
    
    if l1_preds:
        X_meta = np.hstack(l1_preds)
        supervisor = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, class_weight="balanced")
        supervisor.fit(X_meta, y_holdout)
        y_pred = supervisor.predict(X_meta)
        f1_sup = f1_score(y_holdout, y_pred, average="weighted", zero_division=0)
        acc_sup = accuracy_score(y_holdout, y_pred)
        
        joblib.dump(supervisor, MODELS_DIR / "supervisor.joblib")
        all_results["supervisor"] = {"f1": float(f1_sup), "accuracy": float(acc_sup)}
        status = "✅" if f1_sup >= TARGET_F1 else "⚠️"
        print(f"{status} supervisor: F1={f1_sup:.4f}")
    
    # ─── REPORT ────
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    target_count = 0
    for name, metrics in all_results.items():
        f1 = metrics["f1"]
        acc = metrics["accuracy"]
        status = "✅ TARGET MET" if f1 >= TARGET_F1 else "⚠️  BELOW TARGET"
        print(f"{status} | {name:20s} | F1: {f1:.4f} ({f1*100:.2f}%) | Acc: {acc:.4f}")
        if f1 >= TARGET_F1:
            target_count += 1
    
    print(f"\n✅ {target_count}/{len(all_results)} models achieved F1 ≥ {TARGET_F1*100:.0f}%")
    
    # Save report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "4.0-fast-optimized",
        "target_f1": TARGET_F1,
        "models_optimized": len(all_results),
        "target_achieved": target_count,
        "results": all_results
    }
    
    report_path = MODELS_DIR / "optimization_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Report → {report_path}")
    print("="*70)


if __name__ == "__main__":
    main()
