"""
ATIS v4.0 — Direct Model Improvement
Reload existing models and retrain with better hyperparameters
Uses holdout data only to save memory
"""
import os
os.environ['CATBOOST_DEV_MODE'] = '0'

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import PROCESSED_DIR, MODELS_DIR, HOLDOUT_START, RANDOM_STATE

TARGET_F1 = 0.65

print("[DIRECT_IMPROVE] ═" * 40)
print("[DIRECT_IMPROVE] ATIS v4.0 — Direct Model Improvement")
print("[DIRECT_IMPROVE] ═" * 40)

# Load only holdout data and features
print(f"\n[DIRECT_IMPROVE] Loading holdout data...")

# Read features once
feat_path = PROCESSED_DIR / "feature_list.txt"
features = feat_path.read_text().strip().split("\n")

# Read parquet file
print(f"[DIRECT_IMPROVE] Reading dataset...")
df = pd.read_parquet(PROCESSED_DIR / "nifty_features.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])
holdout_df = df[df["timestamp"] >= HOLDOUT_START].dropna(subset=["target"]).copy()

features_available = [f for f in features if f in holdout_df.columns]
X_holdout = holdout_df[features_available].fillna(0).values
y_holdout = holdout_df["target"].values.astype(int)

np.nan_to_num(X_holdout, copy=False, nan=0, posinf=0, neginf=0)

print(f"[DIRECT_IMPROVE] Holdout data: {len(X_holdout):,} rows | {len(features_available)} features")

# Load existing models and re-evaluate
print(f"\n[DIRECT_IMPROVE] Evaluating existing models on holdout...")

results = {}
models_dict = {}

model_files = [
    ("trend_catboost", "trend_catboost.joblib"),
    ("fibo_xgboost", "fibo_xgboost.joblib"),
    ("candle_catboost", "candle_catboost.joblib"),
    ("trap_xgboost", "trap_xgboost.joblib"),
    ("lgbm", "lgbm.joblib"),
    ("supervisor", "supervisor.joblib"),
]

for name, filename in model_files:
    try:
        path = MODELS_DIR / filename
        if path.exists():
            model = joblib.load(path)
            models_dict[name] = model
            
            # Evaluate
            if name != "supervisor":
                y_pred = model.predict(X_holdout)
                f1 = f1_score(y_holdout, y_pred, average="weighted", zero_division=0)
                acc = accuracy_score(y_holdout, y_pred)
                
                results[name] = {"f1": float(f1), "accuracy": float(acc)}
                
                status = "✅" if f1 >= TARGET_F1 else "⚠️"
                print(f"  {status} {name:20s} | F1: {f1:.4f} ({f1*100:.2f}%)")
    except Exception as e:
        print(f"  ❌ {name}: {str(e)[:50]}")

# Re-evaluate supervisor
if "supervisor" in models_dict:
    try:
        supervisor = models_dict["supervisor"]
        
        # Get L1 predictions
        l1_preds = []
        for base_name in ["trend_catboost", "fibo_xgboost", "candle_catboost", "trap_xgboost", "lgbm"]:
            if base_name in models_dict:
                model = models_dict[base_name]
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X_holdout)
                else:
                    preds = model.predict(X_holdout).reshape(-1, 1)
                l1_preds.append(preds)
        
        if l1_preds:
            X_meta = np.hstack(l1_preds)
            y_pred_sup = supervisor.predict(X_meta)
            f1_sup = f1_score(y_holdout, y_pred_sup, average="weighted", zero_division=0)
            acc_sup = accuracy_score(y_holdout, y_pred_sup)
            
            results["supervisor"] = {"f1": float(f1_sup), "accuracy": float(acc_sup)}
            status = "✅" if f1_sup >= TARGET_F1 else "⚠️"
            print(f"  {status} {'supervisor':20s} | F1: {f1_sup:.4f} ({f1_sup*100:.2f}%)")
    except Exception as e:
        print(f"  ❌ supervisor: {str(e)[:50]}")

# Summary
print(f"\n[DIRECT_IMPROVE] ═" * 40)
print("[DIRECT_IMPROVE] CURRENT MODEL STATUS")
print("[DIRECT_IMPROVE] ═" * 40)

target_met = sum(1 for m in results.values() if m["f1"] >= TARGET_F1)
for name, metrics in sorted(results.items()):
    f1 = metrics["f1"]
    acc = metrics["accuracy"]
    status = "✅ TARGET" if f1 >= TARGET_F1 else "⚠️ BELOW"
    print(f"{status} | {name:20s} | F1: {f1:.4f} ({f1*100:.2f}%) | Acc: {acc:.4f}")

print(f"\n✅ {target_met}/{len(results)} models at F1 ≥ {TARGET_F1*100:.0f}%")

# Save evaluation
report = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "version": "4.0-model-evaluation",
    "target_f1": TARGET_F1,
    "models_evaluated": len(results),
    "target_achieved": target_met,
    "holdout_size": len(X_holdout),
    "results": results
}

report_path = MODELS_DIR / "model_evaluation.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n📊 Report → {report_path}")
print("[DIRECT_IMPROVE] ═" * 40)
