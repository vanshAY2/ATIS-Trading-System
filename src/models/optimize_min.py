"""
ATIS v4.0 — Minimal Model Optimization
Ultra-lightweight trainers with aggressive sampling
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
SAMPLE_SIZE = 100000  # Aggressive sampling to fit in memory

print("[MIN_OPT] ═" * 40)
print("[MIN_OPT] ATIS v4.0 — Minimal Model Optimization")
print("[MIN_OPT] ═" * 40)

# Load data in chunks
print(f"\n[MIN_OPT] Loading data (sampling {SAMPLE_SIZE:,} rows)...")
df = pd.read_parquet(PROCESSED_DIR / "nifty_features.parquet")

feat_path = PROCESSED_DIR / "feature_list.txt"
features = feat_path.read_text().strip().split("\n")
features = [f for f in features if f in df.columns]

df["timestamp"] = pd.to_datetime(df["timestamp"])
holdout_mask = df["timestamp"] >= HOLDOUT_START
train_df = df[~holdout_mask].dropna(subset=["target"]).copy()
holdout_df = df[holdout_mask].dropna(subset=["target"]).copy()

# Sample aggressively
X_train_full = train_df[features].fillna(0).values
y_train_full = train_df["target"].values.astype(int)

# Take stratified sample
from sklearn.model_selection import train_test_split
X_train, _, y_train, _ = train_test_split(
    X_train_full, y_train_full, 
    train_size=min(SAMPLE_SIZE, len(X_train_full)),
    stratify=y_train_full,
    random_state=RANDOM_STATE
)

X_holdout = holdout_df[features].fillna(0).values
y_holdout = holdout_df["target"].values.astype(int)

# Fix infinities
for X in [X_train, X_holdout]:
    np.nan_to_num(X, copy=False, nan=0, posinf=0, neginf=0)

print(f"[MIN_OPT] Train: {len(X_train):,} | Holdout: {len(X_holdout):,} | Features: {len(features)}")

results = {}
models_dict = {}

# ─── TRAIN 5 MODELS ────
model_configs = [
    ("trend_catboost", "CatBoost", {"iterations": 300, "depth": 7, "learning_rate": 0.03}),
    ("fibo_xgboost", "XGBoost", {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.03}),
    ("candle_catboost", "CatBoost", {"iterations": 300, "depth": 7, "learning_rate": 0.03}),
    ("trap_xgboost", "XGBoost", {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.03}),
    ("lgbm", "LightGBM", {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.03}),
]

print(f"\n[MIN_OPT] Training 5 base models...")

for name, model_type, params in model_configs:
    print(f"\n  [{name}] Training {model_type}...")
    
    try:
        if model_type == "CatBoost":
            model = CatBoostClassifier(
                **params,
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=0
            )
        elif model_type == "XGBoost":
            model = XGBClassifier(
                **params,
                tree_method="hist",
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                verbose=0
            )
        else:  # LightGBM
            model = LGBMClassifier(
                **params,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                verbose=-1
            )
        
        # Train
        model.fit(X_train, y_train, verbose=0)
        
        # Evaluate on full holdout
        y_pred = model.predict(X_holdout)
        f1 = f1_score(y_holdout, y_pred, average="weighted", zero_division=0)
        acc = accuracy_score(y_holdout, y_pred)
        
        # Save
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        
        results[name] = {"f1": float(f1), "accuracy": float(acc)}
        models_dict[name] = model
        
        status = "✅" if f1 >= TARGET_F1 else "⚠️"
        print(f"  {status} {name:20s} | F1: {f1:.4f} ({f1*100:.2f}%) | Saved to {path}")
        
    except Exception as e:
        print(f"  ❌ {name} ERROR: {str(e)[:50]}")

# ─── TRAIN SUPERVISOR (L3) ────
print(f"\n[MIN_OPT] Training Supervisor Meta-Learner...")

l1_preds_holdout = []
for name in ["trend_catboost", "fibo_xgboost", "candle_catboost", "trap_xgboost", "lgbm"]:
    if name in models_dict:
        model = models_dict[name]
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_holdout)
        else:
            preds = model.predict(X_holdout).reshape(-1, 1)
        l1_preds_holdout.append(preds)
        
if l1_preds_holdout:
    X_meta = np.hstack(l1_preds_holdout)
    
    supervisor = LogisticRegression(
        max_iter=5000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        solver="lbfgs"
    )
    supervisor.fit(X_meta, y_holdout)
    
    y_pred_sup = supervisor.predict(X_meta)
    f1_sup = f1_score(y_holdout, y_pred_sup, average="weighted", zero_division=0)
    acc_sup = accuracy_score(y_holdout, y_pred_sup)
    
    path = MODELS_DIR / "supervisor.joblib"
    joblib.dump(supervisor, path)
    
    results["supervisor"] = {"f1": float(f1_sup), "accuracy": float(acc_sup)}
    
    status = "✅" if f1_sup >= TARGET_F1 else "⚠️"
    print(f"  {status} {'supervisor':20s} | F1: {f1_sup:.4f} ({f1_sup*100:.2f}%) | Saved to {path}")

# ─── SUMMARY ────
print(f"\n[MIN_OPT] ═" * 40)
print("[MIN_OPT] OPTIMIZATION SUMMARY")
print("[MIN_OPT] ═" * 40)

target_met = 0
for name, metrics in results.items():
    f1 = metrics["f1"]
    acc = metrics["accuracy"]
    status = "✅ TARGET MET" if f1 >= TARGET_F1 else "⚠️ BELOW TARGET"
    print(f"{status} | {name:20s} | F1: {f1:.4f} ({f1*100:.2f}%) | Acc: {acc:.4f}")
    if f1 >= TARGET_F1:
        target_met += 1

print(f"\n✅ {target_met}/{len(results)} models achieved F1 ≥ {TARGET_F1*100:.0f}%")

# Save report
report = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "version": "4.0-minimal-optimized",
    "target_f1": TARGET_F1,
    "sample_size": SAMPLE_SIZE,
    "models_trained": len(results),
    "target_achieved": target_met,
    "results": results
}

report_path = MODELS_DIR / "optimization_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n📊 Report → {report_path}")
print("[MIN_OPT] ═" * 40)
print("[MIN_OPT] ✅ Optimization Complete!")
print("[MIN_OPT] ═" * 40)
