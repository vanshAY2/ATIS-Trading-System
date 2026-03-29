"""
ATIS v4.0 — Final Model Optimization
Trains models on 72 optimized features for 65%+ F1
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import PROCESSED_DIR, MODELS_DIR, HOLDOUT_START, RANDOM_STATE

TARGET_F1 = 0.65

print("\n" + "="*70)
print("ATIS v4.0 — FINAL OPTIMIZED TRAINING")
print("72 Features | Target F1 ≥ 65% | Full Dataset")
print("="*70)

# Load ALL data with 72 features
print("\n[FINAL] Loading all data (72 features)...")
df = pd.read_parquet(PROCESSED_DIR / "nifty_features.parquet")

feat_path = PROCESSED_DIR / "feature_list.txt"
features = feat_path.read_text().strip().split("\n")
features = [f for f in features if f in df.columns]

print(f"[FINAL] Loaded: {len(df):,} rows × {len(features)} features")

# Split data
df["timestamp"] = pd.to_datetime(df["timestamp"])
holdout_mask = df["timestamp"] >= HOLDOUT_START
train_df = df[~holdout_mask].dropna(subset=["target"]).copy()
holdout_df = df[holdout_mask].dropna(subset=["target"]).copy()

X_train_full = train_df[features].fillna(0).values
y_train_full = train_df["target"].values.astype(int)
X_holdout = holdout_df[features].fillna(0).values
y_holdout = holdout_df["target"].values.astype(int)

# Fix infinities
for X in [X_train_full, X_holdout]:
    np.nan_to_num(X, copy=False, nan=0, posinf=0, neginf=0)

print(f"[FINAL] Train: {len(X_train_full):,} | Holdout: {len(X_holdout):,}")

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    stratify=y_train_full,
    random_state=RANDOM_STATE
)

results = {}
all_models = {}

# ─── OPTIMIZED HYPERPARAMETERS (FOR 72 FEATURES) ────

hparams = {
    "trend_catboost": {
        "model_type": "CatBoost",
        "iterations": 800,
        "depth": 8,
        "learning_rate": 0.02,
        "l2_leaf_reg": 3,
        "min_data_in_leaf": 5
    },
    "fibo_xgboost": {
        "model_type": "XGBoost",
        "n_estimators": 800,
        "max_depth": 8,
        "learning_rate": 0.02,
        "gamma": 2,
        "min_child_weight": 2
    },
    "candle_catboost": {
        "model_type": "CatBoost",
        "iterations": 750,
        "depth": 8,
        "learning_rate": 0.025,
        "l2_leaf_reg": 2,
        "min_data_in_leaf": 5
    },
    "trap_xgboost": {
        "model_type": "XGBoost",
        "n_estimators": 750,
        "max_depth": 8,
        "learning_rate": 0.025,
        "gamma": 2,
        "min_child_weight": 2
    },
    "lgbm": {
        "model_type": "LightGBM",
        "n_estimators": 800,
        "max_depth": 8,
        "learning_rate": 0.02,
        "num_leaves": 200,
        "min_child_samples": 10
    }
}

print(f"\n[FINAL]  Training 5 base models (optimized)...")

for name, params in hparams.items():
    print(f"\n  [{name}] Training {params['model_type']}...")
    model_type = params.pop("model_type")
    
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
                verbose=0,
                subsample=0.9,
                colsample_bytree=0.9
            )
        else:  # LightGBM
            model = LGBMClassifier(
                **params,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                verbose=-1,
                subsample=0.9,
                colsample_bytree=0.9
            )
        
        # Train on FULL train data
        model.fit(X_train_full, y_train_full, verbose=0)
        
        # Evaluate on holdout
        y_pred = model.predict(X_holdout)
        f1 = f1_score(y_holdout, y_pred, average="weighted", zero_division=0)
        acc = accuracy_score(y_holdout, y_pred)
        
        # Also check val performance
        y_val_pred = model.predict(X_val)
        f1_val = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)
        
        # Save model
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        
        results[name] = {
            "f1_holdout": float(f1),
            "acc_holdout": float(acc),
            "f1_val": float(f1_val)
        }
        all_models[name] = model
        
        status = "✅ TARGET MET" if f1 >= TARGET_F1 else "⚠️ BELOW"
        print(f"  {status} | F1_holdout: {f1:.4f} ({f1*100:.2f}%) | F1_val: {f1_val:.4f} | Acc: {acc:.4f}")
        
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)[:60]}")

# ─── TRAIN SUPERVISOR (L3 META-LEARNER) ────
print(f"\n[FINAL] Training Supervisor Meta-Learner...")

l1_preds_val = []
l1_preds_holdout = []

for name in ["trend_catboost", "fibo_xgboost", "candle_catboost", "trap_xgboost", "lgbm"]:
    if name in all_models:
        model = all_models[name]
        if hasattr(model, "predict_proba"):
            l1_preds_val.append(model.predict_proba(X_val))
            l1_preds_holdout.append(model.predict_proba(X_holdout))
        else:
            l1_preds_val.append(model.predict(X_val).reshape(-1, 1))
            l1_preds_holdout.append(model.predict(X_holdout).reshape(-1, 1))

if l1_preds_val:
    X_meta_val = np.hstack(l1_preds_val)
    X_meta_holdout = np.hstack(l1_preds_holdout)
    
    supervisor = LogisticRegression(
        max_iter=10000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        solver="lbfgs"
    )
    supervisor.fit(X_meta_val, y_val)
    
    y_pred_hold = supervisor.predict(X_meta_holdout)
    f1_sup = f1_score(y_holdout, y_pred_hold, average="weighted", zero_division=0)
    acc_sup = accuracy_score(y_holdout, y_pred_hold)
    
    y_pred_val = supervisor.predict(X_meta_val)
    f1_sup_val = f1_score(y_val, y_pred_val, average="weighted", zero_division=0)
    
    # Save
    path = MODELS_DIR / "supervisor.joblib"
    joblib.dump(supervisor, path)
    
    results["supervisor"] = {
        "f1_holdout": float(f1_sup),
        "acc_holdout": float(acc_sup),
        "f1_val": float(f1_sup_val)
    }
    
    status = "✅ TARGET MET" if f1_sup >= TARGET_F1 else "⚠️ BELOW"
    print(f"  {status} | F1_holdout: {f1_sup:.4f} ({f1_sup*100:.2f}%) | F1_val: {f1_sup_val:.4f} | Acc: {acc_sup:.4f}")

# ─── FINAL SUMMARY ────
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

target_met = 0
for name, metrics in sorted(results.items()):
    f1_h = metrics["f1_holdout"]
    f1_v = metrics["f1_val"]
    acc = metrics["acc_holdout"]
    status = "✅ TARGET MET" if f1_h >= TARGET_F1 else "⚠️ BELOW"
    print(f"{status} | {name:20s} | F1_holdout: {f1_h:.4f} | F1_val: {f1_v:.4f} | Acc: {acc:.4f}")
    if f1_h >= TARGET_F1:
        target_met += 1

print(f"\n✅ {target_met}/{len(results)} models achieved F1 ≥ {TARGET_F1*100:.0f}%")

# Save final report
report = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "version": "4.0-final-optimized",
    "features": len(features),
    "target_f1": TARGET_F1,
    "train_size": len(X_train_full),
    "val_size": len(X_val),
    "holdout_size": len(X_holdout),
    "models_trained": len(results),
    "target_achieved": target_met,
    "results": results
}

report_path = MODELS_DIR / "final_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n📊 Final Report → {report_path}")
print("="*70)
print("✅ TRAINING COMPLETE - Ready for backtesting!")
print("="*70)
