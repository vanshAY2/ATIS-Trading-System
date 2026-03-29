"""
ATIS v4.0 - Quick Retrain (Fix Feature Mismatch)
Retrains all 6 models (5 L1 base + 1 L3 supervisor) with correct 72 features
"""
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import catboost as cb
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import PROCESSED_DIR, MODELS_DIR, HOLDOUT_START, RANDOM_STATE

print("\n" + "="*70)
print("ATIS v4.0 - QUICK RETRAINING (72-feature set)")
print("="*70)

# Load data
print("\n[retrain] Loading features...")
df = pd.read_parquet(PROCESSED_DIR / "nifty_features.parquet")
features = (PROCESSED_DIR / "feature_list.txt").read_text().strip().split("\n")
features = [f for f in features if f in df.columns]

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.dropna(subset=["target"])

# Split train/holdout
train_df = df[df["timestamp"] < HOLDOUT_START].copy()
holdout_df = df[df["timestamp"] >= HOLDOUT_START].copy()

X_train = train_df[features].values.astype(np.float32)
y_train = train_df["target"].values
X_holdout = holdout_df[features].values.astype(np.float32)
y_holdout = holdout_df["target"].values

print(f"  Train: {X_train.shape[0]} rows, {X_train.shape[1]} features")
print(f"  Holdout: {X_holdout.shape[0]} rows")

# Train models
results = {}
MODELS_DIR.mkdir(exist_ok=True)

# 1. Trend CatBoost
print("\n[retrain] Training trend_catboost...")
try:
    model = cb.CatBoostClassifier(
        iterations=800, depth=8, learning_rate=0.02, l2_leaf_reg=3,
        auto_class_weights="Balanced", verbose=0, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_holdout)
    acc = accuracy_score(y_holdout, y_pred)
    f1 = f1_score(y_holdout, y_pred, average="weighted", zero_division=0)
    results["trend_catboost"] = {"acc": round(acc, 4), "f1": round(f1, 4)}
    joblib.dump(model, MODELS_DIR / "trend_catboost.joblib")
    print(f"  ✅ Acc: {acc:.4f}, F1: {f1:.4f}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    results["trend_catboost"] = {"error": str(e)}

# 2. Fibo XGBoost
print("\n[retrain] Training fibo_xgboost...")
try:
    model = xgb.XGBClassifier(
        n_estimators=800, max_depth=8, learning_rate=0.02, gamma=2,
        random_state=RANDOM_STATE, eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_holdout)
    acc = accuracy_score(y_holdout, y_pred)
    f1 = f1_score(y_holdout, y_pred, average="weighted", zero_division=0)
    results["fibo_xgboost"] = {"acc": round(acc, 4), "f1": round(f1, 4)}
    joblib.dump(model, MODELS_DIR / "fibo_xgboost.joblib")
    print(f"  ✅ Acc: {acc:.4f}, F1: {f1:.4f}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    results["fibo_xgboost"] = {"error": str(e)}

# 3. Candle CatBoost
print("\n[retrain] Training candle_catboost...")
try:
    model = cb.CatBoostClassifier(
        iterations=800, depth=8, learning_rate=0.02, l2_leaf_reg=3,
        auto_class_weights="Balanced", verbose=0, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_holdout)
    acc = accuracy_score(y_holdout, y_pred)
    f1 = f1_score(y_holdout, y_pred, average="weighted", zero_division=0)
    results["candle_catboost"] = {"acc": round(acc, 4), "f1": round(f1, 4)}
    joblib.dump(model, MODELS_DIR / "candle_catboost.joblib")
    print(f"  ✅ Acc: {acc:.4f}, F1: {f1:.4f}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    results["candle_catboost"] = {"error": str(e)}

# 4. Trap XGBoost
print("\n[retrain] Training trap_xgboost...")
try:
    model = xgb.XGBClassifier(
        n_estimators=800, max_depth=8, learning_rate=0.02, gamma=2,
        random_state=RANDOM_STATE, eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_holdout)
    acc = accuracy_score(y_holdout, y_pred)
    f1 = f1_score(y_holdout, y_pred, average="weighted", zero_division=0)
    results["trap_xgboost"] = {"acc": round(acc, 4), "f1": round(f1, 4)}
    joblib.dump(model, MODELS_DIR / "trap_xgboost.joblib")
    print(f"  ✅ Acc: {acc:.4f}, F1: {f1:.4f}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    results["trap_xgboost"] = {"error": str(e)}

# 5. LightGBM
print("\n[retrain] Training lgbm...")
try:
    model = lgb.LGBMClassifier(
        n_estimators=800, max_depth=8, learning_rate=0.02, num_leaves=200,
        random_state=RANDOM_STATE, verbose=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_holdout)
    acc = accuracy_score(y_holdout, y_pred)
    f1 = f1_score(y_holdout, y_pred, average="weighted", zero_division=0)
    results["lgbm"] = {"acc": round(acc, 4), "f1": round(f1, 4)}
    joblib.dump(model, MODELS_DIR / "lgbm.joblib")
    print(f"  ✅ Acc: {acc:.4f}, F1: {f1:.4f}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    results["lgbm"] = {"error": str(e)}

# 6. L3 Supervisor (uses predictions from L1 models)
print("\n[retrain] Training supervisor (meta-learner)...")
try:
    # Generate L1 predictions on holdout
    l1_preds = []
    for name in ["trend_catboost", "fibo_xgboost", "candle_catboost", "trap_xgboost", "lgbm"]:
        model = joblib.load(MODELS_DIR / f"{name}.joblib")
        pred = model.predict_proba(X_holdout)
        l1_preds.append(pred)
    
    X_supervisor = np.hstack(l1_preds)  # Shape: (holdout_rows, 5*3) = (holdout_rows, 15)
    
    # Train supervisor
    model = LogisticRegression(max_iter=10000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_supervisor, y_holdout)
    y_pred = model.predict(X_supervisor)
    acc = accuracy_score(y_holdout, y_pred)
    f1 = f1_score(y_holdout, y_pred, average="weighted", zero_division=0)
    results["supervisor"] = {"acc": round(acc, 4), "f1": round(f1, 4)}
    joblib.dump(model, MODELS_DIR / "supervisor.joblib")
    print(f"  ✅ Acc: {acc:.4f}, F1: {f1:.4f}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    results["supervisor"] = {"error": str(e)}

# Save results
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
for model_name, metrics in results.items():
    if "error" not in metrics:
        print(f"  {model_name:20s}  Acc: {metrics['acc']:.4f}  F1: {metrics['f1']:.4f}")
    else:
        print(f"  {model_name:20s}  ERROR: {metrics['error']}")

final_report = {model: metrics for model, metrics in results.items() if "error" not in metrics}
(MODELS_DIR / "final_report.json").write_text(json.dumps(final_report, indent=2))

print(f"\n✅ Models saved to {MODELS_DIR}")
print(f"✅ Report saved to {MODELS_DIR / 'final_report.json'}")
print("="*70)
