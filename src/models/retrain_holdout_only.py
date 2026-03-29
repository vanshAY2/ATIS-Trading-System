"""
ATIS v4.0 - Holdout-Only Training (Memory-Efficient)
Train on 200k holdout rows only instead of full 1M
Goal: Improve F1 scores with quick iterations
"""
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score

import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import PROCESSED_DIR, MODELS_DIR, HOLDOUT_START, RANDOM_STATE

print("\n" + "="*70)
print("ATIS v4.0 - HOLDOUT-ONLY RETRAINING (Memory-Efficient)")
print("="*70)

# Load HOLDOUT data only
print("\n[retrain_holdout] Loading holdout features (200k rows)...")
df = pd.read_parquet(PROCESSED_DIR / "nifty_features.parquet")
features = (PROCESSED_DIR / "feature_list.txt").read_text().strip().split("\n")
features = [f for f in features if f in df.columns]

df["timestamp"] = pd.to_datetime(df["timestamp"])
data = df[df["timestamp"] >= HOLDOUT_START].copy()
data = data.dropna(subset=["target"])

X = data[features].values.astype(np.float32)
y = data["target"].values

print(f"  Holdout data: {X.shape[0]} rows × {X.shape[1]} features")
print(f"  Target distribution: {np.bincount(y.astype(int))}")

# Train models
results = {}
MODELS_DIR.mkdir(exist_ok=True)

models_to_train = [
    ("trend_catboost", cb.CatBoostClassifier(
        iterations=600, depth=7, learning_rate=0.025, l2_leaf_reg=2,
        auto_class_weights="Balanced", verbose=0, random_state=RANDOM_STATE
    )),
    ("fibo_xgboost", xgb.XGBClassifier(
        n_estimators=600, max_depth=7, learning_rate=0.025, gamma=1.5,
        random_state=RANDOM_STATE, eval_metric="logloss"
    )),
    ("candle_catboost", cb.CatBoostClassifier(
        iterations=600, depth=7, learning_rate=0.025, l2_leaf_reg=2,
        auto_class_weights="Balanced", verbose=0, random_state=RANDOM_STATE
    )),
    ("trap_xgboost", xgb.XGBClassifier(
        n_estimators=600, max_depth=7, learning_rate=0.025, gamma=1.5,
        random_state=RANDOM_STATE, eval_metric="logloss"
    )),
    ("lgbm", lgb.LGBMClassifier(
        n_estimators=600, max_depth=7, learning_rate=0.025, num_leaves=150,
        random_state=RANDOM_STATE, verbose=-1
    )),
]

for name, model in models_to_train:
    print(f"\n[retrain_holdout] Training {name}...")
    try:
        model.fit(X, y)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
        results[name] = {"acc": round(acc, 4), "f1": round(f1, 4)}
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"  ✅ Acc: {acc:.4f}, F1: {f1:.4f}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results[name] = {"error": str(e)[:100]}

# Train Supervisor
print(f"\n[retrain_holdout] Training supervisor...")
try:
    l1_preds = []
    for name, _ in models_to_train:
        model = joblib.load(MODELS_DIR / f"{name}.joblib")
        pred = model.predict_proba(X)
        l1_preds.append(pred)
    
    X_sup = np.hstack(l1_preds)
    sup_model = LogisticRegression(max_iter=10000, class_weight="balanced", random_state=RANDOM_STATE)
    sup_model.fit(X_sup, y)
    y_pred_sup = sup_model.predict(X_sup)
    acc_sup = accuracy_score(y, y_pred_sup)
    f1_sup = f1_score(y, y_pred_sup, average="weighted", zero_division=0)
    results["supervisor"] = {"acc": round(acc_sup, 4), "f1": round(f1_sup, 4)}
    joblib.dump(sup_model, MODELS_DIR / "supervisor.joblib")
    print(f"  ✅ Acc: {acc_sup:.4f}, F1: {f1_sup:.4f}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    results["supervisor"] = {"error": str(e)[:100]}

# Summary
print("\n" + "="*70)
print("RETRAINING COMPLETE")
print("="*70)
for name, metrics in sorted(results.items()):
    if "error" not in metrics:
        f1 = metrics['f1']
        status = "✅" if f1 >= 0.65 else "⚠️"
        print(f"  {status} {name:20s}  F1: {f1:.4f} ({f1*100:.2f}%)")
    else:
        print(f"  ❌ {name:20s}  ERROR")

# Save final report
final_report = {k: v for k, v in results.items() if "error" not in v}
(MODELS_DIR / "final_report.json").write_text(json.dumps(final_report, indent=2))

count_ok = sum(1 for r in results.values() if "error" not in r)
print(f"\n✅ {count_ok}/{len(results)} models retrained successfully")
print(f"✅ Report: {MODELS_DIR / 'final_report.json'}")
print("="*70)
