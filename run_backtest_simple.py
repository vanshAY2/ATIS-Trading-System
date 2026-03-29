"""
ATIS v4.0 - Simplified Backtest
Runs backtesting WITHOUT LSTM/Sentiment dependencies
Evaluates ensemble performance on 2024-2026 holdout data
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import PROCESSED_DIR, MODELS_DIR, HOLDOUT_START, TARGET_ACCURACY

print("\n" + "="*70)
print("ATIS v4.0 - SIMPLIFIED BACKTEST (No LSTM/Sentiment)")
print("="*70)

# Load holdout data
print("\n[backtest] Loading holdout features...")
df = pd.read_parquet(PROCESSED_DIR / "nifty_features.parquet")
features_txt = (PROCESSED_DIR / "feature_list.txt").read_text().strip().split("\n")
features = [f for f in features_txt if f in df.columns]

df["timestamp"] = pd.to_datetime(df["timestamp"])
holdout = df[df["timestamp"] >= HOLDOUT_START].copy()
holdout = holdout.dropna(subset=["target"])

X = holdout[features].values.astype(np.float32)
y = holdout["target"].values

print(f"  Holdout: {len(holdout)} rows, {len(features)} features")
print(f"  Target: {np.bincount(y.astype(int))}")

# Test L1 models individually
print("\n" + "-"*70)
print("  L1 BASE MODELS EVALUATION")
print("-"*70)

l1_results = {}
l1_models = {}

for name in ["trend_catboost", "fibo_xgboost", "candle_catboost", "trap_xgboost", "lgbm"]:
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        print(f"  ❌ {name:25s} NOT FOUND")
        continue
    
    try:
        model = joblib.load(path)
        l1_models[name] = model
        y_pred = model.predict(X)
        
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
        prec = precision_score(y, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y, y_pred, average="weighted", zero_division=0)
        
        l1_results[name] = {"acc": acc, "f1": f1, "prec": prec, "rec": rec}
        
        status = "✅" if f1 >= TARGET_ACCURACY else "⚠️"
        print(f"  {status} {name:20s}  F1: {f1:.4f}  Acc: {acc:.4f}  P: {prec:.4f}  R: {rec:.4f}")
    except Exception as e:
        print(f"  ❌ {name:20s}  ERROR: {str(e)[:50]}")

# Ensemble voting (L3 Supervisor simulation)
print("\n" + "-"*70)
print("  L3 ENSEMBLE (Majority Voting)")
print("-"*70)

if len(l1_models) >= 3:
    # Collect votes from L1 models
    all_preds = []
    for name in sorted(l1_models.keys()):
        pred = l1_models[name].predict(X)
        all_preds.append(pred)
    
    all_preds = np.array(all_preds)  # Shape: (5, num_samples)
    
    # Majority voting
    y_ensemble = np.apply_along_axis(lambda x: np.bincount(x.astype(int), minlength=3).argmax(), 0, all_preds)
    
    acc_ens = accuracy_score(y, y_ensemble)
    f1_ens = f1_score(y, y_ensemble, average="weighted", zero_division=0)
    prec_ens = precision_score(y, y_ensemble, average="weighted", zero_division=0)
    rec_ens = recall_score(y, y_ensemble, average="weighted", zero_division=0)
    
    status = "✅" if f1_ens >= TARGET_ACCURACY else "⚠️"
    print(f"  {status} {'ensemble_voting':20s}  F1: {f1_ens:.4f}  Acc: {acc_ens:.4f}  P: {prec_ens:.4f}  R: {rec_ens:.4f}")
    
    l1_results["ensemble_voting"] = {"acc": acc_ens, "f1": f1_ens, "prec": prec_ens, "rec": rec_ens}

# Summary statistics
print("\n" + "="*70)
print("BACKTEST SUMMARY")
print("="*70)

total_models = len(l1_results)
passing = sum(1 for r in l1_results.values() if r.get("f1", 0) >= TARGET_ACCURACY)
avg_f1 = np.mean([r.get("f1", 0) for r in l1_results.values()])

print(f"  Total Models: {total_models}")
print(f"  Passing (F1≥{TARGET_ACCURACY:.2f}): {passing}/{total_models}")
print(f"  Average F1: {avg_f1:.4f}")

# Save results
report = {}
for name, metrics in l1_results.items():
    report[name] = {k: round(v, 4) for k, v in metrics.items()}

(MODELS_DIR / "backtest_report.json").write_text(json.dumps(report, indent=2))
print(f"\n✅ Backtest report saved: {MODELS_DIR / 'backtest_report.json'}")
print("="*70)
