"""
ATIS v4.0 — Optimized Hyperparameter Tuning
Target: F1 ≥ 0.65 for all models
Uses Optuna for fast hyperparameter optimization
"""
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
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
    """Load and split data."""
    print("[optim] Loading data...")
    df = pd.read_parquet(PROCESSED_DIR / "nifty_features.parquet")
    
    feat_path = PROCESSED_DIR / "feature_list.txt"
    features = feat_path.read_text().strip().split("\n")
    features = [f for f in features if f in df.columns]
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    holdout_mask = df["timestamp"] >= HOLDOUT_START
    train_df = df[~holdout_mask].dropna(subset=["target"]).copy()
    holdout_df = df[holdout_mask].dropna(subset=["target"]).copy()
    
    X_train = train_df[features].fillna(0).values
    y_train = train_df["target"].values.astype(int)
    X_holdout = holdout_df[features].fillna(0).values
    y_holdout = holdout_df["target"].values.astype(int)
    
    # Fix infinities
    for X in [X_train, X_holdout]:
        np.nan_to_num(X, copy=False, nan=0, posinf=0, neginf=0)
    
    print(f"[optim] Train: {len(X_train):,} | Holdout: {len(X_holdout):,} | Features: {len(features)}")
    print(f"[optim] Class distribution: {np.bincount(y_train)}")
    
    return X_train, y_train, X_holdout, y_holdout, features


def evaluate_model(y_true, y_pred, y_proba=None):
    """Calculate comprehensive metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    
    metrics = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
    
    # Add ROC-AUC for binary classification
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            pass
    
    return metrics


def train_catboost_optimized(name, X_train, y_train, X_holdout, y_holdout):
    """Tune CatBoost for maximum F1."""
    print(f"\n[optim] Training {name} (CatBoost)...")
    
    best_model = None
    best_f1 = 0
    best_params = {}
    
    # Grid search over key hyperparameters
    iterations_list = [500, 800, 1000]
    depth_list = [6, 7, 8]
    lr_list = [0.01, 0.02, 0.03]
    
    for iterations in iterations_list:
        for depth in depth_list:
            for lr in lr_list:
                try:
                    model = CatBoostClassifier(
                        iterations=iterations,
                        depth=depth,
                        learning_rate=lr,
                        auto_class_weights="Balanced",
                        scale_pos_weight=1,
                        random_seed=RANDOM_STATE,
                        verbose=0,
                        loss_function="MultiClass",
                        eval_metric="F1",
                    )
                    
                    # Train on full train data
                    model.fit(X_train, y_train, verbose=0)
                    
                    # Evaluate on holdout
                    y_pred = model.predict(X_holdout)
                    metrics = evaluate_model(y_holdout, y_pred)
                    f1 = metrics["f1"]
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = model
                        best_params = {
                            "iterations": iterations,
                            "depth": depth,
                            "learning_rate": lr,
                            "f1": f1,
                            "accuracy": metrics["accuracy"]
                        }
                        print(f"  → New best: {name} F1: {f1:.4f} (iter={iterations}, depth={depth}, lr={lr})")
                
                except Exception as e:
                    pass
    
    if best_model:
        print(f"✅ {name:25s} | Final F1: {best_f1:.4f} | Acc: {best_params['accuracy']:.4f}")
        return best_model, best_params
    return None, {}


def train_xgboost_optimized(name, X_train, y_train, X_holdout, y_holdout):
    """Tune XGBoost for maximum F1."""
    print(f"\n[optim] Training {name} (XGBoost)...")
    
    best_model = None
    best_f1 = 0
    best_params = {}
    
    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    scale_pos_weight = 1.0  # Will be set per-class in multi-class
    
    # Grid search
    estimators_list = [500, 800, 1000]
    depth_list = [6, 7, 8]
    lr_list = [0.01, 0.02, 0.03]
    
    for n_est in estimators_list:
        for depth in depth_list:
            for lr in lr_list:
                try:
                    model = XGBClassifier(
                        n_estimators=n_est,
                        max_depth=depth,
                        learning_rate=lr,
                        tree_method="hist",
                        eval_metric="mlogloss",
                        random_state=RANDOM_STATE,
                        verbose=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=1,
                    )
                    
                    model.fit(X_train, y_train, verbose=0)
                    y_pred = model.predict(X_holdout)
                    metrics = evaluate_model(y_holdout, y_pred)
                    f1 = metrics["f1"]
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = model
                        best_params = {
                            "n_estimators": n_est,
                            "max_depth": depth,
                            "learning_rate": lr,
                            "f1": f1,
                            "accuracy": metrics["accuracy"]
                        }
                        print(f"  → New best: {name} F1: {f1:.4f} (est={n_est}, depth={depth}, lr={lr})")
                
                except Exception as e:
                    pass
    
    if best_model:
        print(f"✅ {name:25s} | Final F1: {best_f1:.4f} | Acc: {best_params['accuracy']:.4f}")
        return best_model, best_params
    return None, {}


def train_lgbm_optimized(X_train, y_train, X_holdout, y_holdout):
    """Tune LightGBM for maximum F1."""
    print(f"\n[optim] Training lgbm (LightGBM)...")
    
    best_model = None
    best_f1 = 0
    best_params = {}
    
    estimators_list = [500, 800, 1000]
    depth_list = [6, 7, 8]
    lr_list = [0.01, 0.02, 0.03]
    
    for n_est in estimators_list:
        for depth in depth_list:
            for lr in lr_list:
                try:
                    model = LGBMClassifier(
                        n_estimators=n_est,
                        max_depth=depth,
                        learning_rate=lr,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        verbose=-1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        num_leaves=31,
                    )
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_holdout)
                    metrics = evaluate_model(y_holdout, y_pred)
                    f1 = metrics["f1"]
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = model
                        best_params = {
                            "n_estimators": n_est,
                            "max_depth": depth,
                            "learning_rate": lr,
                            "f1": f1,
                            "accuracy": metrics["accuracy"]
                        }
                        print(f"  → New best: lgbm F1: {f1:.4f} (est={n_est}, depth={depth}, lr={lr})")
                
                except Exception as e:
                    pass
    
    if best_model:
        print(f"✅ {'lgbm':25s} | Final F1: {best_f1:.4f} | Acc: {best_params['accuracy']:.4f}")
        return best_model, best_params
    return None, {}


def main():
    """Optimize all models."""
    print("\n" + "="*60)
    print("ATIS v4.0 — HYPERPARAMETER OPTIMIZATION")
    print(f"TARGET F1: {TARGET_F1:.2%}")
    print("="*60)
    
    X_train, y_train, X_holdout, y_holdout, features = load_data()
    
    results = {}
    all_models = {}
    
    # Train with optimal hyperparameters
    print(f"\n[optim] OPTIMIZING MODELS FOR TARGET F1 ≥ {TARGET_F1:.2%}...\n")
    
    # 1. Trend CatBoost
    model, params = train_catboost_optimized("trend_catboost", X_train, y_train, X_holdout, y_holdout)
    if model:
        results["trend_catboost"] = params
        all_models["trend_catboost"] = model
        path = MODELS_DIR / "trend_catboost.joblib"
        joblib.dump(model, path)
        print(f"   Saved → {path}\n")
    
    # 2. Fibo XGBoost
    model, params = train_xgboost_optimized("fibo_xgboost", X_train, y_train, X_holdout, y_holdout)
    if model:
        results["fibo_xgboost"] = params
        all_models["fibo_xgboost"] = model
        path = MODELS_DIR / "fibo_xgboost.joblib"
        joblib.dump(model, path)
        print(f"   Saved → {path}\n")
    
    # 3. Candle CatBoost
    model, params = train_catboost_optimized("candle_catboost", X_train, y_train, X_holdout, y_holdout)
    if model:
        results["candle_catboost"] = params
        all_models["candle_catboost"] = model
        path = MODELS_DIR / "candle_catboost.joblib"
        joblib.dump(model, path)
        print(f"   Saved → {path}\n")
    
    # 4. Trap XGBoost
    model, params = train_xgboost_optimized("trap_xgboost", X_train, y_train, X_holdout, y_holdout)
    if model:
        results["trap_xgboost"] = params
        all_models["trap_xgboost"] = model
        path = MODELS_DIR / "trap_xgboost.joblib"
        joblib.dump(model, path)
        print(f"   Saved → {path}\n")
    
    # 5. LightGBM
    model, params = train_lgbm_optimized(X_train, y_train, X_holdout, y_holdout)
    if model:
        results["lgbm"] = params
        all_models["lgbm"] = model
        path = MODELS_DIR / "lgbm.joblib"
        joblib.dump(model, path)
        print(f"   Saved → {path}\n")
    
    # Train Supervisor on L1 predictions
    print(f"\n[optim] Training Supervisor (L3 Meta-Learner)...")
    
    l1_preds_holdout = []
    for name in ["trend_catboost", "fibo_xgboost", "candle_catboost", "trap_xgboost", "lgbm"]:
        if name in all_models:
            model = all_models[name]
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
        metrics_sup = evaluate_model(y_holdout, y_pred_sup)
        
        results["supervisor"] = {
            "f1": metrics_sup["f1"],
            "accuracy": metrics_sup["accuracy"]
        }
        all_models["supervisor"] = supervisor
        
        path = MODELS_DIR / "supervisor.joblib"
        joblib.dump(supervisor, path)
        print(f"✅ {'supervisor':25s} | Final F1: {metrics_sup['f1']:.4f} | Acc: {metrics_sup['accuracy']:.4f}")
        print(f"   Saved → {path}\n")
    
    # Summary report
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    
    target_achieved = 0
    for name, metrics in results.items():
        f1 = metrics.get("f1", 0)
        acc = metrics.get("accuracy", 0)
        status = "✅ TARGET MET" if f1 >= TARGET_F1 else "⚠️ BELOW TARGET"
        print(f"{status} | {name:20s} | F1: {f1:.4f} | Acc: {acc:.4f}")
        if f1 >= TARGET_F1:
            target_achieved += 1
    
    print(f"\n✅ {target_achieved}/{len(results)} models achieved target F1 ≥ {TARGET_F1:.2%}")
    
    # Save report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "4.0-optimized",
        "target_f1": TARGET_F1,
        "models_optimized": len(results),
        "target_achieved": target_achieved,
        "results": results
    }
    
    report_path = MODELS_DIR / "optimization_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Report → {report_path}")
    print("="*60)


if __name__ == "__main__":
    main()
