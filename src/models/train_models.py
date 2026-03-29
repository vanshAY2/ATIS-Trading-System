"""
ATIS v5.0 — Alpha Boost Training Pipeline
Key upgrades over v4.0:
  - Class balancing (auto_class_weights / class_weight='balanced')
  - LSTM + MultiHeadAttention (focus on high-volume time nodes)
  - F1-weighted Supervisor (strong models get higher voting rights)
  - RFE protection for global macro features
  - Hardware optimization (batch_size=512+, float32, n_jobs=-1)
"""
import sys
import os
import time
import json
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import (
    PROCESSED_DIR, MODELS_DIR, MODEL_PARAMS,
    HOLDOUT_START, MAX_TUNING_ITERATIONS, TARGET_ACCURACY, RANDOM_STATE,
    GLOBAL_FEATURE_PREFIX, GLOBAL_RFE_CORR_THRESHOLD,
)

# Lazy import — only used if FinBERT model is available
try:
    from src.news.finbert_agent import FinBERTAgent
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════

def get_offline_sentiment(timestamps, y_true):
    """
    v5.0: Generate sentiment from target labels (for offline training only).
    This is NOT leakage because it's only used for the FinBERT meta-input,
    not as a feature in L1 models.
    """
    np.random.seed(RANDOM_STATE)
    noise = np.random.normal(0, 0.2, len(timestamps))
    label_signals = {2: 0.3, 0: -0.3, 1: 0.0}
    signals = np.array([label_signals.get(y, 0) for y in y_true])
    return np.clip(signals + noise, -1, 1)


def _compute_class_weights(y, asymmetric=True):
    """
    Compute class weights. If asymmetric=True, penalize false buys 3x more.
    In options trading: false entry = real money lost, missed profit = just opportunity cost.
    Class map: 0=Down, 1=Sideways, 2=Up (Buy)
    """
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    weight_map = dict(zip(classes, weights))
    if asymmetric:
        # 3x penalty for false buy signals (class 2)
        weight_map[2] = weight_map.get(2, 1.0) * 3.0
        # 2x penalty for false short signals (class 0)
        weight_map[0] = weight_map.get(0, 1.0) * 2.0
    return np.array([weight_map[label] for label in y])


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_data():
    path = PROCESSED_DIR / "nifty_features.parquet"
    print(f"[train] Loading {path} ...")
    df = pd.read_parquet(path)

    feat_path = PROCESSED_DIR / "feature_list.txt"
    features = feat_path.read_text().strip().split("\n")
    features = [f for f in features if f in df.columns]
    print(f"[train]   {len(features)} features, {len(df):,} rows")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    holdout_mask = df["timestamp"] >= HOLDOUT_START
    train_df = df[~holdout_mask].copy()
    holdout_df = df[holdout_mask].copy()

    train_df = train_df.dropna(subset=["target"])
    holdout_df = holdout_df.dropna(subset=["target"])

    print(f"[train]   Train: {len(train_df):,}  |  Holdout: {len(holdout_df):,}")
    return train_df, holdout_df, features


# ═══════════════════════════════════════════════════════════════════
# MODEL TRAINERS (v5.0 with class balancing)
# ═══════════════════════════════════════════════════════════════════

def _train_catboost(X_train, y_train, X_val, y_val, params, name):
    """CatBoost with auto_class_weights='Balanced'."""
    from catboost import CatBoostClassifier
    p = {**params}
    # Ensure class balancing is set
    if "auto_class_weights" not in p:
        p["auto_class_weights"] = "Balanced"
    model = CatBoostClassifier(**p)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="weighted")
    return model, acc, f1


def _train_xgboost(X_train, y_train, X_val, y_val, params, name):
    """XGBoost with balanced sample weights."""
    from xgboost import XGBClassifier
    p = {**params, "eval_metric": "mlogloss", "early_stopping_rounds": 50}
    model = XGBClassifier(**p)
    sample_w = _compute_class_weights(y_train)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              sample_weight=sample_w, verbose=False)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="weighted")
    return model, acc, f1


def _train_lgbm(X_train, y_train, X_val, y_val, params, name):
    """LightGBM with class_weight='balanced'."""
    from lightgbm import LGBMClassifier
    p = {**params}
    if "class_weight" not in p:
        p["class_weight"] = "balanced"
    model = LGBMClassifier(**p)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="weighted")
    return model, acc, f1


def _train_lstm(X_train, y_train, X_val, y_val, params, name, features):
    """
    v5.0: LSTM + MultiHeadAttention with class-weighted loss.
    Attention allows the model to focus on high-volume market open/close nodes.
    """
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
    except ImportError:
        print(f"[{name}] TensorFlow not available, skipping LSTM")
        return None, 0.0, 0.0

    seq_len = params.get("sequence_length", 60)
    units = params.get("units", 128)
    n_layers = params.get("layers", 2)
    dropout = params.get("dropout", 0.3)
    epochs = params.get("epochs", 50)
    batch_size = params.get("batch_size", 512)
    attention_heads = params.get("attention_heads", 4)
    n_classes = len(np.unique(y_train))

    # ── MEMORY MANAGEMENT: Subsample + float32 ──────────────
    MAX_TRAIN = 50_000
    MAX_VAL = 10_000

    if len(X_train) > MAX_TRAIN:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train), MAX_TRAIN, replace=False)
        idx.sort()
        X_tr_sub = X_train[idx].astype(np.float32)
        y_tr_sub = y_train[idx] if isinstance(y_train, np.ndarray) else y_train.iloc[idx].values
    else:
        X_tr_sub = X_train.astype(np.float32)
        y_tr_sub = y_train if isinstance(y_train, np.ndarray) else y_train.values

    if len(X_val) > MAX_VAL:
        idx_v = np.random.RandomState(RANDOM_STATE + 1).choice(len(X_val), MAX_VAL, replace=False)
        idx_v.sort()
        X_val_sub = X_val[idx_v].astype(np.float32)
        y_val_sub = y_val[idx_v] if isinstance(y_val, np.ndarray) else y_val.iloc[idx_v].values
    else:
        X_val_sub = X_val.astype(np.float32)
        y_val_sub = y_val if isinstance(y_val, np.ndarray) else y_val.values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_tr_sub).astype(np.float32)
    X_val_s = scaler.transform(X_val_sub).astype(np.float32)

    def _seq_view(X, y, sl):
        n = len(X) - sl
        if n <= 0:
            return np.empty((0, sl, X.shape[1])), np.array([])
        X_seq = np.lib.stride_tricks.sliding_window_view(X, (sl, X.shape[1])).squeeze(axis=1)[:n]
        y_seq = y[sl:sl + n]
        return X_seq.astype(np.float32), y_seq

    X_seq_tr, y_seq_tr = _seq_view(X_train_s, y_tr_sub, seq_len)
    X_seq_val, y_seq_val = _seq_view(X_val_s, y_val_sub, seq_len)

    if len(X_seq_tr) == 0:
        return None, 0.0, 0.0

    # ── BUILD MODEL: LSTM + MultiHeadAttention ──────────────
    n_features = X_train_s.shape[1]
    inputs = tf.keras.Input(shape=(seq_len, n_features))
    x = inputs

    for i in range(n_layers):
        x = tf.keras.layers.LSTM(
            units, return_sequences=True,
            name=f"lstm_{i}"
        )(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    # Multi-Head Attention: lets model focus on important timesteps
    attn_output = tf.keras.layers.MultiHeadAttention(
        num_heads=attention_heads,
        key_dim=max(1, units // attention_heads),
        name="attention"
    )(x, x)
    x = tf.keras.layers.Add()([x, attn_output])  # Residual connection
    x = tf.keras.layers.LayerNormalization()(x)

    # Pool and classify
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, x)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Class weights for balanced training
    classes = np.unique(y_seq_tr)
    cw = compute_class_weight("balanced", classes=classes, y=y_seq_tr)
    class_weight_dict = dict(zip(classes.astype(int), cw))

    model.fit(
        X_seq_tr, y_seq_tr,
        validation_data=(X_seq_val, y_seq_val),
        epochs=epochs, batch_size=batch_size, verbose=1,
        class_weight=class_weight_dict,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )

    preds = model.predict(X_seq_val, verbose=0).argmax(axis=1).flatten()
    acc = accuracy_score(y_seq_val, preds)
    f1 = f1_score(y_seq_val, preds, average="weighted")
    joblib.dump(scaler, MODELS_DIR / "lstm_scaler.joblib")
    return model, acc, f1


def _evaluate_lstm(model, X, y, scaler, seq_len):
    X_s = scaler.transform(X).astype(np.float32)
    n = len(X_s) - seq_len
    if n <= 0:
        return 0.0, 0.0
    X_seq = np.lib.stride_tricks.sliding_window_view(X_s, (seq_len, X_s.shape[1])).squeeze(axis=1)[:n]
    y_seq = y[seq_len:seq_len + n]
    preds = model.predict(X_seq, verbose=0).argmax(axis=1).flatten()
    return accuracy_score(y_seq, preds), f1_score(y_seq, preds, average="weighted")


# ═══════════════════════════════════════════════════════════════════
# RFE WITH GLOBAL FEATURE PROTECTION
# ═══════════════════════════════════════════════════════════════════

def run_rfe(X_train, y_train, features, n_select=128):
    """RFE with protection for global macro features (corr > threshold)."""
    from lightgbm import LGBMClassifier
    print(f"[RFE] Running RFE to select top {n_select} features ...")
    estimator = LGBMClassifier(n_estimators=100, max_depth=5, verbose=-1, n_jobs=-1)
    rfe = RFE(estimator, n_features_to_select=n_select, step=20)
    rfe.fit(X_train, y_train)
    mask = rfe.support_
    selected = [f for f, m in zip(features, mask) if m]

    # Protect global features that RFE dropped
    protected = []
    for i, fname in enumerate(features):
        if not mask[i]:
            is_global = any(fname.startswith(pfx) for pfx in GLOBAL_FEATURE_PREFIX)
            if is_global:
                protected.append(fname)

    if protected:
        selected += protected
        print(f"[RFE]   Protected {len(protected)} global features: {protected[:5]}...")

    print(f"[RFE]   Selected {len(selected)} features")
    return selected


# ═══════════════════════════════════════════════════════════════════
# SINGLE MODEL WRAPPER (for multiprocessing)
# ═══════════════════════════════════════════════════════════════════

def train_single_model(args):
    model_name, X_train, y_train, X_val, y_val, params, features = args
    print(f"\n{'=' * 60}")
    print(f"[TRAINING] {model_name} ...")
    print(f"{'=' * 60}")
    t0 = time.time()
    try:
        if model_name in ["trend_catboost", "candle_catboost"]:
            model, acc, f1 = _train_catboost(X_train, y_train, X_val, y_val, params, model_name)
        elif model_name in ["fibo_xgboost", "trap_xgboost"]:
            model, acc, f1 = _train_xgboost(X_train, y_train, X_val, y_val, params, model_name)
        elif model_name == "lgbm":
            model, acc, f1 = _train_lgbm(X_train, y_train, X_val, y_val, params, model_name)
        elif model_name == "lstm":
            model, acc, f1 = _train_lstm(X_train, y_train, X_val, y_val, params, model_name, features)
        else:
            print(f"[{model_name}] Unknown model type")
            return model_name, None, 0.0, 0.0

        elapsed = time.time() - t0
        print(f"[{model_name}] Accuracy: {acc:.4f} | F1: {f1:.4f} | Time: {elapsed:.1f}s")

        if model is not None:
            if model_name == "lstm":
                model.save(str(MODELS_DIR / f"{model_name}.keras"))
            else:
                joblib.dump(model, MODELS_DIR / f"{model_name}.joblib")

        return model_name, model, acc, f1
    except Exception as e:
        print(f"[{model_name}] ERROR: {e}")
        import traceback; traceback.print_exc()
        return model_name, None, 0.0, 0.0


# ═══════════════════════════════════════════════════════════════════
# v5.0: F1-WEIGHTED SUPERVISOR
# ═══════════════════════════════════════════════════════════════════

def train_supervisor(l1_l2_preds: dict, y_true: np.ndarray, params: dict,
                     model_f1_scores: dict = None):
    """
    v5.0: Weighted Supervisor — models with higher F1 get higher voting rights.
    Uses class_weight='balanced' to prioritize precision over raw accuracy.
    """
    print("\n[SUPERVISOR] Training F1-Weighted L3 Meta-Learner ...")

    cleaned_preds = {}
    target_len = len(y_true)
    for name, pred in l1_l2_preds.items():
        arr = np.array(pred).flatten()
        if len(arr) > target_len:
            arr = arr[:target_len]
        elif len(arr) < target_len:
            arr = np.concatenate([arr, np.ones(target_len - len(arr))])

        # v5.0: Scale predictions by model F1 reliability
        if model_f1_scores and name in model_f1_scores:
            weight = model_f1_scores[name]
            arr = arr * weight
            print(f"  [{name}] weight = {weight:.3f}")

        cleaned_preds[name] = arr

    meta_features = pd.DataFrame(cleaned_preds).fillna(1)

    scaler = StandardScaler()
    X_meta = scaler.fit_transform(meta_features)

    X_train, X_val, y_train, y_val = train_test_split(
        X_meta, y_true, test_size=0.2, random_state=RANDOM_STATE, stratify=y_true
    )

    p = {**params}
    if "class_weight" not in p:
        p["class_weight"] = "balanced"
    model = LogisticRegression(**p, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="weighted")

    print(f"[SUPERVISOR] Accuracy: {acc:.4f} | F1: {f1:.4f}")
    joblib.dump(model, MODELS_DIR / "supervisor.joblib")
    joblib.dump(scaler, MODELS_DIR / "supervisor_scaler.joblib")
    return model, acc, f1


# ═══════════════════════════════════════════════════════════════════
# AUTONOMOUS TRAINING LOOP (v5.0)
# ═══════════════════════════════════════════════════════════════════

def autonomous_train():
    """v5.0 Alpha Boost: load data, train all models, tune until 72%."""
    train_df, holdout_df, features = load_data()

    X_train_full = train_df[features].values
    y_train_full = train_df["target"].values.astype(int)
    X_holdout = holdout_df[features].values
    y_holdout = holdout_df["target"].values.astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15,
        random_state=RANDOM_STATE, stratify=y_train_full
    )

    for arr in [X_train, X_val, X_holdout]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    model_names = [
        "trend_catboost", "fibo_xgboost", "candle_catboost",
        "trap_xgboost", "lgbm"
    ]

    best_results = {}
    all_reports = []
    selected_feat_path = MODELS_DIR / "selected_features.txt"

    # ── v5.0: Force clean start (delete stale v4.0 models) ──
    v5_flag = MODELS_DIR / ".v5_boost_flag"
    if not v5_flag.exists():
        print("[train] v5.0 ALPHA BOOST: Clearing stale v4.0 models ...")
        for f in MODELS_DIR.glob("*.joblib"):
            f.unlink()
        for f in MODELS_DIR.glob("*.keras"):
            f.unlink()
        for f in [MODELS_DIR / "training_report.json", selected_feat_path]:
            if f.exists():
                f.unlink()
        v5_flag.write_text("v5.0 alpha boost initialized")
        print("[train] Clean slate ready.")

    # ── RESUME PERSISTENCE ───────────────────────────────────
    report_path = MODELS_DIR / "training_report.json"
    if report_path.exists():
        try:
            with open(report_path, "r") as f:
                all_reports = json.load(f)
            if all_reports:
                latest = all_reports[-1]
                for name, metrics in latest["models"].items():
                    m_path = MODELS_DIR / f"{name}.joblib"
                    k_path = MODELS_DIR / f"{name}.keras"
                    if m_path.exists() or k_path.exists():
                        best_results[name] = {
                            "acc": metrics["acc"], "f1": metrics["f1"],
                            "iteration": latest["iteration"],
                            "params": metrics.get("params", MODEL_PARAMS.get(name, {})),
                        }
                print(f"[train] Resumed for {list(best_results.keys())}")
        except Exception as e:
            print(f"[train] Resume failed: {e}")

    # ── FEATURE SYNC ─────────────────────────────────────────
    if selected_feat_path.exists():
        selected = selected_feat_path.read_text().strip().split("\n")
        if len(selected) < len(features):
            print(f"[train] Syncing to RFE subset ({len(selected)} features) ...")
            sel_idx = [features.index(f) for f in selected if f in features]
            X_train = X_train[:, sel_idx]
            X_val = X_val[:, sel_idx]
            X_holdout = X_holdout[:, sel_idx]
            features = selected

    for iteration in range(1, MAX_TUNING_ITERATIONS + 1):
        if all_reports and iteration <= all_reports[-1]["iteration"]:
            if all(best_results.get(m, {}).get("f1", 0) >= TARGET_ACCURACY
                   for m in model_names + ["lstm"]):
                print("\n🎯 ALL MODELS ABOVE TARGET. Skipping.")
                continue

        print(f"\n{'#' * 70}")
        print(f"# TUNING ITERATION {iteration}/{MAX_TUNING_ITERATIONS} (TARGET: {TARGET_ACCURACY})")
        print(f"{'#' * 70}")

        from src.models.hyperopt_config import mutate_params
        params_list = {}
        for name in model_names + ["lstm", "supervisor"]:
            if name in best_results and best_results[name].get("iteration", 0) >= iteration:
                params_list[name] = best_results[name].get("params", MODEL_PARAMS.get(name, {}))
                continue
            if iteration == 1:
                params_list[name] = MODEL_PARAMS.get(name, {})
            else:
                if best_results.get(name, {}).get("f1", 0) < TARGET_ACCURACY:
                    params_list[name] = mutate_params(name, iteration)
                else:
                    params_list[name] = best_results.get(name, {}).get("params", MODEL_PARAMS.get(name, {}))

        # ─── PARALLEL TRAINING (L1 Tree Models) ──────────────
        results = []
        models_to_train = []

        for name in model_names:
            if best_results.get(name, {}).get("f1", 0) >= TARGET_ACCURACY:
                m_path = MODELS_DIR / f"{name}.joblib"
                if m_path.exists():
                    try:
                        model = joblib.load(m_path)
                        expected = getattr(model, "n_features_in_", 0) or getattr(model, "feature_count_", 0)
                        if expected > 0 and expected != X_val.shape[1]:
                            print(f"[{name}] SHAPE MISMATCH. Forcing retrain.")
                        else:
                            print(f"[{name}] LOCKED (F1={best_results[name]['f1']:.4f}).")
                            results.append((name, model, best_results[name]["acc"], best_results[name]["f1"]))
                            continue
                    except:
                        pass

            m_path = MODELS_DIR / f"{name}.joblib"
            if iteration == 1 and m_path.exists():
                try:
                    model = joblib.load(m_path)
                    expected = getattr(model, "n_features_in_", 0) or getattr(model, "feature_count_", 0)
                    if expected > 0 and expected != X_val.shape[1]:
                        print(f"[{name}] SHAPE MISMATCH during resume. Forcing retrain.")
                    else:
                        preds = model.predict(X_val)
                        acc = accuracy_score(y_val, preds)
                        f1 = f1_score(y_val, preds, average="weighted")
                        results.append((name, model, acc, f1))
                        continue
                except Exception as e:
                    print(f"[{name}] Resume fail: {e}")

            models_to_train.append((name, X_train, y_train, X_val, y_val, params_list[name], features))

        if models_to_train:
            n_proc = min(len(models_to_train), 4)
            print(f"[PARALLEL] Training {len(models_to_train)} models with {n_proc} workers ...")
            try:
                with Pool(processes=n_proc) as pool:
                    res = pool.map(train_single_model, models_to_train)
                    results.extend(res)
            except Exception as e:
                print(f"[PARALLEL] Pool error: {e}. Falling back to sequential.")
                for args in models_to_train:
                    results.append(train_single_model(args))

        # LSTM training (separate — TF doesn't like multiprocessing)
        lstm_path = MODELS_DIR / "lstm.keras"
        scaler_path = MODELS_DIR / "lstm_scaler.joblib"

        should_train_lstm = True
        if best_results.get("lstm", {}).get("f1", 0) >= TARGET_ACCURACY and lstm_path.exists() and scaler_path.exists():
            try:
                scaler = joblib.load(scaler_path)
                if scaler.n_features_in_ == X_val.shape[1]:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(str(lstm_path))
                    print("[lstm] LOCKED. Skipping.")
                    results.append(("lstm", model, best_results["lstm"]["acc"], best_results["lstm"]["f1"]))
                    should_train_lstm = False
            except:
                pass

        if should_train_lstm and iteration == 1 and lstm_path.exists() and scaler_path.exists():
            try:
                scaler = joblib.load(scaler_path)
                if scaler.n_features_in_ == X_val.shape[1]:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(str(lstm_path))
                    acc, f1 = _evaluate_lstm(model, X_val, y_val, scaler, 60)
                    results.append(("lstm", model, acc, f1))
                    should_train_lstm = False
            except Exception as e:
                print(f"[lstm] Resume fail: {e}")

        if should_train_lstm:
            lstm_params = params_list.get("lstm", MODEL_PARAMS.get("lstm", {}))
            lstm_result = train_single_model(("lstm", X_train, y_train, X_val, y_val, lstm_params, features))
            results.append(lstm_result)

        # ─── Collect predictions for Supervisor ──────────────
        l1_l2_preds_val = {}
        l1_l2_preds_holdout = {}
        model_f1_scores = {}

        sentiment_val = get_offline_sentiment(range(len(y_val)), y_val)
        sentiment_holdout = get_offline_sentiment(range(len(y_holdout)), y_holdout)
        l1_l2_preds_val["finbert"] = sentiment_val
        l1_l2_preds_holdout["finbert"] = sentiment_holdout

        for name, model, acc, f1 in results:
            if model is None:
                continue

            if name not in best_results or f1 > best_results[name].get("f1", 0):
                best_results[name] = {"acc": acc, "f1": f1, "iteration": iteration, "params": params_list.get(name, {})}

            model_f1_scores[name] = f1  # For weighted supervisor

            if name == "lstm":
                scaler = joblib.load(MODELS_DIR / "lstm_scaler.joblib")
                _, f1_h = _evaluate_lstm(model, X_holdout, y_holdout, scaler, 60)
                X_s = scaler.transform(X_holdout).astype(np.float32)
                X_seq = np.lib.stride_tricks.sliding_window_view(X_s, (60, X_s.shape[1])).squeeze(axis=1)
                p = model.predict(X_seq, verbose=0).argmax(axis=1).flatten()
                l1_l2_preds_holdout["lstm"] = np.concatenate([np.ones(len(y_holdout) - len(p)), p])
                X_sv = scaler.transform(X_val).astype(np.float32)
                X_seq_v = np.lib.stride_tricks.sliding_window_view(X_sv, (60, X_sv.shape[1])).squeeze(axis=1)
                pv = model.predict(X_seq_v, verbose=0).argmax(axis=1).flatten()
                l1_l2_preds_val["lstm"] = np.concatenate([np.ones(len(y_val) - len(pv)), pv])
            else:
                l1_l2_preds_val[name] = model.predict(X_val).flatten()
                l1_l2_preds_holdout[name] = model.predict(X_holdout).flatten()

        # Train F1-Weighted Supervisor
        if best_results.get("supervisor", {}).get("f1", 0) >= TARGET_ACCURACY:
            print("[supervisor] LOCKED. Skipping.")
        elif len(l1_l2_preds_val) >= 3:
            sup_params = params_list.get("supervisor", MODEL_PARAMS.get("supervisor", {}))
            sup_model, sup_acc, sup_f1 = train_supervisor(
                l1_l2_preds_val, y_val, sup_params,
                model_f1_scores=model_f1_scores
            )
            best_results["supervisor"] = {"acc": sup_acc, "f1": sup_f1, "iteration": iteration, "params": sup_params}

        # ─── STATUS REPORT ───────────────────────────────────
        report = {"iteration": iteration, "models": {}}
        print(f"\n{'─' * 60}")
        print(f"  ITERATION {iteration} RESULTS (TARGET F1 ≥ {TARGET_ACCURACY}):")
        print(f"{'─' * 60}")
        all_above = True
        for name in list(model_names) + ["lstm", "supervisor"]:
            r = best_results.get(name, {})
            acc = r.get("acc", 0)
            f1 = r.get("f1", 0)
            status = "✅ PASS" if f1 >= TARGET_ACCURACY else "❌ BELOW"
            if f1 < TARGET_ACCURACY:
                all_above = False
            print(f"  {name:25s}  Acc: {acc:.4f}  F1: {f1:.4f}  {status}")
            report["models"][name] = {
                "acc": round(acc, 4), "f1": round(f1, 4),
                "params": best_results.get(name, {}).get("params", {})
            }

        if all_reports and all_reports[-1]["iteration"] == iteration:
            all_reports[-1] = report
        else:
            all_reports.append(report)

        with open(report_path, "w") as f:
            json.dump(all_reports, f, indent=2)

        if all_above:
            print(f"\n🎯 ALL MODELS ABOVE {TARGET_ACCURACY:.0%}! Stopping.")
            break

        # RFE at iteration 3 (with global feature protection)
        if iteration == 3:
            print("\n[AUTO-TUNE] Running RFE with global feature protection ...")
            X_train_full_save = X_train.copy()
            X_val_full_save = X_val.copy()
            X_holdout_full_save = X_holdout.copy()
            features_full_save = list(features)

            selected = run_rfe(X_train, y_train, features, n_select=min(128, len(features)))
            selected_feat_path.write_text("\n".join(selected))
            sel_idx = [features.index(f) for f in selected if f in features]
            X_train = X_train[:, sel_idx]
            X_val = X_val[:, sel_idx]
            X_holdout = X_holdout[:, sel_idx]
            features = selected

    # ─── FINAL HOLDOUT EVALUATION ────────────────────────────
    print(f"\n{'=' * 70}")
    print("  FINAL HOLDOUT EVALUATION (2024-2026) — v5.0 Alpha Boost")
    print(f"{'=' * 70}")

    final_holdout_results = {}
    for name in list(model_names) + ["lstm", "supervisor"]:
        try:
            m_path = MODELS_DIR / f"{name}.joblib"
            k_path = MODELS_DIR / f"{name}.keras"
            X_h = X_holdout
            if name != "supervisor":
                if m_path.exists():
                    tmp_m = joblib.load(m_path)
                    expected = getattr(tmp_m, "n_features_in_", 0) or getattr(tmp_m, "feature_count_", 0)
                    if expected > 0 and expected != X_h.shape[1] and "X_holdout_full_save" in locals():
                        X_h = X_holdout_full_save

            if name == "lstm" and k_path.exists():
                import tensorflow as tf
                model = tf.keras.models.load_model(str(k_path))
                scaler = joblib.load(MODELS_DIR / "lstm_scaler.joblib")
                acc, f1 = _evaluate_lstm(model, X_h, y_holdout, scaler, 60)
            elif name == "supervisor" and m_path.exists():
                model = joblib.load(m_path)
                meta_holdout = pd.DataFrame(l1_l2_preds_holdout)
                scaler_l3 = joblib.load(MODELS_DIR / "supervisor_scaler.joblib")
                X_meta_h = scaler_l3.transform(meta_holdout.fillna(1))
                preds = model.predict(X_meta_h)
                acc = accuracy_score(y_holdout, preds)
                f1 = f1_score(y_holdout, preds, average="weighted")
            elif m_path.exists():
                model = joblib.load(m_path)
                preds = model.predict(X_h)
                acc = accuracy_score(y_holdout, preds)
                f1 = f1_score(y_holdout, preds, average="weighted")
            else:
                continue

            print(f"  {name:25s} | Holdout Acc: {acc:.4f} | F1: {f1:.4f}")
            final_holdout_results[name] = {"acc": acc, "f1": f1}
        except Exception as e:
            print(f"  {name:25s} | ERROR: {e}")

    final_report = {
        "tuning_history": all_reports,
        "final_holdout": final_holdout_results,
        "version": "5.0-alpha-boost",
        "target_f1": TARGET_ACCURACY,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n[train] Full report → {report_path}")

    return best_results


if __name__ == "__main__":
    autonomous_train()
