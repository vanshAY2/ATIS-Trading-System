"""
ATIS v5.0 — Hyperparameter Search Spaces for Autonomous Tuning
Updated with attention params, batch_size 512+, class balancing.
"""
import copy
import random
from config.settings import MODEL_PARAMS, RANDOM_STATE

random.seed(RANDOM_STATE)


def _perturb(val, pct=0.3, vtype="float", choices=None):
    if choices:
        return random.choice(choices)
    if vtype == "int":
        delta = max(1, int(val * pct))
        return max(1, val + random.randint(-delta, delta))
    else:
        delta = val * pct
        return max(1e-5, val + random.uniform(-delta, delta))


SEARCH_SPACES = {
    "trend_catboost": {
        "iterations":    lambda v: _perturb(v, 0.3, "int"),
        "depth":         lambda v: random.choice([5, 6, 7, 8, 9]),
        "learning_rate": lambda v: _perturb(v, 0.5),
        "l2_leaf_reg":   lambda v: random.choice([1, 3, 5, 7, 9]),
    },
    "fibo_xgboost": {
        "n_estimators":     lambda v: _perturb(v, 0.3, "int"),
        "max_depth":        lambda v: random.choice([5, 6, 7, 8, 9]),
        "learning_rate":    lambda v: _perturb(v, 0.5),
        "subsample":        lambda v: random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": lambda v: random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
    },
    "candle_catboost": {
        "iterations":    lambda v: _perturb(v, 0.3, "int"),
        "depth":         lambda v: random.choice([5, 6, 7, 8, 9]),
        "learning_rate": lambda v: _perturb(v, 0.5),
        "l2_leaf_reg":   lambda v: random.choice([1, 3, 5, 7, 9]),
    },
    "trap_xgboost": {
        "n_estimators":     lambda v: _perturb(v, 0.3, "int"),
        "max_depth":        lambda v: random.choice([5, 6, 7, 8, 9]),
        "learning_rate":    lambda v: _perturb(v, 0.5),
        "subsample":        lambda v: random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": lambda v: random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
    },
    "lgbm": {
        "n_estimators":     lambda v: _perturb(v, 0.3, "int"),
        "max_depth":        lambda v: random.choice([6, 7, 8, 9, 10]),
        "learning_rate":    lambda v: _perturb(v, 0.5),
        "subsample":        lambda v: random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": lambda v: random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
    },
    "lstm": {
        "units":           lambda v: random.choice([64, 96, 128, 192]),
        "layers":          lambda v: random.choice([1, 2, 3]),
        "dropout":         lambda v: random.choice([0.2, 0.3, 0.4]),
        "sequence_length": lambda v: random.choice([30, 60, 90]),
        "epochs":          lambda v: random.choice([30, 50, 75]),
        "batch_size":      lambda v: random.choice([512, 1024]),
        "attention_heads":  lambda v: random.choice([2, 4, 8]),
    },
    "supervisor": {
        "C": lambda v: random.choice([0.01, 0.1, 0.5, 1.0, 5.0, 10.0]),
    },
}


def mutate_params(model_name: str, iteration: int = 0) -> dict:
    """Return a mutated copy of the model's hyperparameters."""
    base = copy.deepcopy(MODEL_PARAMS.get(model_name, {}))
    space = SEARCH_SPACES.get(model_name, {})

    for key, mutator in space.items():
        if key in base:
            base[key] = mutator(base[key])

    return base
