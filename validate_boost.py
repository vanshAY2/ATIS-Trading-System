"""
ATIS v5.0 — Validation Report Generator
Compares v4.0 final_report.json with v5.0 training_report.json holdout results.
Outputs: validation_report.json
"""
import json
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models" / "saved"

# v4.0 baseline holdout scores (from final_report.json)
V4_BASELINE = {
    "trend_catboost":  {"acc": 0.3982, "f1": 0.4334},
    "fibo_xgboost":    {"acc": 0.6944, "f1": 0.6647},
    "candle_catboost": {"acc": 0.4814, "f1": 0.5041},
    "trap_xgboost":    {"acc": 0.6892, "f1": 0.6650},
    "lgbm":            {"acc": 0.6789, "f1": 0.6635},
    "lstm":            {"acc": 0.7010, "f1": 0.5778},
    "supervisor":      {"acc": 0.6681, "f1": 0.6828},
}


def generate_report():
    """Compare v4.0 vs v5.0 holdout results."""
    report_path = MODELS_DIR / "training_report.json"
    if not report_path.exists():
        print("[validate] No training_report.json found. Run training first.")
        return

    with open(report_path, "r") as f:
        data = json.load(f)

    v5_holdout = data.get("final_holdout", {})
    if not v5_holdout:
        print("[validate] No holdout results in report. Run full training pipeline.")
        return

    print("=" * 70)
    print("  ATIS v4.0 → v5.0 ALPHA BOOST VALIDATION REPORT")
    print("=" * 70)
    print(f"{'Model':<25s} {'v4.0 F1':>10s} {'v5.0 F1':>10s} {'Delta':>10s} {'Status':>10s}")
    print("-" * 70)

    comparison = {}
    all_improved = True

    for name in V4_BASELINE:
        v4_f1 = V4_BASELINE[name]["f1"]
        v5_data = v5_holdout.get(name, {})
        v5_f1 = v5_data.get("f1", 0)
        delta = v5_f1 - v4_f1
        status = "✅ UP" if delta > 0 else "❌ DOWN"
        if delta <= 0:
            all_improved = False

        print(f"  {name:<23s} {v4_f1:>10.4f} {v5_f1:>10.4f} {delta:>+10.4f} {status:>10s}")
        comparison[name] = {
            "v4_f1": round(v4_f1, 4),
            "v5_f1": round(v5_f1, 4),
            "delta": round(delta, 4),
            "improved": delta > 0,
        }

    # Check if supervisor F1 >= 0.72
    sup_f1 = v5_holdout.get("supervisor", {}).get("f1", 0)
    target_met = sup_f1 >= 0.72

    print("-" * 70)
    print(f"  Supervisor Holdout F1: {sup_f1:.4f}  {'🎯 TARGET MET (≥0.72)' if target_met else '❌ BELOW TARGET'}")
    print(f"  All models improved:   {'✅ YES' if all_improved else '❌ NO'}")
    print("=" * 70)

    # Save report
    validation = {
        "v4_baseline": V4_BASELINE,
        "v5_holdout": v5_holdout,
        "comparison": comparison,
        "supervisor_target_met": target_met,
        "all_improved": all_improved,
        "version": "5.0-alpha-boost",
    }

    out_path = MODELS_DIR.parent.parent / "validation_report.json"
    with open(out_path, "w") as f:
        json.dump(validation, f, indent=2)
    print(f"\n[validate] Report saved → {out_path}")


if __name__ == "__main__":
    generate_report()
