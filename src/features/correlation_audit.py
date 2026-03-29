"""
ATIS v5.0 — Pre-Training Correlation Audit
Validates that global macro features actually correlate with NIFTY before training.
Run AFTER build_features, BEFORE train_models.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import PROCESSED_DIR

def run_audit():
    path = PROCESSED_DIR / "nifty_features.parquet"
    feat_path = PROCESSED_DIR / "feature_list.txt"

    print(f"[audit] Loading {path} ...")
    df = pd.read_parquet(path)
    features = feat_path.read_text().strip().split("\n")
    features = [f for f in features if f in df.columns]

    print(f"[audit] {len(features)} features, {len(df):,} rows\n")

    # ── 1. TARGET CORRELATION ────────────────────────────────
    print("=" * 70)
    print("  TOP 30 FEATURES BY |CORRELATION| WITH TARGET")
    print("=" * 70)

    target = df["target"].astype(float)
    correlations = {}
    for f in features:
        try:
            corr = df[f].astype(float).corr(target)
            if not np.isnan(corr):
                correlations[f] = corr
        except:
            pass

    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in sorted_corr[:30]:
        bar = "█" * int(abs(corr) * 50)
        sign = "+" if corr > 0 else "-"
        print(f"  {name:35s} {sign}{abs(corr):.4f}  {bar}")

    # ── 2. GLOBAL FEATURES CHECK ─────────────────────────────
    global_prefixes = ["gap_", "forex_", "us_vix_", "dxy_", "spx_", "ndx_", "global_",
                       "overnight_", "gap_magnitude", "gap_adj_"]
    global_feats = [f for f in features if any(f.startswith(p) for p in global_prefixes)]

    if global_feats:
        print(f"\n{'=' * 70}")
        print(f"  GLOBAL MACRO FEATURE CORRELATIONS ({len(global_feats)} features)")
        print(f"{'=' * 70}")
        for f in global_feats:
            corr = correlations.get(f, 0)
            status = "✅ USEFUL" if abs(corr) > 0.02 else "⚠️  WEAK"
            print(f"  {f:35s} {corr:+.4f}  {status}")
    else:
        print("\n[audit] No global features found. Run fetch_global_data first.")

    # ── 3. v5.0 NEW FEATURES CHECK ──────────────────────────
    v5_prefixes = ["ha_", "gap_adj_", "overnight_", "gap_magnitude",
                   "rsi_14_vix", "atr_price", "macd_vix", "bb_width_vix",
                   "is_high_volume", "is_opening", "is_closing",
                   "vix_x_", "mtf_"]
    v5_feats = [f for f in features if any(f.startswith(p) for p in v5_prefixes)]

    if v5_feats:
        print(f"\n{'=' * 70}")
        print(f"  v5.0 NEW FEATURE CORRELATIONS ({len(v5_feats)} features)")
        print(f"{'=' * 70}")
        for f in sorted(v5_feats, key=lambda x: abs(correlations.get(x, 0)), reverse=True):
            corr = correlations.get(f, 0)
            status = "✅" if abs(corr) > 0.02 else "⚠️"
            print(f"  {f:35s} {corr:+.4f}  {status}")

    # ── 4. REDUNDANCY CHECK (high inter-correlation) ─────────
    print(f"\n{'=' * 70}")
    print(f"  REDUNDANCY CHECK (feature pairs with |corr| > 0.95)")
    print(f"{'=' * 70}")
    # Sample 50 features for speed
    sample_feats = [f for f in features if f in df.columns][:50]
    corr_matrix = df[sample_feats].astype(float).corr()
    redundant = []
    for i, f1 in enumerate(sample_feats):
        for j, f2 in enumerate(sample_feats):
            if i < j and abs(corr_matrix.loc[f1, f2]) > 0.95:
                redundant.append((f1, f2, corr_matrix.loc[f1, f2]))
    if redundant:
        for f1, f2, c in redundant[:15]:
            print(f"  {f1:25s} ↔ {f2:25s}  corr={c:.3f}")
        if len(redundant) > 15:
            print(f"  ... and {len(redundant)-15} more pairs")
    else:
        print("  None found (good!)")

    # ── 5. DATA LEAKAGE CHECK ────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  DATA LEAKAGE CHECK")
    print(f"{'=' * 70}")
    suspicious = [(f, c) for f, c in sorted_corr if abs(c) > 0.50]
    if suspicious:
        print("  ⚠️  Features with |corr| > 0.50 to target (possible leakage):")
        for f, c in suspicious:
            print(f"    {f:35s} {c:+.4f}")
    else:
        print("  ✅ No features with suspiciously high target correlation.")

    # ── 6. SUMMARY ───────────────────────────────────────────
    n_useful = sum(1 for _, c in sorted_corr if abs(c) > 0.01)
    n_weak = sum(1 for _, c in sorted_corr if abs(c) <= 0.01)
    print(f"\n{'=' * 70}")
    print(f"  AUDIT SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total features:     {len(features)}")
    print(f"  Useful (|r|>0.01):  {n_useful}")
    print(f"  Weak (|r|≤0.01):    {n_weak}")
    print(f"  Global features:    {len(global_feats)}")
    print(f"  v5.0 new features:  {len(v5_feats)}")
    print(f"  Redundant pairs:    {len(redundant)}")
    print(f"  Leakage suspects:   {len(suspicious)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_audit()
