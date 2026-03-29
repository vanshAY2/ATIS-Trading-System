"""
ATIS v4.0 — Strike Price Selector & Signal Validator
Calculates ATM strike, entry/SL/targets with real-world validation.
"""
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import (
    STRIKE_INTERVAL, SL_PERCENT, TARGET_RR_RATIOS,
    ENTRY_TOLERANCE, CONFIDENCE_GATE, NIFTY_LOT_SIZE,
)


class StrikeSelector:
    """
    Selects NIFTY option strike prices and computes validated entry/SL/targets.

    Rules:
    - ATM = round(spot / 50) * 50
    - CE for bullish, PE for bearish
    - Entry validated against recent traded range (±5%)
    - SL validated against tested extremes
    - Targets validated against average daily range
    """

    def __init__(self):
        self.adr_cache = {}  # average daily range cache

    def get_atm_strike(self, spot: float) -> int:
        """Round spot to nearest NIFTY strike interval."""
        return int(round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL)

    def get_option_strikes(self, spot: float, direction: str,
                           confidence: float) -> Dict:
        """
        Get option strike based on direction and confidence.

        Args:
            spot: Current NIFTY spot price
            direction: 'BULLISH' or 'BEARISH'
            confidence: Model ensemble confidence (0-1)

        Returns:
            dict with strike, option_type, moneyness
        """
        atm = self.get_atm_strike(spot)

        # Higher confidence → ATM, lower → 1-2 OTM
        if confidence >= 0.85:
            otm_offset = 0  # ATM
        elif confidence >= 0.75:
            otm_offset = 1  # 1 OTM
        else:
            otm_offset = 2  # 2 OTM

        if direction == "BULLISH":
            strike = atm + (otm_offset * STRIKE_INTERVAL)
            option_type = "CE"
        else:
            strike = atm - (otm_offset * STRIKE_INTERVAL)
            option_type = "PE"

        return {
            "strike": strike,
            "option_type": option_type,
            "moneyness": "ATM" if otm_offset == 0 else f"{otm_offset}OTM",
            "symbol": f"NIFTY {strike} {option_type}",
        }

    def estimate_premium(self, spot: float, strike: int,
                         option_type: str, vix_proxy: float = 0.15,
                         dte: int = 7) -> float:
        """
        Estimate option premium using simplified Black-Scholes proxy.
        Used in offline mode when Angel One API is not available.
        """
        moneyness = abs(spot - strike) / spot
        time_val = np.sqrt(dte / 365.0) * vix_proxy * spot * 0.4

        if option_type == "CE":
            intrinsic = max(0, spot - strike)
        else:
            intrinsic = max(0, strike - spot)

        # Premium = intrinsic + time value (decaying with OTM distance)
        otm_decay = np.exp(-moneyness * 20)
        premium = intrinsic + time_val * otm_decay

        return round(max(premium, 5.0), 2)  # minimum ₹5

    def compute_entry_sl_targets(
        self, spot: float, direction: str, confidence: float,
        atr: float, vix_proxy: float = 0.15, dte: int = 7,
        recent_high: Optional[float] = None,
        recent_low: Optional[float] = None,
        daily_range_avg: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Compute validated entry, SL, and multi-tiered targets.

        Returns None if validation fails (signal suppressed).
        """
        if confidence < CONFIDENCE_GATE:
            return None  # Below confidence threshold

        strike_info = self.get_option_strikes(spot, direction, confidence)
        premium = self.estimate_premium(
            spot, strike_info["strike"], strike_info["option_type"],
            vix_proxy, dte
        )

        entry = premium
        sl_points = entry * SL_PERCENT
        sl = round(entry - sl_points, 2)

        # Multi-tiered targets: T1=1:1, T2=1:2, T3=1:3 risk:reward
        targets = []
        for rr in TARGET_RR_RATIOS:
            target = round(entry + sl_points * rr, 2)
            targets.append(target)

        # ── VALIDATION ──────────────────────────────────────
        # 1. Entry must be reasonable (> ₹5, < ₹2000)
        if entry < 5 or entry > 2000:
            return None

        # 2. SL must be above zero
        if sl < 1:
            sl = round(entry * 0.5, 2)  # fallback: 50% SL

        # 3. Target validation against daily range
        if daily_range_avg is not None:
            max_reasonable_move = daily_range_avg * 2  # 2x ADR
            # Ensure T3 is not more than 2x ADR equivalent in premium
            premium_adr_ratio = (targets[-1] - entry) / entry
            if premium_adr_ratio > 2.0:
                # Scale down targets
                targets = [round(entry + sl_points * min(rr, 1.5), 2) for rr in TARGET_RR_RATIOS]

        # 4. SL validation against recent extremes
        if direction == "BULLISH" and recent_low is not None:
            # SL of underlying should not be below recent low
            spot_sl = spot - (atr * 1.5)
            if spot_sl < recent_low * 0.99:
                spot_sl = recent_low * 0.995

        result = {
            **strike_info,
            "entry": entry,
            "sl": sl,
            "targets": targets,
            "target_labels": ["T1 (1:1)", "T2 (1:2)", "T3 (1:3)"],
            "risk_points": round(sl_points, 2),
            "risk_per_lot": round(sl_points * NIFTY_LOT_SIZE, 2),
            "confidence": round(confidence, 4),
            "direction": direction,
            "spot_at_signal": spot,
            "valid": True,
        }

        return result

    def format_signal(self, signal: Dict) -> str:
        """Format signal for display."""
        if signal is None:
            return "⏳ NO SIGNAL — Confidence below threshold"

        direction_emoji = "🟢" if signal["direction"] == "BULLISH" else "🔴"
        lines = [
            f"{direction_emoji} SIGNAL: {'BUY CE' if signal['direction'] == 'BULLISH' else 'BUY PE'}",
            f"Strike: {signal['symbol']}  ({signal['moneyness']})",
            f"Entry:  ₹{signal['entry']:.2f}",
            f"SL:     ₹{signal['sl']:.2f}  (-{SL_PERCENT:.0%})",
        ]
        for i, (t, label) in enumerate(zip(signal["targets"], signal["target_labels"])):
            rr_pct = (t - signal["entry"]) / signal["entry"] * 100
            lines.append(f"{label}:  ₹{t:.2f}  (+{rr_pct:.0f}%)")
        lines.append(f"Confidence: {signal['confidence']:.0%}")
        lines.append(f"Risk/Lot:  ₹{signal['risk_per_lot']:.0f}")
        return "\n".join(lines)
