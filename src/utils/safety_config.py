"""
ATIS v5.0 — Guardian Protocol: Safety Configuration & Circuit Breakers
Enforces risk limits, baseline F1 comparisons, and VIX-based trading halts.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
STATE_FILE = ROOT / "dev_state.json"

class GuardianProtocol:
    """
    Guardian Agent responsible for:
    1. F1 Score Baseline Comparison
    2. VIX Circuit Breaker (Global Risk)
    3. Stop Loss Enforcement (30% max)
    4. Trade Authorization Gate
    """
    
    def __init__(self, baseline_f1: float = 0.65, vix_threshold: float = 25.0, sl_percent: float = 0.30):
        self.baseline_f1 = baseline_f1
        self.vix_threshold = vix_threshold
        self.sl_percent = sl_percent
        self.state = self._load_state()
        
    def _load_state(self) -> Dict:
        """Load dev_state.json"""
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_state(self):
        """Save updated state"""
        self.state['timestamp'] = datetime.now().isoformat() + 'Z'
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    # ─── F1 BASELINE COMPARISON ───────────────────────────────────
    
    def check_f1_improvement(self, model_name: str, new_f1: float) -> Tuple[bool, str]:
        """
        Compare new F1 against baseline.
        Returns: (approved: bool, message: str)
        """
        baseline = self.baseline_f1
        improvement = ((new_f1 - baseline) / baseline) * 100
        
        if new_f1 >= baseline:
            status = f"✅ {model_name}: F1 {new_f1:.4f} >= baseline {baseline:.4f} (+{improvement:.1f}%)"
            self.state['metrics']['current_f1'] = new_f1
            self._save_state()
            return True, status
        else:
            status = f"⚠️ {model_name}: F1 {new_f1:.4f} < baseline {baseline:.4f} ({improvement:.1f}%)"
            return False, status
    
    # ─── GLOBAL VIX CIRCUIT BREAKER ────────────────────────────────
    
    def check_vix_circuit(self, vix_value: float) -> Tuple[bool, str]:
        """
        Global VIX > 25: HALT all trades.
        Returns: (trading_allowed: bool, message: str)
        """
        self.state['metrics']['global_vix'] = vix_value
        self._save_state()
        
        if vix_value > self.vix_threshold:
            msg = f"🛑 VIX CIRCUIT BREAKER TRIGGERED: VIX {vix_value:.2f} > {self.vix_threshold}"
            return False, msg
        else:
            msg = f"✅ VIX OK: {vix_value:.2f} <= {self.vix_threshold}"
            return True, msg
    
    # ─── STOP LOSS ENFORCEMENT ────────────────────────────────────
    
    def validate_sl(self, entry_price: float, sl_price: float) -> Tuple[bool, str]:
        """
        Enforce 30% max stop loss.
        Returns: (valid: bool, message: str)
        """
        sl_distance = abs(entry_price - sl_price) / entry_price
        
        if sl_distance > self.sl_percent:
            msg = f"❌ SL TOO FAR: {sl_distance*100:.2f}% > {self.sl_percent*100:.1f}% max"
            return False, msg
        else:
            msg = f"✅ SL VALID: {sl_distance*100:.2f}% <= {self.sl_percent*100:.1f}%"
            return True, msg
    
    # ─── US GAP ANALYSIS ──────────────────────────────────────────
    
    def check_us_gap(self, spy_close_prev: float, spy_today_open: float) -> Tuple[str, float]:
        """
        US Gap Analysis: If SPY closed down >1% overnight, flag bearish pressure.
        Returns: (signal: "bearish"|"neutral"|"bullish", gap_percent: float)
        """
        gap_pct = ((spy_today_open - spy_close_prev) / spy_close_prev) * 100
        self.state['metrics']['us_gap'] = gap_pct
        
        if gap_pct < -1.0:
            signal = "bearish"
            msg = f"📉 BEARISH GAP: SPY gap {gap_pct:.2f}% down overnight"
        elif gap_pct > 1.0:
            signal = "bullish"
            msg = f"📈 BULLISH GAP: SPY gap {gap_pct:.2f}% up overnight"
        else:
            signal = "neutral"
            msg = f"➡️ NEUTRAL GAP: SPY gap {gap_pct:.2f}%"
        
        self._save_state()
        return signal, gap_pct
    
    # ─── FOREX CORRELATION (CAUTION FLAG) ─────────────────────────
    
    def check_forex_strength(self, usdinr_current: float, usdinr_ema5: float) -> Tuple[str, bool]:
        """
        Forex Check: USD/INR > 5-day EMA = CAUTION flag for NIFTY bulls.
        Returns: (signal: "strong"|"weak", caution_flag: bool)
        """
        caution = usdinr_current > usdinr_ema5
        self.state['metrics']['forex_strength'] = float(usdinr_current)
        self._save_state()
        
        if caution:
            signal = "strong"
            msg = f"⚠️ CAUTION: USD/INR {usdinr_current:.4f} > 5-EMA {usdinr_ema5:.4f} (bearish for INR)"
        else:
            signal = "weak"
            msg = f"✅ NEUTRAL: USD/INR {usdinr_current:.4f} <= 5-EMA {usdinr_ema5:.4f}"
        
        return signal, caution
    
    # ─── TRADE AUTHORIZATION GATE ───────────────────────────────
    
    def authorize_trade(self, checks: Dict[str, bool]) -> Tuple[bool, str]:
        """
        Final approval gate for trade execution.
        Required checks:
          - f1_approved: Model F1 >= baseline
          - vix_ok: VIX <= threshold
          - sl_valid: Stop loss within 30%
          - no_us_gap: US gap not >-1%
        
        Returns: (approved: bool, reason: str)
        """
        all_pass = all(checks.values())
        
        if all_pass:
            msg = "✅ TRADE AUTHORIZED: All checks passed"
            return True, msg
        else:
            failures = [k for k, v in checks.items() if not v]
            msg = f"❌ TRADE REJECTED: {', '.join(failures)}"
            return False, msg
    
    def status_report(self) -> str:
        """Generate guardian status summary"""
        report = f"""
╔═══════════════════════════════════════╗
║    GUARDIAN PROTOCOL STATUS REPORT    ║
╚═══════════════════════════════════════╝

🛡️ BASELINE F1:        {self.baseline_f1:.4f}
📊 CURRENT F1:         {self.state['metrics'].get('current_f1', 0.0):.4f}
🌍 GLOBAL VIX:         {self.state['metrics'].get('global_vix', 'N/A')}
📉 US GAP:             {self.state['metrics'].get('us_gap', 'N/A')}
💱 FOREX STRENGTH:     {self.state['metrics'].get('forex_strength', 'N/A')}

🚨 CIRCUIT BREAKER:    {"🟢 ENABLED" if self.state['circuit_breakers'].get('enabled', True) else "🔴 DISABLED"}
🛑 VIX THRESHOLD:      {self.vix_threshold}
📉 SL MAX:             {self.sl_percent*100:.1f}%
"""
        return report


# ─── UTILITY FUNCTIONS ───────────────────────────────────────────

def get_guardian() -> GuardianProtocol:
    """Singleton accessor"""
    return GuardianProtocol()


if __name__ == "__main__":
    guardian = GuardianProtocol()
    print(guardian.status_report())
