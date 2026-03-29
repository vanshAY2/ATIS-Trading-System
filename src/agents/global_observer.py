"""
ATIS v5.0 — Global Observer Agent
Monitors market-wide signals and data alignment:
- US Market close (SPY/QQQ) → NIFTY open alignment
- USD/INR forex strength
- Global VIX levels
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import GLOBAL_SYMBOLS
from src.utils.safety_config import GuardianProtocol

STATE_FILE = ROOT / "dev_state.json"


class GlobalObserver:
    """
    Global Observer Agent:
    - Fetches US market close data (SPY, QQQ)
    - Fetches forex rates (USD/INR)
    - Detects US overnight gaps
    - Generates global alpha signals
    """
    
    def __init__(self):
        self.guardian = GuardianProtocol()
        self.state = self._load_state()
        self.symbols = GLOBAL_SYMBOLS
        
    def _load_state(self) -> Dict:
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def update_agent_status(self, status: str, task: str, progress: int = 0):
        """Update state with agent's current activity"""
        self.state['active_agent'] = 'GlobalObserver'
        self.state['agents']['GlobalObserver']['status'] = status
        self.state['agents']['GlobalObserver']['last_task'] = task
        self.state['agents']['GlobalObserver']['progress'] = progress
        self.state['timestamp'] = datetime.now().isoformat() + 'Z'
        self._save_state()
    
    # ─── DATA ALIGNMENT LOGIC ─────────────────────────────────────
    
    def sync_market_times(self) -> Dict[str, datetime]:
        """
        Sync US Market close (16:00 EST) with NIFTY open (9:15 IST next day)
        
        Returns:
        {
            'us_market_close': datetime (prev day 16:00 EST),
            'nifty_open': datetime (today 9:15 IST),
            'time_gap_hours': float (hours between events)
        }
        """
        now = datetime.now()
        
        # Assuming we're getting NIFTY data → find corresponding US close
        # US market closes at 16:00 EST (21:00 UTC / 02:30 IST)
        us_close_utc = now.replace(hour=21, minute=0, second=0, microsecond=0)
        
        # If already past US close, use yesterday's close
        if now > us_close_utc:
            us_close_utc = us_close_utc - timedelta(days=1)
        
        # NIFTY opens at 9:15 IST (03:45 UTC)
        nifty_open_utc = us_close_utc + timedelta(hours=6, minutes=15)
        
        time_gap = (nifty_open_utc - us_close_utc).total_seconds() / 3600
        
        alignment = {
            'us_market_close': us_close_utc.isoformat(),
            'nifty_open': nifty_open_utc.isoformat(),
            'time_gap_hours': time_gap,
            'status': 'aligned'
        }
        
        self.state['metrics']['data_alignment'] = 'synced'
        self._save_state()
        
        return alignment
    
    # ─── US GAP ANALYSIS ──────────────────────────────────────────
    
    def analyze_us_gap(self, spy_df: pd.DataFrame) -> Dict:
        """
        Analyze SPY overnight gap from previous close to today's open.
        
        Args:
            spy_df: DataFrame with columns ['timestamp', 'open', 'close']
        
        Returns:
        {
            'gap_percent': float,
            'signal': 'bullish' | 'neutral' | 'bearish',
            'pressure': 'bearish_pressure' | 'neutral' | 'bullish_pressure',
            'recommendation': str
        }
        """
        if len(spy_df) < 2:
            return {'error': 'Insufficient SPY data'}
        
        prev_close = spy_df.iloc[-2]['close']
        today_open = spy_df.iloc[-1]['open']
        
        signal, gap_pct = self.guardian.check_us_gap(prev_close, today_open)
        
        return {
            'gap_percent': gap_pct,
            'signal': signal,
            'prev_close': float(prev_close),
            'today_open': float(today_open),
            'pressure': 'bearish_pressure' if gap_pct < -1.0 else ('bullish_pressure' if gap_pct > 1.0 else 'neutral'),
            'recommendation': f"Apply {signal} bias to NIFTY predictions"
        }
    
    # ─── FOREX STRENGTH CHECK ────────────────────────────────────
    
    def analyze_forex_strength(self, usdinr_df: pd.DataFrame) -> Dict:
        """
        Check USD/INR strength relative to 5-day EMA.
        
        Args:
            usdinr_df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close']
        
        Returns:
        {
            'usdinr_current': float,
            'usdinr_ema5': float,
            'signal': 'strong' | 'weak',
            'caution': bool,
            'interpretation': str
        }
        """
        if len(usdinr_df) < 5:
            return {'error': 'Insufficient USD/INR data for EMA5'}
        
        usdinr_df['ema5'] = usdinr_df['close'].ewm(span=5, adjust=False).mean()
        
        current = usdinr_df.iloc[-1]['close']
        ema5 = usdinr_df.iloc[-1]['ema5']
        
        signal, caution = self.guardian.check_forex_strength(current, ema5)
        
        return {
            'usdinr_current': float(current),
            'usdinr_ema5': float(ema5),
            'signal': signal,
            'caution': caution,
            'interpretation': f"USD/INR is {signal}. {'⚠️ CAUTION for NIFTY bulls' if caution else '✅ Supportive for NIFTY bulls'}"
        }
    
    # ─── GLOBAL VIX MONITORING ─────────────────────────────────
    
    def get_vix_status(self, vix_value: float) -> Dict:
        """
        Check if trading should be halted due to VIX.
        
        Returns:
        {
            'vix': float,
            'threshold': float,
            'trading_allowed': bool,
            'status': str
        }
        """
        allowed, msg = self.guardian.check_vix_circuit(vix_value)
        
        return {
            'vix': vix_value,
            'threshold': self.guardian.vix_threshold,
            'trading_allowed': allowed,
            'status': msg
        }
    
    # ─── COMPOSITE GLOBAL SIGNAL ───────────────────────────────
    
    def generate_global_signal(self, 
                                spy_df: pd.DataFrame,
                                usdinr_df: pd.DataFrame,
                                vix_value: float) -> Dict:
        """
        Composite signal combining all global factors.
        
        Returns weighted global bias for NIFTY predictions:
        {
            'global_bias': 'bullish' | 'neutral' | 'bearish',
            'confidence': float (0-1),
            'factors': {
                'us_gap': {...},
                'forex': {...},
                'vix': {...}
            }
        }
        """
        self.update_agent_status('active', 'Generating global signal', 50)
        
        us_gap = self.analyze_us_gap(spy_df)
        forex = self.analyze_forex_strength(usdinr_df)
        vix = self.get_vix_status(vix_value)
        
        # Score calculation
        score = 0
        if us_gap.get('signal') == 'bearish': score -= 1
        elif us_gap.get('signal') == 'bullish': score += 1
        
        if forex.get('caution'): score -= 0.5
        else: score += 0.5
        
        if not vix['trading_allowed']: score -= 2
        
        # Normalize bias
        if score > 0.5:
            bias = 'bullish'
            conf = min(abs(score) / 2, 1.0)
        elif score < -0.5:
            bias = 'bearish'
            conf = min(abs(score) / 2, 1.0)
        else:
            bias = 'neutral'
            conf = 0.5
        
        self.update_agent_status('active', 'Global signal ready', 100)
        
        return {
            'global_bias': bias,
            'confidence': conf,
            'score': score,
            'trading_allowed': vix['trading_allowed'],
            'factors': {
                'us_gap': us_gap,
                'forex': forex,
                'vix': vix
            }
        }
    
    def status_report(self) -> str:
        """Global Observer status"""
        report = f"""
╔════════════════════════════════════╗
║     GLOBAL OBSERVER STATUS         ║
╚════════════════════════════════════╝

📊 Agent Status: {self.state['agents']['GlobalObserver'].get('status', 'idle')}
🎯 Current Task: {self.state['agents']['GlobalObserver'].get('last_task', 'N/A')}
⏱️ Data Alignment: {self.state['metrics'].get('data_alignment', 'pending')}

🌍 Tracked Symbols:
{chr(10).join(f"   • {k}: {v}" for k, v in self.symbols.items())}

📍 Monitoring:
   ✓ US Gap Analysis (SPY overnight gaps)
   ✓ Forex Strength (USD/INR)
   ✓ Global VIX Levels
"""
        return report


if __name__ == "__main__":
    observer = GlobalObserver()
    print(observer.status_report())
    
    # Test sync
    print("\n" + "="*50)
    print("Market Time Synchronization:")
    print("="*50)
    alignment = observer.sync_market_times()
    for k, v in alignment.items():
        print(f"  {k}: {v}")
