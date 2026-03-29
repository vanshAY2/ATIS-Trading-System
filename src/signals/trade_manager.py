"""
ATIS v4.0 — Trade Manager
Prediction locking, live tracking, and trade journal.
"""
import sys
import csv
import threading
from datetime import datetime, date
from typing import Dict, List, Optional
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import TRADES_DIR


class Trade:
    """Represents a single locked trade prediction."""

    def __init__(self, signal: Dict, timestamp: datetime):
        self.id = f"T{timestamp.strftime('%Y%m%d_%H%M%S')}"
        self.signal = signal
        self.timestamp = timestamp
        self.status = "LOCKED"  # LOCKED → WIN / LOSS / TIMEOUT
        self.entry_price = signal["entry"]
        self.sl = signal["sl"]
        self.targets = signal["targets"]
        self.targets_hit = [False] * len(signal["targets"])
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0.0
        self.pnl_percent = 0.0

    def check_price(self, current_price: float, current_time: datetime) -> str:
        """
        Check if current option price has hit SL or any target.
        Returns status after check.
        """
        if self.status != "LOCKED":
            return self.status

        # Check SL
        if current_price <= self.sl:
            self.status = "LOSS"
            self.exit_price = self.sl
            self.exit_time = current_time
            self.pnl = self.sl - self.entry_price
            self.pnl_percent = self.pnl / self.entry_price * 100
            return "LOSS"

        # Check targets (T1, T2, T3)
        for i, target in enumerate(self.targets):
            if not self.targets_hit[i] and current_price >= target:
                self.targets_hit[i] = True

        # If T3 hit — full win
        if self.targets_hit[-1]:
            self.status = "WIN"
            self.exit_price = self.targets[-1]
            self.exit_time = current_time
            self.pnl = self.exit_price - self.entry_price
            self.pnl_percent = self.pnl / self.entry_price * 100
            return "WIN"

        return "LOCKED"

    def force_close(self, exit_price: float, current_time: datetime, reason: str = "TIMEOUT"):
        """Force close at EOD or on news override."""
        self.status = reason
        self.exit_price = exit_price
        self.exit_time = current_time
        self.pnl = exit_price - self.entry_price
        self.pnl_percent = self.pnl / self.entry_price * 100

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.signal.get("symbol", ""),
            "direction": self.signal.get("direction", ""),
            "strike": self.signal.get("strike", 0),
            "option_type": self.signal.get("option_type", ""),
            "entry": self.entry_price,
            "sl": self.sl,
            "t1": self.targets[0] if len(self.targets) > 0 else 0,
            "t2": self.targets[1] if len(self.targets) > 1 else 0,
            "t3": self.targets[2] if len(self.targets) > 2 else 0,
            "t1_hit": self.targets_hit[0] if len(self.targets_hit) > 0 else False,
            "t2_hit": self.targets_hit[1] if len(self.targets_hit) > 1 else False,
            "t3_hit": self.targets_hit[2] if len(self.targets_hit) > 2 else False,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else "",
            "status": self.status,
            "pnl": round(self.pnl, 2),
            "pnl_percent": round(self.pnl_percent, 2),
            "confidence": self.signal.get("confidence", 0),
        }


class TradeManager:
    """
    Manages trade lifecycle: lock → track → resolve.
    Maintains full trade journal in CSV.
    """

    def __init__(self):
        self.active_trade: Optional[Trade] = None
        self.trade_history: List[Trade] = []
        self._lock = threading.Lock()
        self.journal_path = TRADES_DIR / "trade_log.csv"
        self._init_journal()

    def _init_journal(self):
        """Create journal CSV if it doesn't exist."""
        if not self.journal_path.exists():
            with open(self.journal_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._get_fields())
                writer.writeheader()

    def _get_fields(self) -> List[str]:
        return [
            "id", "timestamp", "symbol", "direction", "strike", "option_type",
            "entry", "sl", "t1", "t2", "t3", "t1_hit", "t2_hit", "t3_hit",
            "exit_price", "exit_time", "status", "pnl", "pnl_percent", "confidence",
        ]

    def can_lock(self) -> bool:
        """Check if a new trade can be locked (no active trade)."""
        with self._lock:
            return self.active_trade is None or self.active_trade.status != "LOCKED"

    def lock_trade(self, signal: Dict) -> Optional[Trade]:
        """
        Lock a new trade prediction.
        Returns None if a trade is already active.
        """
        with self._lock:
            if self.active_trade and self.active_trade.status == "LOCKED":
                return None  # Can't lock while another trade is active

            trade = Trade(signal, datetime.now())
            self.active_trade = trade
            print(f"[TradeManager] 🔒 LOCKED: {trade.id} — {signal['symbol']} "
                  f"@ ₹{signal['entry']:.2f}")
            return trade

    def update_price(self, current_price: float) -> Optional[str]:
        """Update active trade with current option price."""
        with self._lock:
            if not self.active_trade or self.active_trade.status != "LOCKED":
                return None

            status = self.active_trade.check_price(current_price, datetime.now())

            if status in ("WIN", "LOSS", "TIMEOUT"):
                self._finalize_trade()

            return status

    def force_close_eod(self, current_price: float):
        """Force close at end of day."""
        with self._lock:
            if self.active_trade and self.active_trade.status == "LOCKED":
                self.active_trade.force_close(current_price, datetime.now(), "TIMEOUT")
                self._finalize_trade()

    def force_close_news(self, current_price: float):
        """Force close due to high-impact news event."""
        with self._lock:
            if self.active_trade and self.active_trade.status == "LOCKED":
                self.active_trade.force_close(current_price, datetime.now(), "NEWS_OVERRIDE")
                self._finalize_trade()

    def _finalize_trade(self):
        """Write trade to journal and move to history."""
        trade = self.active_trade
        if trade:
            # Append to CSV
            with open(self.journal_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._get_fields())
                writer.writerow(trade.to_dict())

            self.trade_history.append(trade)
            emoji = {"WIN": "✅", "LOSS": "❌", "TIMEOUT": "⏰", "NEWS_OVERRIDE": "📰"}.get(trade.status, "")
            print(f"[TradeManager] {emoji} {trade.status}: {trade.id} "
                  f"P&L: ₹{trade.pnl:.2f} ({trade.pnl_percent:+.1f}%)")

    def get_active_trade(self) -> Optional[Dict]:
        """Get current active trade as dict."""
        with self._lock:
            if self.active_trade:
                return self.active_trade.to_dict()
            return None

    def get_statistics(self) -> Dict:
        """Get overall trading statistics."""
        trades = self.trade_history
        if not trades:
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                    "total_pnl": 0, "avg_pnl": 0, "max_win": 0, "max_loss": 0}

        wins = [t for t in trades if t.status == "WIN"]
        losses = [t for t in trades if t.status == "LOSS"]
        pnls = [t.pnl for t in trades]

        return {
            "total": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "timeouts": len([t for t in trades if t.status == "TIMEOUT"]),
            "win_rate": round(len(wins) / max(1, len(wins) + len(losses)) * 100, 1),
            "total_pnl": round(sum(pnls), 2),
            "avg_pnl": round(sum(pnls) / len(pnls), 2) if pnls else 0,
            "max_win": round(max(pnls), 2) if pnls else 0,
            "max_loss": round(min(pnls), 2) if pnls else 0,
        }
