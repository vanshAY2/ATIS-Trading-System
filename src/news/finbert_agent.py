"""
ATIS v4.0 — FinBERT News Sentiment Agent
Real-time news sentiment scoring using ProsusAI/finbert.
High-impact keyword detection for market-moving events.
"""
import sys
import time
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import HIGH_IMPACT_KEYWORDS
from src.news.news_fetcher import NewsFetcher


class FinBERTAgent:
    """
    FinBERT-based news sentiment agent.
    - Scores each headline from -1 (bearish) to +1 (bullish)
    - Flags high-impact keywords for L3 Supervisor override
    - Aggregates rolling sentiment from latest N headlines
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", max_headlines: int = 20):
        self.model_name = model_name
        self.max_headlines = max_headlines
        self.fetcher = NewsFetcher()
        self._pipeline = None
        self._headlines_cache: List[Dict] = []
        self._running = False
        self._lock = threading.Lock()

    def load_model(self):
        """Load FinBERT pipeline. Downloads ~440MB on first run."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            print("[FinBERT] Loading ProsusAI/finbert model ...")
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                top_k=None,
                device=-1,  # CPU
            )
            print("[FinBERT] Model loaded successfully.")
        except ImportError:
            print("[FinBERT] transformers/torch not installed. Running in offline mode.")
            self._pipeline = None
        except Exception as e:
            print(f"[FinBERT] Model load failed: {e}. Running in offline mode.")
            self._pipeline = None

    def score_headline(self, text: str) -> Tuple[float, bool]:
        """
        Score a single headline.
        Returns: (sentiment_score, is_high_impact)
        """
        # Check high-impact keywords
        text_upper = text.upper()
        is_high_impact = any(kw.upper() in text_upper for kw in HIGH_IMPACT_KEYWORDS)

        # FinBERT scoring
        if self._pipeline is None:
            return 0.0, is_high_impact

        try:
            result = self._pipeline(text[:512])  # truncate to model max
            if isinstance(result, list) and len(result) > 0:
                scores = result[0] if isinstance(result[0], list) else result
                score_map = {item["label"].lower(): item["score"] for item in scores}
                # Convert to -1 to +1
                pos = score_map.get("positive", 0)
                neg = score_map.get("negative", 0)
                sentiment = pos - neg  # -1 to +1
                return round(sentiment, 4), is_high_impact
        except Exception:
            pass

        return 0.0, is_high_impact

    def fetch_and_score(self) -> Dict:
        """Fetch latest headlines and score them all."""
        headlines = self.fetcher.fetch_all()

        scored = []
        for h in headlines[:self.max_headlines]:
            score, is_high_impact = self.score_headline(h["title"])
            scored.append({
                **h,
                "sentiment": score,
                "high_impact": is_high_impact,
            })

        with self._lock:
            self._headlines_cache = scored

        return self.get_aggregate()

    def get_aggregate(self) -> Dict:
        """
        Get aggregated sentiment state.
        Returns dict with score, override flag, and headline details.
        """
        with self._lock:
            cache = list(self._headlines_cache)

        if not cache:
            return {
                "finbert_sentiment_score": 0.0,
                "news_override": False,
                "high_impact_count": 0,
                "headline_count": 0,
                "headlines": [],
            }

        scores = [h["sentiment"] for h in cache]
        high_impact = [h for h in cache if h.get("high_impact")]

        # Weighted average: high-impact headlines get 3x weight
        weights = [3.0 if h.get("high_impact") else 1.0 for h in cache]
        weighted_score = np.average(scores, weights=weights)

        return {
            "finbert_sentiment_score": round(float(weighted_score), 4),
            "news_override": len(high_impact) > 0,
            "high_impact_count": len(high_impact),
            "headline_count": len(cache),
            "headlines": cache[:10],  # top 10 for dashboard
        }

    def start_background_polling(self, interval: int = 60):
        """Start background thread that polls news every `interval` seconds."""
        self._running = True

        def _poll():
            while self._running:
                try:
                    self.fetch_and_score()
                except Exception as e:
                    print(f"[FinBERT] Polling error: {e}")
                time.sleep(interval)

        t = threading.Thread(target=_poll, daemon=True)
        t.start()
        print(f"[FinBERT] Background polling started (every {interval}s)")

    def stop(self):
        self._running = False
