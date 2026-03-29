"""
ATIS v4.0 — Multi-Source News Fetcher
Fetches headlines from 4 News APIs + RSS feeds with rate-limit rotation.
"""
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Optional

import feedparser
import requests

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import (
    NEWSAPI_KEY, GNEWS_API_KEY, NEWSDATA_API_KEY,
    MARKETAUX_API_KEY, RSS_FEEDS,
)


class NewsFetcher:
    """
    Multi-source news fetcher with graceful fallback.
    Priority: NewsAPI → GNews → NewsData → MarketAux → RSS (always-on).
    """

    def __init__(self):
        self._seen = set()  # deduplicate headlines
        self._api_calls = {"newsapi": 0, "gnews": 0, "newsdata": 0, "marketaux": 0}
        self._daily_limits = {"newsapi": 500, "gnews": 100, "newsdata": 200, "marketaux": 100}

    def fetch_all(self, query: str = "NIFTY India market") -> List[Dict]:
        """Fetch from all available sources, return deduplicated headlines."""
        headlines = []
        headlines += self._fetch_newsapi(query)
        headlines += self._fetch_gnews(query)
        headlines += self._fetch_newsdata(query)
        headlines += self._fetch_marketaux(query)
        headlines += self._fetch_rss()

        # Deduplicate
        unique = []
        for h in headlines:
            key = hashlib.md5(h["title"].lower().encode()).hexdigest()
            if key not in self._seen:
                self._seen.add(key)
                unique.append(h)

        return unique

    def _fetch_newsapi(self, query: str) -> List[Dict]:
        if not NEWSAPI_KEY or self._api_calls["newsapi"] >= self._daily_limits["newsapi"]:
            return []
        try:
            url = "https://newsapi.org/v2/everything"
            params = {"q": query, "language": "en", "sortBy": "publishedAt",
                      "pageSize": 10, "apiKey": NEWSAPI_KEY}
            r = requests.get(url, params=params, timeout=5)
            self._api_calls["newsapi"] += 1
            if r.status_code == 200:
                articles = r.json().get("articles", [])
                return [{"title": a["title"], "source": "newsapi",
                         "published": a.get("publishedAt", ""),
                         "url": a.get("url", "")} for a in articles if a.get("title")]
        except Exception:
            pass
        return []

    def _fetch_gnews(self, query: str) -> List[Dict]:
        if not GNEWS_API_KEY or self._api_calls["gnews"] >= self._daily_limits["gnews"]:
            return []
        try:
            url = "https://gnews.io/api/v4/search"
            params = {"q": query, "lang": "en", "max": 10, "token": GNEWS_API_KEY}
            r = requests.get(url, params=params, timeout=5)
            self._api_calls["gnews"] += 1
            if r.status_code == 200:
                articles = r.json().get("articles", [])
                return [{"title": a["title"], "source": "gnews",
                         "published": a.get("publishedAt", ""),
                         "url": a.get("url", "")} for a in articles if a.get("title")]
        except Exception:
            pass
        return []

    def _fetch_newsdata(self, query: str) -> List[Dict]:
        if not NEWSDATA_API_KEY or self._api_calls["newsdata"] >= self._daily_limits["newsdata"]:
            return []
        try:
            url = "https://newsdata.io/api/1/news"
            params = {"q": query, "language": "en", "apikey": NEWSDATA_API_KEY}
            r = requests.get(url, params=params, timeout=5)
            self._api_calls["newsdata"] += 1
            if r.status_code == 200:
                articles = r.json().get("results", [])
                return [{"title": a["title"], "source": "newsdata",
                         "published": a.get("pubDate", ""),
                         "url": a.get("link", "")} for a in articles if a.get("title")]
        except Exception:
            pass
        return []

    def _fetch_marketaux(self, query: str) -> List[Dict]:
        if not MARKETAUX_API_KEY or self._api_calls["marketaux"] >= self._daily_limits["marketaux"]:
            return []
        try:
            url = "https://api.marketaux.com/v1/news/all"
            params = {"search": query, "language": "en", "limit": 10,
                      "api_token": MARKETAUX_API_KEY}
            r = requests.get(url, params=params, timeout=5)
            self._api_calls["marketaux"] += 1
            if r.status_code == 200:
                articles = r.json().get("data", [])
                return [{"title": a["title"], "source": "marketaux",
                         "published": a.get("published_at", ""),
                         "url": a.get("url", "")} for a in articles if a.get("title")]
        except Exception:
            pass
        return []

    def _fetch_rss(self) -> List[Dict]:
        """RSS feeds — always available, no API key needed."""
        headlines = []
        for url in RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    headlines.append({
                        "title": entry.get("title", ""),
                        "source": "rss",
                        "published": entry.get("published", ""),
                        "url": entry.get("link", ""),
                    })
            except Exception:
                continue
        return headlines

    def get_status(self) -> Dict:
        return {k: f"{v}/{self._daily_limits[k]}" for k, v in self._api_calls.items()}
