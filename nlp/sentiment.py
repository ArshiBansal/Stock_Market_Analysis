"""
Zephyr – News fetching
Retrieves company-related headlines (primarily via yfinance)
and performs light cleaning / deduplication.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from config import TICKER_TO_REGION

logger = logging.getLogger(__name__)


def fetch_news_for_ticker(
    ticker: str,
    max_items: int = 30,
) -> pd.DataFrame:
    """
    Fetch recent news for a ticker using yfinance.

    Returns a DataFrame with columns:
        headline, publisher, published, link, ticker, region
    """
    try:
        t = yf.Ticker(ticker)
        raw_news = t.news or []

        if not raw_news:
            logger.info(f"No news returned for {ticker}")
            return pd.DataFrame()

        rows = []
        for item in raw_news[:max_items]:
            # yfinance news structure (can vary slightly)
            content = item.get("content") or item

            headline = (
                content.get("title")
                or content.get("headline")
                or item.get("title")
                or ""
            )
            publisher = (
                content.get("provider", {}).get("displayName")
                if isinstance(content.get("provider"), dict)
                else content.get("publisher")
                or item.get("publisher")
                or "Unknown"
            )
            link = (
                content.get("clickThroughUrl", {}).get("url")
                if isinstance(content.get("clickThroughUrl"), dict)
                else content.get("link")
                or item.get("link")
                or ""
            )

            # Timestamp handling
            pub_ts = None
            for key in ("pubDate", "providerPublishTime", "published"):
                val = content.get(key) or item.get(key)
                if val:
                    try:
                        if isinstance(val, (int, float)):
                            pub_ts = datetime.fromtimestamp(val, tz=timezone.utc)
                        else:
                            pub_ts = pd.to_datetime(val, utc=True)
                        break
                    except Exception:
                        continue

            if not headline:
                continue

            rows.append({
                "headline": headline.strip(),
                "publisher": str(publisher).strip(),
                "published": pub_ts,
                "link": link,
                "ticker": ticker,
                "region": TICKER_TO_REGION.get(ticker, "Unknown"),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.dropna(subset=["headline"])
            df = df.sort_values("published", ascending=False, na_position="last")
            df = df.reset_index(drop=True)

        return df

    except Exception as e:
        logger.error(f"News fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def deduplicate_headlines(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Remove near-duplicate headlines.
    Simple exact-match deduplication on the headline text.
    """
    if df.empty:
        return df

    subset = subset or ["headline"]
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    after = len(df)

    if before != after:
        logger.debug(f"Deduplicated news: {before} → {after}")

    return df.reset_index(drop=True)


def fetch_and_prepare_news(
    ticker: str,
    max_items: int = 30,
    sentiment_model: str = "FinBERT",
) -> pd.DataFrame:
    """
    Convenience function: fetch → deduplicate → add sentiment.
    """
    from .sentiment import analyze_sentiment_finbert, analyze_sentiment_vader

    df = fetch_news_for_ticker(ticker, max_items=max_items)
    if df.empty:
        return df

    df = deduplicate_headlines(df)

    analyzer = (
        analyze_sentiment_finbert
        if sentiment_model == "FinBERT"
        else analyze_sentiment_vader
    )

    sentiments = []
    for text in df["headline"]:
        res = analyzer(text)
        sentiments.append(res)

    sent_df = pd.DataFrame(sentiments)
    df = pd.concat([df.reset_index(drop=True), sent_df], axis=1)

    return df