"""
Zephyr – Sentiment analysis
Supports VADER (baseline) and FinBERT (finance-specific).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from config import VADER_THRESHOLDS, SENTIMENT_LABELS

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# VADER
# ------------------------------------------------------------------
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
    _HAS_VADER = True
except ImportError:
    _HAS_VADER = False
    _VADER = None


def analyze_sentiment_vader(text: str) -> Dict[str, float | str]:
    """
    Return VADER scores + label for a single headline/text.
    """
    if not _HAS_VADER or not text or not isinstance(text, str):
        return {
            "compound": 0.0,
            "pos": 0.0,
            "neu": 1.0,
            "neg": 0.0,
            "label": "Neutral",
            "model": "VADER",
        }

    scores = _VADER.polarity_scores(text)
    label = get_sentiment_label(scores["compound"], method="vader")
    return {
        "compound": round(scores["compound"], 4),
        "pos": round(scores["pos"], 4),
        "neu": round(scores["neu"], 4),
        "neg": round(scores["neg"], 4),
        "label": label,
        "model": "VADER",
    }


# ------------------------------------------------------------------
# FinBERT
# ------------------------------------------------------------------
_FINBERT_PIPELINE = None
_HAS_FINBERT = False

def _load_finbert():
    global _FINBERT_PIPELINE, _HAS_FINBERT
    if _FINBERT_PIPELINE is not None:
        return

    try:
        from transformers import pipeline
        _FINBERT_PIPELINE = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        _HAS_FINBERT = True
        logger.info("FinBERT model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load FinBERT: {e}")
        _HAS_FINBERT = False


def analyze_sentiment_finbert(text: str) -> Dict[str, float | str]:
    """
    Return FinBERT prediction for a single headline.
    Labels are mapped to Positive / Negative / Neutral.
    """
    if not text or not isinstance(text, str):
        return {
            "label": "Neutral",
            "score": 0.0,
            "model": "FinBERT",
        }

    _load_finbert()
    if not _HAS_FINBERT:
        # Fallback to VADER if FinBERT is unavailable
        return analyze_sentiment_vader(text)

    try:
        result = _FINBERT_PIPELINE(text[:512])[0]  # safety truncation
        raw_label = result["label"].lower()
        score = float(result["score"])

        # FinBERT returns: positive / negative / neutral
        if raw_label == "positive":
            label = "Positive"
        elif raw_label == "negative":
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "label": label,
            "score": round(score, 4),
            "raw_label": raw_label,
            "model": "FinBERT",
        }
    except Exception as e:
        logger.error(f"FinBERT inference failed: {e}")
        return analyze_sentiment_vader(text)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def get_sentiment_label(
    score: float,
    method: str = "vader",
    thresholds: Optional[Dict] = None,
) -> str:
    """
    Convert a continuous score into Positive / Neutral / Negative.
    """
    thresholds = thresholds or VADER_THRESHOLDS

    if method == "vader":
        if score >= thresholds.get("positive", 0.05):
            return "Positive"
        elif score <= thresholds.get("negative", -0.05):
            return "Negative"
        else:
            return "Neutral"

    # Generic fallback
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    return "Neutral"


def analyze_headlines(
    headlines: List[str],
    model: str = "FinBERT",
) -> List[Dict]:
    """
    Analyze a list of headlines and return sentiment results.
    """
    results = []
    analyzer = analyze_sentiment_finbert if model == "FinBERT" else analyze_sentiment_vader

    for text in headlines:
        res = analyzer(text)
        res["headline"] = text
        results.append(res)
    return results


def aggregate_sentiment(
    df: pd.DataFrame,
    label_col: str = "label",
) -> Dict[str, Dict]:
    """
    Aggregate sentiment counts and percentages from a DataFrame
    that already contains a 'label' column (Positive/Neutral/Negative).
    """
    if df.empty or label_col not in df.columns:
        return {
            label: {"count": 0, "pct": 0.0}
            for label in SENTIMENT_LABELS
        }

    counts = df[label_col].value_counts()
    total = len(df)

    agg = {}
    for label in SENTIMENT_LABELS:
        cnt = int(counts.get(label, 0))
        agg[label] = {
            "count": cnt,
            "pct": round(cnt / total * 100, 1) if total > 0 else 0.0,
        }
    return agg