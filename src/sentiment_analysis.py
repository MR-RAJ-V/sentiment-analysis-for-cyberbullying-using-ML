"""VADER sentiment features and human-readable sentiment labels."""
from __future__ import annotations

from functools import lru_cache
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()
SENTIMENT_COLUMNS = ("negative", "neutral", "positive", "compound")


@lru_cache(maxsize=100_000)
def _cached_scores(text: str) -> tuple[float, float, float, float]:
    values = _analyzer.polarity_scores(text or "")
    return tuple(float(values[key]) for key in ("neg", "neu", "pos", "compound"))


def sentiment_scores(text: str) -> dict[str, float]:
    # VADER is deterministic; caching avoids rescoring repeated normalized texts
    # during grouped CV and keeps the returned mapping caller-owned.
    return dict(zip(SENTIMENT_COLUMNS, _cached_scores(str(text or ""))))


def sentiment_label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"
