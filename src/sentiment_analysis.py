"""VADER sentiment features and human-readable sentiment labels."""
from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()
SENTIMENT_COLUMNS = ("negative", "neutral", "positive", "compound")


def sentiment_scores(text: str) -> dict[str, float]:
    scores = _analyzer.polarity_scores(text or "")
    return {"negative": scores["neg"], "neutral": scores["neu"],
            "positive": scores["pos"], "compound": scores["compound"]}


def sentiment_label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"
