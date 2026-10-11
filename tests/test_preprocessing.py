from src.preprocessing import preprocess_text
from src.sentiment_analysis import sentiment_label, sentiment_scores


def test_preprocessing_normalizes_without_dropping_negation_or_emoji():
    value = preprocess_text("I do NOT like this 😡 https://example.com @someone #StopIt")
    assert value.startswith("i do not like this")
    assert "url" in value and "user" in value
    assert "#stopit" in value and "😡" in value


def test_missing_text_is_empty_and_whitespace_is_collapsed():
    assert preprocess_text(None) == ""
    assert preprocess_text("  hello\n\t world ") == "hello world"


def test_vader_scores_and_labels():
    scores = sentiment_scores("I absolutely love this wonderful idea!")
    assert set(scores) == {"negative", "neutral", "positive", "compound"}
    assert scores["compound"] > 0
    assert sentiment_label(scores["compound"]) == "Positive"
    assert sentiment_label(0) == "Neutral"
