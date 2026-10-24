"""Feature helpers shared between model training and serving."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.utils.validation import check_is_fitted

from src.sentiment_analysis import SENTIMENT_COLUMNS, sentiment_scores
from src.preprocessing import preprocess_text


class TextNormalizer(BaseEstimator, TransformerMixin):
    """Deterministic text normalization that can live inside sklearn pipelines."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray([preprocess_text(text) for text in X], dtype=object)


class SentimentTransformer(BaseEstimator, TransformerMixin):
    """Return VADER features as a sparse matrix for text feature unions."""

    def fit(self, X, y=None):
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        return sentiment_matrix(list(X))


def text_feature_union(*, include_char: bool = False, include_sentiment: bool = True,
                       word_max_features: int = 50000,
                       char_max_features: int = 50000) -> FeatureUnion:
    """Create fold-fitted TF-IDF features with optional character and VADER inputs."""
    transformers = [
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                  max_features=word_max_features,
                                  sublinear_tf=True, strip_accents="unicode",
                                  token_pattern=r"(?u)\b\w+\b")),
    ]
    if include_char:
        transformers.append(("char", TfidfVectorizer(analyzer="char_wb",
            ngram_range=(3, 5), min_df=3, max_features=char_max_features,
            sublinear_tf=True)))
    if include_sentiment:
        transformers.append(("sentiment", SentimentTransformer()))
    return FeatureUnion(transformers)


def sentiment_matrix(texts: list[str]):
    rows = [[sentiment_scores(text)[key] for key in SENTIMENT_COLUMNS] for text in texts]
    return csr_matrix(np.asarray(rows, dtype=np.float32))


def combine_features(text_features, texts: list[str]):
    return hstack([text_features, sentiment_matrix(texts)], format="csr", dtype=np.float32)
