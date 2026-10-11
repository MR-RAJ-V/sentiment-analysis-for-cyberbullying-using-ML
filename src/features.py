"""Feature helpers shared between model training and serving."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack

from src.sentiment_analysis import SENTIMENT_COLUMNS, sentiment_scores


def sentiment_matrix(texts: list[str]):
    rows = [[sentiment_scores(text)[key] for key in SENTIMENT_COLUMNS] for text in texts]
    return csr_matrix(np.asarray(rows, dtype=np.float32))


def combine_features(text_features, texts: list[str]):
    return hstack([text_features, sentiment_matrix(texts)], format="csr", dtype=np.float32)
