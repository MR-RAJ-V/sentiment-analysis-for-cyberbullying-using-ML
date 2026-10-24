import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


@pytest.fixture
def lightweight_bundle():
    vectorizer = TfidfVectorizer().fit(["friendly and kind", "hurtful and mean"])
    model = DummyClassifier(strategy="constant", constant="not_cyberbullying")
    model.fit(np.zeros((2, 1)), ["not_cyberbullying", "age"])
    return {
        "model": model,
        "vectorizer": vectorizer,
        "labels": ["age", "not_cyberbullying"],
        "model_name": "Test dummy",
        "feature_mode": "tfidf_plus_vader",
        "text_max_chars": 5000,
    }
