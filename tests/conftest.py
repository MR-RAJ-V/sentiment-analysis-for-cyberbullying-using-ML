import pytest
from sklearn.dummy import DummyClassifier

from src.train_models import make_pipeline


@pytest.fixture
def lightweight_bundle():
    labels = ["age", "ethnicity", "gender", "not_cyberbullying",
              "other_cyberbullying", "religion"]
    texts = ["a common kind supportive sample"] * len(labels)
    model = DummyClassifier(strategy="constant", constant="not_cyberbullying")
    pipeline = make_pipeline(model, include_char=True, include_sentiment=True)
    pipeline.fit(texts, labels)
    return {
        "schema_version": 2,
        "pipeline": pipeline,
        "labels": labels,
        "model_name": "Test dummy",
        "feature_mode": "pipeline_text_features",
        "seed": 42,
        "text_max_chars": 5000,
        "training_data": {"name": "fixture", "sha256": "fixture"},
        "split": {"strategy": "fixture", "random_seed": 42},
        "package_versions": {},
        "evaluation_metrics": {"selected_model": {},
                               "current_configuration_baseline": {}},
    }
