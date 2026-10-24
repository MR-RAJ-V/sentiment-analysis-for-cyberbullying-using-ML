import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.svm import LinearSVC

from src.paper_experiments import make_pipeline, merge_experiment_rows


def test_filter_only_merge_is_idempotent_and_preserves_other_runs():
    previous = [
        {"data_variant": "all_posts", "model": "SVM", "smote": False,
         "accuracy": 0.8},
        {"data_variant": "negative_polarity_only", "model": "SVM",
         "smote": False, "accuracy": 0.7},
    ]
    replacement = [{"data_variant": "negative_polarity_only", "model": "SVM",
                    "smote": False, "accuracy": 0.75}]
    once = merge_experiment_rows(previous, replacement)
    twice = merge_experiment_rows(once, replacement)
    assert once == twice
    assert len(once) == 2
    assert next(row for row in once if row["data_variant"] == "all_posts")["accuracy"] == 0.8
    assert next(row for row in once if row["data_variant"] ==
                "negative_polarity_only")["accuracy"] == 0.75


def test_vectorizer_is_inside_group_cv_pipeline():
    texts = np.asarray([
        "insult awful target", "insult awful target", "support kind target",
        "support kind target", "attack cruel person", "attack cruel person",
        "friendly helpful person", "friendly helpful person",
    ], dtype=object)
    labels = np.asarray(["bad", "bad", "good", "good", "bad", "bad", "good", "good"])
    groups = np.asarray([f"group-{i // 2}" for i in range(len(texts))])
    pipeline = make_pipeline(LinearSVC(), use_smote=False)
    vectorizer = dict(pipeline.named_steps["features"].transformer_list)["word"]
    assert not hasattr(vectorizer, "vocabulary_")
    folds = list(StratifiedGroupKFold(n_splits=2, shuffle=True,
        random_state=11).split(texts, labels, groups))
    search = GridSearchCV(pipeline, {"classifier__C": [0.5, 1.0]}, cv=folds,
                          scoring="accuracy")
    search.fit(texts, labels)
    assert hasattr(search.best_estimator_.named_steps["features"].transformer_list[0][1],
                   "vocabulary_")
