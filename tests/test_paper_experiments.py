import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.svm import LinearSVC

from src.paper_experiments import (execution_scope, make_pipeline,
                                   merge_experiment_rows,
                                   previous_report_matches_source)


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


def test_report_scope_reflects_filter_only_rerun_and_preserved_rows():
    assert execution_scope(filter_only=True, smoke=False,
                           has_previous_full_report=True) == (
        "paper-method approximation; all-post rows retained and negative-polarity rows refreshed",
        "filter-only; negative-polarity SVM and Extra Trees",
    )


def test_repeated_filter_only_report_does_not_claim_missing_all_post_rows():
    assert execution_scope(filter_only=True, smoke=False,
                           has_previous_full_report=False) == (
        "filter-only; negative-polarity SVM and Extra Trees",
        "filter-only; negative-polarity SVM and Extra Trees",
    )


def test_report_reuse_requires_matching_protocol_and_source():
    report = {"protocol_version": "fold-fitted-vectorizer-grouped-cv-v2",
              "source_sha256": "data-hash", "source_rows": 100}
    assert previous_report_matches_source(report, "data-hash", 100)
    assert not previous_report_matches_source(report, "changed-hash", 100)
    assert not previous_report_matches_source(report, "data-hash", 101)
    old_protocol = {**report, "protocol_version": "old-protocol"}
    assert not previous_report_matches_source(old_protocol, "data-hash", 100)
