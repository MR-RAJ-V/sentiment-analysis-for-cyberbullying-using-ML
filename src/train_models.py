"""Leakage-safe group-aware model selection, evaluation, and persistence."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data_analysis import DEFAULT_DATA, analyze
from src.features import text_feature_union
from src.preprocessing import preprocess_text
from src.sentiment_analysis import sentiment_scores
from src.splitting import (group_class_counts, prepare_text_groups,
                           stratified_group_holdout)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SEED = 42
N_CV_SPLITS = 3


def make_pipeline(estimator, *, include_char: bool = False,
                  include_sentiment: bool = True,
                  word_max_features: int = 50000,
                  char_max_features: int = 40000) -> Pipeline:
    return Pipeline([
        ("features", text_feature_union(include_char=include_char,
            include_sentiment=include_sentiment,
            word_max_features=word_max_features,
            char_max_features=char_max_features)),
        ("classifier", estimator),
    ])


def candidate_searches():
    """Small, justified search spaces covering current and richer text features."""
    return {
        "Word TF-IDF · Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=1000, tol=1e-3, solver="saga",
                random_state=SEED), include_sentiment=False),
            {"classifier__C": [1, 4], "classifier__class_weight": ["balanced"]}),
        "Word TF-IDF + VADER · Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=1000, tol=1e-3, solver="saga",
                random_state=SEED), include_sentiment=True),
            {"classifier__C": [1, 4], "classifier__class_weight": [None, "balanced"]}),
        "Word TF-IDF + VADER · Linear SVM": (
            make_pipeline(LinearSVC(random_state=SEED), include_sentiment=True),
            {"classifier__C": [0.5, 1.5], "classifier__class_weight": ["balanced"]}),
        "Word + character TF-IDF + VADER · Linear SVM": (
            make_pipeline(LinearSVC(random_state=SEED), include_char=True,
                          include_sentiment=True),
            {"classifier__C": [0.5, 1.5], "classifier__class_weight": ["balanced"]}),
        "Word TF-IDF · Complement Naive Bayes": (
            make_pipeline(ComplementNB(), include_sentiment=False),
            {"classifier__alpha": [0.3, 1.0]}),
    }


def metric_summary(model, texts, y_true, labels) -> dict:
    predictions = model.predict(texts)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(precision_score(y_true, predictions, labels=labels,
                                                   average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predictions, labels=labels,
                                             average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, labels=labels,
                                    average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, labels=labels,
                                       average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, predictions,
            labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, predictions,
                                               labels=labels).tolist(),
    }


def _package_versions() -> dict[str, str]:
    packages = ("scikit-learn", "numpy", "scipy", "pandas", "nltk",
                "vaderSentiment", "joblib", "imbalanced-learn")
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _cross_validation_split_summary(y, groups, splits, labels) -> list[dict]:
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        overlap = set(groups[train_idx]).intersection(groups[valid_idx])
        if overlap:
            raise AssertionError(f"CV fold {fold} has {len(overlap)} text groups in both sides")
        rows.append({
            "fold": fold,
            "train_rows": int(len(train_idx)), "validation_rows": int(len(valid_idx)),
            "train_class_counts": group_class_counts(y[train_idx], groups[train_idx]),
            "validation_class_counts": group_class_counts(y[valid_idx], groups[valid_idx]),
            "shared_normalized_text_groups": 0,
        })
    return rows


def train(data_path: Path = DEFAULT_DATA) -> dict:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    data_path = Path(data_path)
    quality = analyze(data_path, REPORT_DIR)
    raw = pd.read_csv(data_path)
    frame = prepare_text_groups(raw)
    if frame.empty:
        raise ValueError("No usable labeled, nonblank text rows were found")

    y = frame["cyberbullying_type"].to_numpy()
    # Normalization is deterministic and corpus-independent; compute it once.
    # Learned vocabulary/IDF remain inside each CV fold's pipeline.
    texts = frame["normalized_text"].to_numpy()
    groups = frame["normalized_text"].to_numpy()
    labels = sorted(frame["cyberbullying_type"].unique().tolist())
    train_indices, test_indices = stratified_group_holdout(y, groups, seed=SEED)
    y_train, y_test = y[train_indices], y[test_indices]
    train_groups, test_groups = groups[train_indices], groups[test_indices]
    normalized_overlap = set(train_groups).intersection(test_groups)
    if normalized_overlap:
        raise AssertionError(f"{len(normalized_overlap)} normalized text groups cross holdout")

    cv_splitter = StratifiedGroupKFold(n_splits=N_CV_SPLITS, shuffle=True,
                                       random_state=SEED)
    cv_splits = list(cv_splitter.split(texts[train_indices], y_train,
                                      groups=train_groups))
    cv_summary = _cross_validation_split_summary(y_train, train_groups,
                                                 cv_splits, labels)

    # Refit the current deployed model's fixed configuration on this same
    # normalized-group training partition for a valid like-for-like comparison.
    baseline = make_pipeline(LogisticRegression(C=4, max_iter=1000, tol=1e-3,
        solver="saga", class_weight="balanced", random_state=SEED),
        include_sentiment=True)
    print("Fitting the fixed current-model baseline on normalized-group training rows...",
          flush=True)
    started = time.perf_counter()
    baseline.fit(texts[train_indices], y_train)
    baseline_fit_seconds = round(time.perf_counter() - started, 2)
    baseline_metrics = metric_summary(baseline, texts[test_indices], y_test, labels)
    baseline_metrics["fit_seconds"] = baseline_fit_seconds

    cv_results = {}
    best_name = None
    best_search = None
    best_cv_score = float("-inf")
    for name, (pipeline, param_grid) in candidate_searches().items():
        print(f"Running grouped CV for {name}...", flush=True)
        search = GridSearchCV(pipeline, param_grid, scoring="f1_macro",
                              cv=cv_splits, n_jobs=1, refit=True,
                              error_score="raise", return_train_score=False)
        started = time.perf_counter()
        search.fit(texts[train_indices], y_train)
        search_seconds = round(time.perf_counter() - started, 2)
        cv_results[name] = {
            "best_macro_f1": float(search.best_score_),
            "best_params": search.best_params_,
            "fit_seconds_including_cv": search_seconds,
            "candidate_scores": [
                {"params": params, "mean_macro_f1": float(score),
                 "std_macro_f1": float(std)}
                for params, score, std in zip(search.cv_results_["params"],
                    search.cv_results_["mean_test_score"],
                    search.cv_results_["std_test_score"])
            ],
        }
        if search.best_score_ > best_cv_score:
            best_name, best_search = name, search
            best_cv_score = float(search.best_score_)
        print(f"Finished {name}: CV macro F1={search.best_score_:.4f}", flush=True)

    if best_search is None:
        raise RuntimeError("All candidate searches failed")
    selected_model = best_search.best_estimator_
    # The final holdout is evaluated only after selection using training folds.
    selected_metrics = metric_summary(selected_model, texts[test_indices], y_test, labels)
    selected_metrics["cv_macro_f1"] = best_cv_score
    selected_metrics["best_params"] = best_search.best_params_
    inference_sample = texts[test_indices[:min(100, len(test_indices))]]
    started = time.perf_counter()
    selected_model.predict(inference_sample)
    selected_metrics["inference_ms_per_text"] = round(
        (time.perf_counter() - started) * 1000 / max(len(inference_sample), 1), 4)

    # Secondary, predeclared policy check: train a clone without ambiguous
    # normalized-text groups, but keep every test row for evaluation.
    labels_per_group = frame.groupby("normalized_text")["cyberbullying_type"].nunique()
    conflicting_groups = set(labels_per_group[labels_per_group > 1].index)
    train_unambiguous = np.asarray([group not in conflicting_groups
                                    for group in train_groups], dtype=bool)
    conflict_policy_diagnostic = {
        "policy": "Exclude conflicting normalized-text groups from training; keep the complete test set.",
        "training_rows_removed": int((~train_unambiguous).sum()),
        "training_groups_removed": int(len(set(train_groups[~train_unambiguous]))),
        "test_rows_removed": 0,
        "metrics": None,
    }
    if train_unambiguous.any() and len(np.unique(y_train[train_unambiguous])) == len(labels):
        conflict_model = clone(selected_model)
        conflict_model.fit(texts[train_indices][train_unambiguous], y_train[train_unambiguous])
        conflict_policy_diagnostic["metrics"] = metric_summary(
            conflict_model, texts[test_indices], y_test, labels)

    # Secondary, predeclared policy check: train a clone without ambiguous
    # normalized-text groups, but keep every test row for evaluation.
    labels_per_group = frame.groupby("normalized_text")["cyberbullying_type"].nunique()
    conflicting_groups = set(labels_per_group[labels_per_group > 1].index)
    train_unambiguous = np.asarray([group not in conflicting_groups
                                    for group in train_groups], dtype=bool)
    conflict_policy_diagnostic = {
        "policy": "Exclude conflicting normalized-text groups from training; keep the complete test set.",
        "training_rows_removed": int((~train_unambiguous).sum()),
        "training_groups_removed": int(len(set(train_groups[~train_unambiguous]))),
        "test_rows_removed": 0,
        "metrics": None,
    }
    if train_unambiguous.any() and len(np.unique(y_train[train_unambiguous])) == len(labels):
        conflict_model = clone(selected_model)
        conflict_model.fit(texts[train_indices][train_unambiguous], y_train[train_unambiguous])
        conflict_policy_diagnostic["metrics"] = metric_summary(
            conflict_model, texts[test_indices], y_test, labels)

    all_negative_test_indices = [index for index, text in enumerate(texts[test_indices])
                                 if sentiment_scores(preprocess_text(text))["compound"] < 0]
    negative_diagnostic = {
        "definition": "Diagnostic subset of untouched test rows with VADER compound < 0.",
        "test_rows": int(len(test_indices)),
        "retained_negative_rows": int(len(all_negative_test_indices)),
        "retained_fraction": float(len(all_negative_test_indices) / len(test_indices)),
    }
    if all_negative_test_indices:
        subset_texts = texts[test_indices][all_negative_test_indices]
        subset_labels = y_test[all_negative_test_indices]
        subset_metrics = metric_summary(selected_model, subset_texts, subset_labels, labels)
        negative_diagnostic.update({
            "accuracy": subset_metrics["accuracy"],
            "macro_f1": subset_metrics["macro_f1"],
            "weighted_f1": subset_metrics["weighted_f1"],
        })

    split_summary = {
        "strategy": "StratifiedGroupKFold 10-fold holdout; first 3 folds held out by normalized text",
        "random_seed": SEED,
        "train_rows": int(len(train_indices)), "test_rows": int(len(test_indices)),
        "train_groups": int(len(set(train_groups))), "test_groups": int(len(set(test_groups))),
        "shared_normalized_text_groups_between_train_and_test": int(len(normalized_overlap)),
        "train_class_counts": group_class_counts(y_train, train_groups),
        "test_class_counts": group_class_counts(y_test, test_groups),
        "cross_validation": cv_summary,
    }
    audit = quality["text_group_audit"]
    group_details = {
        **audit,
        "rows_used_after_exact_text_label_deduplication": int(len(frame)),
        "normalized_groups_in_train": int(len(set(train_groups))),
        "normalized_groups_in_test": int(len(set(test_groups))),
        "conflicting_label_groups_in_train": int(frame.iloc[train_indices].groupby(
            "normalized_text")["cyberbullying_type"].nunique().gt(1).sum()),
        "conflicting_label_groups_in_test": int(frame.iloc[test_indices].groupby(
            "normalized_text")["cyberbullying_type"].nunique().gt(1).sum()),
        "conflicting_examples_policy": "Retain conflicting-label rows and keep each normalized text group within one partition.",
        "exact_duplicate_policy": "Drop repeated exact text/label pairs only; retain contradictory labels.",
    }
    selected_name = best_name
    bundle = {
        "schema_version": 2,
        "pipeline": selected_model,
        "labels": labels,
        "model_name": selected_name,
        "feature_mode": "pipeline_text_features",
        "seed": SEED,
        "text_max_chars": 5000,
        "training_data": {
            "name": data_path.name,
            "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "source_rows": int(len(raw)), "rows_used": int(len(frame)),
            "label_counts": {str(k): int(v) for k, v in
                             frame["cyberbullying_type"].value_counts().sort_index().items()},
        },
        "split": split_summary,
        "package_versions": _package_versions(),
        "evaluation_metrics": {
            "selected_model": selected_metrics,
            "current_configuration_baseline": baseline_metrics,
        },
        "evaluation_metrics": {
            "selected_model": selected_metrics,
            "current_configuration_baseline": baseline_metrics,
        },
    }
    joblib.dump(bundle, MODEL_DIR / "cyberbullying_model.joblib", compress=3)

    summary = {
        "dataset": {
            "rows_used": int(len(frame)), "source_rows": int(len(raw)),
            "labels": labels, "class_counts": bundle["training_data"]["label_counts"],
            "text_group_audit": group_details,
            "source_sha256": bundle["training_data"]["sha256"],
        },
        "split": split_summary,
        "features": {
            "word_tfidf": "unigrams and bigrams; max_features=50000; min_df=2; sublinear_tf",
            "character_tfidf": "char_wb 3-5 grams; candidate-only; max_features=40000",
            "sentiment": ["negative", "neutral", "positive", "compound"],
            "normalization": "Deterministic normalization is applied once with preprocess_text before split/CV; vectorizers are fitted only within CV folds.",
        },
        "primary_selection_metric": "macro_f1",
        "baseline": {
            "name": "Current deployed configuration: word TF-IDF + VADER Logistic Regression (C=4)",
            "metrics": baseline_metrics,
        },
        "selected_model": selected_name,
        "selected_model_metrics": selected_metrics,
        "candidate_cv_results": cv_results,
        "conflicting_label_training_experiment": conflict_policy_diagnostic,
        "paper_filter_diagnostic": negative_diagnostic,
        "model_artifact": "models/cyberbullying_model.joblib",
        "package_versions": bundle["package_versions"],
        "evaluation_note": (
            "Candidates selected by group-aware CV macro F1 on the training partition. "
            "The normalized-text group holdout is isolated and evaluated after selection."),
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    comparison = [
        {"model": "Current configuration baseline", **baseline_metrics},
        {"model": selected_name, **selected_metrics},
    ]
    pd.DataFrame([{k: v for k, v in row.items() if k not in
                   ("classification_report", "confusion_matrix", "best_params")}
                  for row in comparison]).to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    labels_for_matrix = labels
    pd.DataFrame(selected_metrics["confusion_matrix"], index=labels_for_matrix,
                 columns=labels_for_matrix).to_csv(REPORT_DIR / "confusion_matrix.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(9, 7))
        sns.heatmap(selected_metrics["confusion_matrix"], annot=True, fmt="d",
                    cmap="mako", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"Normalized-group holdout · {selected_name}")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
        plt.close()
    except ImportError:
        pass
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    report = train(args.data)
    print(json.dumps({
        "selected_model": report["selected_model"],
        "baseline": report["baseline"]["metrics"],
        "selected_metrics": report["selected_model_metrics"],
        "artifact": report["model_artifact"],
    }, indent=2))
