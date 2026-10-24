"""Approximate the paper's classifier/resampling workflow without CV leakage."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                             GradientBoostingClassifier, RandomForestClassifier)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from nltk.stem import PorterStemmer

from src.data_analysis import DEFAULT_DATA
from src.sentiment_analysis import SENTIMENT_COLUMNS, sentiment_scores
from src.splitting import prepare_text_groups

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
SEED = 42
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
STEMMER = PorterStemmer()
CONFIG_KEYS = ("data_variant", "model", "smote")
PROTOCOL_VERSION = "fold-fitted-vectorizer-grouped-cv-v2"


def execution_scope(*, filter_only: bool, smoke: bool,
                    has_previous_full_report: bool) -> tuple[str, str]:
    """Describe report contents and the most recent execution separately."""
    if smoke:
        scope = "smoke; 6,000-row stratified sample and SVM only"
        return scope, scope
    if filter_only and has_previous_full_report:
        return (
            "paper-method approximation; all-post rows retained and negative-polarity rows refreshed",
            "filter-only; negative-polarity SVM and Extra Trees",
        )
    if filter_only:
        scope = "filter-only; negative-polarity SVM and Extra Trees"
        return scope, scope
    scope = "paper-method approximation; all-post and negative-polarity rows"
    return scope, scope


def previous_report_matches_source(report: dict, source_sha256: str,
                                   source_rows: int) -> bool:
    """Only merge old rows when their data and experiment protocol match."""
    return (
        report.get("protocol_version") == PROTOCOL_VERSION
        and report.get("source_sha256") == source_sha256
        and report.get("source_rows") == source_rows
    )


def paper_preprocess(value: str) -> str:
    """Local approximation of the paper's cleaning and normalization steps."""
    text = URL_RE.sub(" ", unicodedata.normalize("NFKC", str(value)).lower())
    emoji_words = []
    for char in text:
        if unicodedata.category(char).startswith("S"):
            name = unicodedata.name(char, "")
            if name:
                emoji_words.extend(name.lower().replace("-", " ").split())
    text = " ".join([text, *emoji_words])
    return " ".join(STEMMER.stem(token) for token in WORD_RE.findall(text)
                    if token not in ENGLISH_STOP_WORDS)


class PaperSentimentFeatures(BaseEstimator, TransformerMixin):
    """VADER feature block; shift compound to nonnegative for Naive Bayes."""

    def fit(self, X, y=None):
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        rows = []
        for text in X:
            scores = sentiment_scores(text)
            row = [scores[key] for key in SENTIMENT_COLUMNS]
            row[-1] = (row[-1] + 1.0) / 2.0
            rows.append(row)
        return csr_matrix(np.asarray(rows, dtype=np.float32))


def merge_experiment_rows(existing: list[dict], replacements: list[dict]) -> list[dict]:
    """Replace rerun configurations while retaining unrelated experiment rows."""
    merged: dict[tuple, dict] = {}
    for row in existing:
        key = tuple(row.get(field) for field in CONFIG_KEYS)
        merged[key] = row
    for row in replacements:
        key = tuple(row.get(field) for field in CONFIG_KEYS)
        merged[key] = row
    return list(merged.values())


def make_pipeline(estimator, *, use_smote: bool, use_svd: bool = False):
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
            max_features=25000, sublinear_tf=True)),
        ("vader", PaperSentimentFeatures()),
    ])
    steps = [("features", features)]
    if use_svd:
        steps.append(("svd", TruncatedSVD(n_components=40, random_state=SEED)))
    if use_smote:
        steps.append(("smote", SMOTE(random_state=SEED, k_neighbors=3)))
    steps.append(("classifier", estimator))
    return ImbalancedPipeline(steps)


def supported_models(use_smote: bool):
    return {
        "Logistic Regression": (LogisticRegression(max_iter=600, tol=1e-3,
            solver="saga", random_state=SEED), {"classifier__C": [1, 4]}, False),
        "Random Forest": (RandomForestClassifier(n_estimators=24, n_jobs=2,
            random_state=SEED), {"classifier__max_depth": [None, 40]}, False),
        "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
            {"classifier__max_depth": [None, 48]}, False),
        "Naive Bayes": (MultinomialNB(), {"classifier__alpha": [0.3, 1.0]}, False),
        "SVM": (LinearSVC(random_state=SEED), {"classifier__C": [0.5, 1.5]}, False),
        "Gradient Boosting": (GradientBoostingClassifier(n_estimators=20,
            random_state=SEED), {"classifier__max_depth": [1, 2]}, True),
        "Extra Trees": (ExtraTreesClassifier(n_estimators=24, n_jobs=2,
            random_state=SEED), {"classifier__max_depth": [None, 40]}, False),
        "AdaBoost": (AdaBoostClassifier(n_estimators=24, random_state=SEED),
            {"classifier__learning_rate": [0.5, 1.0]}, False),
    }


def run_variant(train_texts, y_train, groups_train, test_texts, y_test,
                labels, data_variant: str, use_smote: bool,
                model_names: set[str] | None = None) -> list[dict]:
    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True,
                                    random_state=SEED)
    cv = list(splitter.split(train_texts, y_train, groups=groups_train))
    for train_idx, validation_idx in cv:
        if set(groups_train[train_idx]).intersection(groups_train[validation_idx]):
            raise AssertionError("Normalized text groups crossed paper CV folds")
    rows = []
    for name, (estimator, grid, use_svd) in supported_models(use_smote).items():
        if model_names is not None and name not in model_names:
            continue
        pipeline = make_pipeline(estimator, use_smote=use_smote, use_svd=use_svd)
        search = GridSearchCV(pipeline, grid, scoring="accuracy", cv=cv, n_jobs=1,
                              refit=True, error_score="raise")
        start = time.perf_counter()
        try:
            search.fit(train_texts, y_train)
            predictions = search.predict(test_texts)
            rows.append({
                "data_variant": data_variant, "model": name, "smote": use_smote,
                "accuracy": float(accuracy_score(y_test, predictions)),
                "macro_f1": float(f1_score(y_test, predictions,
                                             average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_test, predictions,
                                                average="weighted", zero_division=0)),
                "best_params": search.best_params_,
                "cv_accuracy": float(search.best_score_),
                "fit_seconds_including_cv": round(time.perf_counter() - start, 2),
                "status": "ok",
            })
        except Exception as exc:
            rows.append({"data_variant": data_variant, "model": name,
                         "smote": use_smote,
                         "status": f"failed: {type(exc).__name__}: {exc}",
                         "fit_seconds_including_cv": round(time.perf_counter() - start, 2)})
    return rows


def _row_split(ids, labels):
    train_ids, test_ids = train_test_split(ids, test_size=0.30,
        random_state=SEED, stratify=labels)
    return np.asarray(train_ids), np.asarray(test_ids)


def run(data_path: Path = DEFAULT_DATA, *, filter_only: bool = False,
        smoke: bool = False) -> dict:
    if filter_only and smoke:
        raise ValueError("--filter-only and --smoke cannot be combined")
    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / "paper_experiments.json"
    csv_path = REPORT_DIR / "paper_experiments.csv"
    raw = pd.read_csv(data_path)
    source_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    previous = {}
    if filter_only and report_path.exists():
        candidate_previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous_report_matches_source(candidate_previous, source_sha256, len(raw)):
            previous = candidate_previous
        else:
            # Do not merge stale rows or reports made with an older protocol.
            previous = {}
    has_previous_full_report = any(
        row.get("data_variant") == "all_posts"
        for row in previous.get("models", [])
    )
    frame = prepare_text_groups(raw)
    if smoke:
        sample_size = min(6000, len(frame))
        sample_ids, _ = train_test_split(np.arange(len(frame)), train_size=sample_size,
            random_state=SEED, stratify=frame["cyberbullying_type"])
        frame = frame.iloc[sample_ids].reset_index(drop=True)
    original_texts = frame["tweet_text"].to_numpy()
    paper_texts = np.asarray([paper_preprocess(text) for text in original_texts],
                             dtype=object)
    y = frame["cyberbullying_type"].to_numpy()
    labels = sorted(np.unique(y))
    polarity = np.asarray([sentiment_scores(text)["compound"] < 0
                           for text in original_texts], dtype=bool)
    all_ids = np.arange(len(frame))
    train_ids, test_ids = _row_split(all_ids, y)

    new_rows = []
    if smoke:
        for use_smote in (False, True):
            new_rows.extend(run_variant(paper_texts[train_ids], y[train_ids],
                paper_texts[train_ids], paper_texts[test_ids],
                y[test_ids], labels, "smoke_all_posts", use_smote,
                model_names={"SVM"}))
    elif not filter_only:
        train_groups = paper_texts[train_ids]
        for use_smote in (False, True):
            new_rows.extend(run_variant(paper_texts[train_ids], y[train_ids],
                train_groups, paper_texts[test_ids], y[test_ids], labels,
                "all_posts", use_smote))

    filtered_ids = np.flatnonzero(polarity)
    filtered_y = y[filtered_ids]
    if not smoke and len(np.unique(filtered_y)) >= 2 and min(np.unique(filtered_y, return_counts=True)[1]) > 3:
        filter_train, filter_test = _row_split(filtered_ids, filtered_y)
        subset_rows = []
        for use_smote in (False, True):
            subset_rows.extend(run_variant(paper_texts[filter_train], y[filter_train],
                paper_texts[filter_train], paper_texts[filter_test], y[filter_test],
                sorted(np.unique(y[filtered_ids])),
                "negative_polarity_only", use_smote,
                model_names={"SVM", "Extra Trees"}))
        if filter_only:
            new_rows = subset_rows
        else:
            new_rows.extend(subset_rows)
    elif not smoke:
        subset_rows = [{"data_variant": "negative_polarity_only", "model": "all",
                        "smote": False,
                        "status": "skipped: filtered subset lacks enough examples across labels"}]
        if filter_only:
            new_rows = subset_rows
        else:
            new_rows.extend(subset_rows)

    existing_rows = previous.get("models", []) if filter_only else []
    rows = merge_experiment_rows(existing_rows, new_rows)
    normalized_groups = paper_texts
    overlap = set(normalized_groups[train_ids]).intersection(normalized_groups[test_ids])
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "run_scope": execution_scope(filter_only=filter_only, smoke=smoke,
                                      has_previous_full_report=has_previous_full_report)[0],
        "last_run_scope": execution_scope(filter_only=filter_only, smoke=smoke,
                                          has_previous_full_report=has_previous_full_report)[1],
        "source_name": Path(data_path).name,
        "source_sha256": source_sha256,
        "source_rows": int(len(raw)), "rows_used": int(len(frame)),
        "split": "70:30 stratified row split to mirror paper description; group overlap is measured",
        "random_seed": SEED,
        "train_rows": int(len(train_ids)), "test_rows": int(len(test_ids)),
        "duplicate_normalized_groups_crossing_row_split": int(len(overlap)),
        "paper_preprocessing_approximation": (
            "lowercase, Unicode emoji names, URL removal, English stopword removal, "
            "Porter stemming as a lemmatization proxy; exact resources and code unavailable"),
        "features": "fold-fitted word TF-IDF unigrams/bigrams plus four VADER features",
        "tfidf_max_features": 25000,
        "smote": "imblearn pipeline; applied within each CV training fold only",
        "tuning": "2-fold StratifiedGroupKFold CV with small grids; scoring accuracy",
        "negative_polarity_filter": "VADER compound < 0; filter-only scores use a new 70:30 stratified split of retained posts",
        "negative_rows_total": int(polarity.sum()),
        "negative_rows_train": int(polarity[train_ids].sum()),
        "negative_rows_test": int(polarity[test_ids].sum()),
        "negative_filter_model_scope": ["SVM", "Extra Trees"],
        "gradient_boosting": "SVD 40 dimensions; low tree count for tractability",
        "xgboost": "not run; dependency is not installed in the project environment",
        "models": rows,
    }
    if filter_only and previous:
        # Preserve compatible split/data metadata and unrelated full-data rows.
        for key, value in previous.items():
            if key not in {"models", "run_scope", "last_run_scope"}:
                report[key] = value
        report["run_scope"], report["last_run_scope"] = execution_scope(
            filter_only=True, smoke=False,
            has_previous_full_report=has_previous_full_report)
        report["models"] = rows
    if smoke:
        report_path = REPORT_DIR / "paper_experiments_smoke.json"
        csv_path = REPORT_DIR / "paper_experiments_smoke.csv"
        report["evaluation_note"] = "Smoke run only; do not compare these scores to full-data results."
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--filter-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Run SVM with and without fold-local SMOTE on a 6,000-row sample")
    args = parser.parse_args()
    result = run(args.data, filter_only=args.filter_only, smoke=args.smoke)
    report_name = ("reports/paper_experiments_smoke.json" if args.smoke
                   else "reports/paper_experiments.json")
    print(json.dumps({"report": report_name,
        "models": [row for row in result["models"]
                   if row.get("data_variant") == "negative_polarity_only"
                   or not args.filter_only]}, indent=2))


if __name__ == "__main__":
    main()
