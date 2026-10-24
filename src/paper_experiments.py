"""Run a bounded, leakage-aware approximation of the supplied paper's pipeline.

This is an experiment runner only; it does not replace the serving model.
The paper does not publish exact split indices or hyperparameter grids, so the
settings below are explicit reproducible approximations.
"""
from __future__ import annotations

import json
import argparse
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                             GradientBoostingClassifier,
                             RandomForestClassifier)
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from nltk.stem import PorterStemmer

from src.data_analysis import DEFAULT_DATA
from src.sentiment_analysis import sentiment_scores

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
SEED = 42
PROGRESS_ROWS: list[dict] = []
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
STEMMER = PorterStemmer()


def paper_preprocess(value: str) -> str:
    """Approximate described lowercasing, emoji naming, filtering and lemmatizing."""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = URL_RE.sub(" ", text)
    emoji_words = []
    for char in text:
        if unicodedata.category(char).startswith("S"):
            name = unicodedata.name(char, "")
            if name:
                emoji_words.extend(name.lower().replace("-", " ").split())
    # The paper says emojis are translated to meanings; Unicode names provide a
    # deterministic local approximation without an emoji-description package.
    text = " ".join([text, *emoji_words])
    tokens = [STEMMER.stem(t) for t in WORD_RE.findall(text)
              if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def supported_models():
    # GridSearchCV grids are deliberately small because the paper does not
    # publish its search space. XGBoost is handled separately as unavailable.
    return {
        "Logistic Regression": (LogisticRegression(max_iter=600, tol=1e-3,
                                                     solver="saga", random_state=SEED),
                                {"classifier__C": [1, 4]}),
        "Random Forest": (RandomForestClassifier(n_estimators=24, n_jobs=2,
                                                  random_state=SEED),
                          {"classifier__max_depth": [None, 40]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                          {"classifier__max_depth": [None, 48]}),
        "Naive Bayes": (MultinomialNB(), {"classifier__alpha": [0.1, 1.0]}),
        "SVM": (LinearSVC(random_state=SEED), {"classifier__C": [0.5, 1.5]}),
        "Gradient Boosting": (GradientBoostingClassifier(n_estimators=20,
                                                          random_state=SEED),
                              {"classifier__max_depth": [1, 2]}),
        "Extra Trees": (ExtraTreesClassifier(n_estimators=24, n_jobs=2,
                                              random_state=SEED),
                        {"classifier__max_depth": [None, 40]}),
        "AdaBoost": (AdaBoostClassifier(n_estimators=24, random_state=SEED),
                     {"classifier__learning_rate": [0.5, 1.0]}),
    }


def run_variant(X_train, y_train, X_test, y_test, labels, resample: bool,
                data_variant: str, model_names: set[str] | None = None) -> list[dict]:
    rows = []
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    for name, (estimator, grid) in supported_models().items():
        if model_names is not None and name not in model_names:
            continue
        print(f"Starting {data_variant} | {name} | SMOTE={resample}", flush=True)
        fit_X, fit_y = X_train, y_train
        # Sklearn GradientBoosting requires dense data. For that classifier only,
        # reduce sparse TF-IDF to a compact dense semantic representation.
        steps = []
        if name == "Gradient Boosting":
            steps.append(("svd", TruncatedSVD(n_components=40,
                                               random_state=SEED)))
            # Keep the dense-only boosting search tractable at this corpus size.
            if len(y_train) > 10000:
                keep, _ = train_test_split(np.arange(len(y_train)), train_size=10000,
                                           random_state=SEED, stratify=y_train)
                fit_X, fit_y = X_train[keep], np.asarray(y_train)[keep]
        if resample:
            steps.append(("smote", SMOTE(random_state=SEED, k_neighbors=3)))
        steps.append(("classifier", estimator))
        pipeline = Pipeline(steps)
        params = dict(grid)
        if resample:
            params["smote__k_neighbors"] = [3]
        search = GridSearchCV(pipeline, params, scoring="accuracy", cv=cv,
                              n_jobs=1, refit=True, error_score="raise")
        started = time.perf_counter()
        try:
            search.fit(fit_X, fit_y)
            pred = search.predict(X_test)
            rows.append({
                "data_variant": data_variant, "model": name,
                "smote": resample, "accuracy": float(accuracy_score(y_test, pred)),
                "macro_f1": float(f1_score(y_test, pred, average="macro")),
                "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
                "best_params": search.best_params_,
                "cv_accuracy": float(search.best_score_),
                "fit_seconds": round(time.perf_counter() - started, 2),
                "status": "ok",
            })
            PROGRESS_ROWS.append(rows[-1])
            pd.DataFrame(PROGRESS_ROWS).to_csv(REPORT_DIR / "paper_experiments_progress.csv",
                                               index=False)
            print(f"Finished {data_variant} | {name} | SMOTE={resample}: "
                  f"accuracy={rows[-1]['accuracy']:.4f}", flush=True)
        except Exception as exc:
            rows.append({"data_variant": data_variant, "model": name,
                         "smote": resample, "status": f"failed: {type(exc).__name__}: {exc}",
                         "fit_seconds": round(time.perf_counter() - started, 2)})
            PROGRESS_ROWS.append(rows[-1])
            pd.DataFrame(PROGRESS_ROWS).to_csv(REPORT_DIR / "paper_experiments_progress.csv",
                                               index=False)
            print(f"Failed {data_variant} | {name} | SMOTE={resample}: {exc}", flush=True)
    return rows


def main(filter_only: bool = False):
    REPORT_DIR.mkdir(exist_ok=True)
    frame = pd.read_csv(DEFAULT_DATA).dropna(subset=["tweet_text", "cyberbullying_type"])
    frame["tweet_text"] = frame["tweet_text"].astype(str).str.strip()
    frame = frame[frame["tweet_text"].ne("")].drop_duplicates(
        subset=["tweet_text", "cyberbullying_type"])
    def englishish(text: str) -> bool:
        letters = [char for char in text if char.isalpha()]
        return not letters or sum(char.isascii() for char in letters) / len(letters) >= 0.80
    frame = frame[frame["tweet_text"].map(englishish)].reset_index(drop=True)
    original = frame["tweet_text"].tolist()
    y = frame["cyberbullying_type"].to_numpy()
    clean = [paper_preprocess(text) for text in original]
    vader = np.asarray([[sentiment_scores(text)[key] for key in
                         ("negative", "neutral", "positive", "compound")]
                        for text in original], dtype=np.float32)
    polarity = vader[:, 3] < 0
    # Multinomial Naive Bayes requires nonnegative inputs; shift VADER compound
    # from [-1, 1] into [0, 1] while retaining its ordering.
    vader[:, 3] = (vader[:, 3] + 1.0) / 2.0

    # A fixed 70:30 stratified row split mirrors the paper's stated ratio.
    # Group leakage is measured and documented because the paper gives no
    # duplicate handling or split seed.
    labels = sorted(np.unique(y))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                 max_features=25000, sublinear_tf=True)

    rows = []
    previous_report = None
    if filter_only and (REPORT_DIR / "paper_experiments.json").exists():
        previous_report = json.loads((REPORT_DIR / "paper_experiments.json").read_text(
            encoding="utf-8"))
        rows = [row for row in previous_report.get("models", [])
                if row.get("status") != "skipped: filtered data lacks sufficient examples in every label"]
    if not filter_only:
        train_ids, test_ids = train_test_split(np.arange(len(frame)), test_size=0.30,
                                               random_state=SEED, stratify=y)
        X_text_train = vectorizer.fit_transform([clean[i] for i in train_ids])
        X_text_test = vectorizer.transform([clean[i] for i in test_ids])
        X_train = hstack([X_text_train, csr_matrix(vader[train_ids])], format="csr")
        X_test = hstack([X_text_test, csr_matrix(vader[test_ids])], format="csr")
        # Full-data variant: test all classifiers, with and without SMOTE.
        for use_smote in (False, True):
            rows.extend(run_variant(X_train, y[train_ids], X_test, y[test_ids],
                                    labels, use_smote, "all_posts"))

    # Paper's stated strict polarity filter, applied before splitting as written.
    # This changes the evaluation population and is reported separately.
    filtered_ids = np.flatnonzero(polarity)
    yf = y[filtered_ids]
    if len(np.unique(yf)) >= 2 and min(np.unique(yf, return_counts=True)[1]) > 3:
        trf, tef = train_test_split(filtered_ids, test_size=0.30,
                                    random_state=SEED, stratify=yf)
        Xft = vectorizer.fit_transform([clean[i] for i in trf])
        Xfe = vectorizer.transform([clean[i] for i in tef])
        Xf_train = hstack([Xft, csr_matrix(vader[trf])], format="csr")
        Xf_test = hstack([Xfe, csr_matrix(vader[tef])], format="csr")
        # The negative-only filter is expensive to sweep across every model;
        # evaluate the paper's headline Extra Trees and strongest local linear
        # baseline with both resampling settings.
        for use_smote in (False, True):
            rows.extend(run_variant(Xf_train, y[trf], Xf_test, y[tef], labels,
                                    use_smote, "negative_polarity_only",
                                    model_names={"Extra Trees", "SVM"}))
    else:
        rows.append({"data_variant": "negative_polarity_only", "status":
                     "skipped: filtered data lacks sufficient examples in every label"})

    report = {
        "source": str(DEFAULT_DATA), "random_seed": SEED,
        "split": "70:30 stratified row split; exact-text overlap is measured below",
        "train_rows": int(len(train_ids)) if not filter_only else None,
        "test_rows": int(len(test_ids)) if not filter_only else None,
        "duplicate_text_groups_crossing_split": (int(len(
            set(frame.iloc[train_ids].tweet_text) & set(frame.iloc[test_ids].tweet_text)))
            if not filter_only else None),
        "paper_filter": "VADER compound < 0; metrics apply only to that retained subset",
        "paper_preprocessing_approximation": (
            "lowercase, Unicode emoji names, URL removal, English stopword removal, "
            "and Porter stemming as a local proxy for lemmatization; the paper's "
            "exact resources and code are not supplied"),
        "features": "word TF-IDF unigrams/bigrams plus four VADER scores",
        "tfidf_max_features": 25000,
        "naive_bayes_note": "VADER compound shifted from [-1, 1] into [0, 1] for NB compatibility",
        "smote": "training folds only; k_neighbors=3",
        "tuning": "2-fold stratified CV and small explicit grids; paper grids are unspecified",
        "gradient_boosting_note": "SVD reduced to 40 dimensions and fit on a stratified 10,000-row training subset for tractability",
        "negative_polarity_model_scope": "Linear SVM and Extra Trees only",
        "xgboost": "not run: package is unavailable in the project environment",
        "negative_rows": int(polarity.sum()),
        "models": rows,
    }
    if filter_only and previous_report:
        for key in ("split", "train_rows", "test_rows",
                    "duplicate_text_groups_crossing_split"):
            report[key] = previous_report[key]
    (REPORT_DIR / "paper_experiments.json").write_text(json.dumps(report, indent=2),
                                                          encoding="utf-8")
    pd.DataFrame(rows).to_csv(REPORT_DIR / "paper_experiments.csv", index=False)
    (REPORT_DIR / "paper_experiments_progress.csv").unlink(missing_ok=True)
    print(json.dumps({"report": "reports/paper_experiments.json",
                      "rows": rows,
                      "cross_split_duplicate_text_groups": report[
                          "duplicate_text_groups_crossing_split"]}, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter-only", action="store_true")
    main(filter_only=parser.parse_args().filter_only)
