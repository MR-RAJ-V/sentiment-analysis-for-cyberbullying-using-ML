"""Leakage-safe experiments and persistence for the local detector."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from src.data_analysis import DEFAULT_DATA, analyze
from src.features import combine_features
from src.preprocessing import preprocess_text
from src.sentiment_analysis import sentiment_scores

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SEED = 42


def _metrics(model, X_test, y_test, labels):
    pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
        "classification_report": classification_report(y_test, pred, labels=labels,
                                                          output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
    }


def train(data_path: Path = DEFAULT_DATA) -> dict:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    quality = analyze(data_path, REPORT_DIR)
    frame = pd.read_csv(data_path).dropna(subset=["tweet_text", "cyberbullying_type"])
    frame["tweet_text"] = frame["tweet_text"].astype(str).str.strip()
    frame = frame[frame["tweet_text"].ne("")].drop_duplicates(subset=["tweet_text", "cyberbullying_type"])
    frame["clean_text"] = frame["tweet_text"].map(preprocess_text)
    # Group by original tweet text so repeated/conflicting-label posts cannot
    # occur on both sides of the holdout. Three of ten stratified group folds
    # form an approximately 30% test split.
    y_all = frame["cyberbullying_type"].to_numpy()
    groups = frame["tweet_text"].to_numpy()
    fold_pairs = list(StratifiedGroupKFold(n_splits=10, shuffle=True,
        random_state=SEED).split(np.zeros(len(frame)), y_all, groups=groups))
    test_indices = np.sort(np.concatenate([fold_pairs[i][1] for i in range(3)]))
    train_indices = np.setdiff1d(np.arange(len(frame)), test_indices)
    X_all_text = frame["clean_text"].to_numpy()
    X_train_text = X_all_text[train_indices].tolist()
    X_test_text = X_all_text[test_indices].tolist()
    y_train, y_test = y_all[train_indices].tolist(), y_all[test_indices].tolist()
    shared_text_groups = set(groups[train_indices]).intersection(groups[test_indices])
    labels = sorted(frame["cyberbullying_type"].unique().tolist())
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                                 sublinear_tf=True, strip_accents="unicode",
                                 token_pattern=r"(?u)\b\w+\b")
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    X_train_full = combine_features(X_train_tfidf, X_train_text)
    X_test_full = combine_features(X_test_tfidf, X_test_text)

    # The six classes are close to balanced, so no synthetic resampling is used.
    # This avoids interpolating sparse lexical vectors while retaining all real posts.
    candidates = {
        "TF-IDF only · Logistic Regression": (LogisticRegression(C=4, max_iter=1000,
            tol=1e-3, solver="saga", class_weight="balanced", random_state=SEED), X_train_tfidf, X_test_tfidf),
        "TF-IDF + VADER · Logistic Regression": (LogisticRegression(C=4, max_iter=1000,
            tol=1e-3, solver="saga", class_weight="balanced", random_state=SEED), X_train_full, X_test_full),
        "TF-IDF + VADER · Linear SVM": (LinearSVC(C=1.5, class_weight="balanced", random_state=SEED),
            X_train_full, X_test_full),
        "TF-IDF only · Complement NB": (ComplementNB(alpha=0.3), X_train_tfidf, X_test_tfidf),
        "TF-IDF + VADER · Extra Trees": (ExtraTreesClassifier(n_estimators=160, max_depth=48,
            min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
            n_jobs=-1, random_state=SEED), X_train_full, X_test_full),
    }
    results = {}
    selected_model = None
    selected_metrics = None
    fit_durations = {}
    for name, (model, Xtr, Xte) in candidates.items():
        start = time.perf_counter()
        model.fit(Xtr, y_train)
        fit_durations[name] = round(time.perf_counter() - start, 2)
        result = _metrics(model, Xte, y_test, labels)
        result["fit_seconds"] = fit_durations[name]
        results[name] = result
        if name == "TF-IDF + VADER · Logistic Regression":
            selected_model, selected_metrics = model, result

    # Reproduce the paper's strict negative-polarity filter as a separate,
    # intentionally selection-biased diagnostic; never use it for serving.
    test_compound = [sentiment_scores(text)["compound"] for text in X_test_text]
    negative_indices = [i for i, score in enumerate(test_compound) if score < 0]
    polarity_diagnostic = {
        "definition": "Subset of the untouched test split with VADER compound < 0; diagnostic only.",
        "test_rows": len(y_test), "retained_negative_rows": len(negative_indices),
        "retained_fraction": len(negative_indices) / len(y_test),
        "accuracy_on_retained_subset": None,
        "macro_f1_on_retained_subset": None,
    }
    if negative_indices:
        filtered = _metrics(selected_model, X_test_full[negative_indices],
                            [y_test[i] for i in negative_indices], labels)
        polarity_diagnostic["accuracy_on_retained_subset"] = filtered["accuracy"]
        polarity_diagnostic["macro_f1_on_retained_subset"] = filtered["macro_f1"]

    # Save exactly the trained vectorizer and classifier used by the API.
    bundle = {
        "model": selected_model, "vectorizer": vectorizer,
        "labels": labels, "model_name": "Logistic Regression (TF-IDF + VADER)",
        "feature_mode": "tfidf_plus_vader", "seed": SEED,
        "text_max_chars": 5000,
    }
    joblib.dump(bundle, MODEL_DIR / "cyberbullying_model.joblib", compress=3)
    summary = {
        "dataset": {"rows_used": int(len(frame)), "source_rows": quality["rows"],
                    "removed_duplicates_and_invalid": int(quality["rows"] - len(frame)),
                    "labels": labels, "class_counts": {str(k): int(v) for k, v in
                        frame["cyberbullying_type"].value_counts().sort_index().items()}},
        "split": {"strategy": "stratified group holdout (3 of 10 folds); duplicate source texts stay together",
                  "random_seed": SEED, "train_rows": len(y_train), "test_rows": len(y_test),
                  "shared_text_groups_between_train_and_test": len(shared_text_groups)},
        "features": {"tfidf_vocabulary_size": len(vectorizer.vocabulary_),
                     "tfidf_ngrams": "unigrams and bigrams", "max_features": 50000,
                     "sentiment": ["negative", "neutral", "positive", "compound"]},
        "imbalance_strategy": "No SMOTE: observed class counts are nearly balanced; preserve sparse TF-IDF geometry.",
        "selected_model": bundle["model_name"], "model_artifact": "models/cyberbullying_model.joblib",
        "models": results, "paper_filter_diagnostic": polarity_diagnostic,
        "evaluation_note": "Single stratified holdout; parameters fixed in advance; no test-set tuning.",
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([{"model": name, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                   "weighted_f1": m["weighted_f1"], "fit_seconds": m["fit_seconds"]}
                  for name, m in results.items()]).to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    # A readable, portable confusion matrix for the dashboard.
    pd.DataFrame(selected_metrics["confusion_matrix"], index=labels, columns=labels).to_csv(
        REPORT_DIR / "confusion_matrix.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(9, 7))
        sns.heatmap(selected_metrics["confusion_matrix"], annot=True, fmt="d", cmap="mako",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Holdout confusion matrix · Logistic Regression + VADER")
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
    print(json.dumps({"selected_model": report["selected_model"],
                      "selected_metrics": report["models"]["TF-IDF + VADER · Logistic Regression"],
                      "artifact": report["model_artifact"]}, indent=2))
