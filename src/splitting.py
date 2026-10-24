"""Deterministic group-aware data splits and text-label ambiguity audits."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.preprocessing import preprocess_text


def prepare_text_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Return valid rows with the exact normalized text used for grouping/features."""
    prepared = frame.dropna(subset=["tweet_text", "cyberbullying_type"]).copy()
    prepared["tweet_text"] = prepared["tweet_text"].astype(str).str.strip()
    prepared = prepared[prepared["tweet_text"].ne("")]
    # Drop only exact text/label repeats. Keep label conflicts and all other rows.
    prepared = prepared.drop_duplicates(subset=["tweet_text", "cyberbullying_type"])
    prepared["normalized_text"] = prepared["tweet_text"].map(preprocess_text)
    prepared = prepared.reset_index(drop=True)
    return prepared


def text_group_audit(frame: pd.DataFrame) -> dict:
    """Summarize exact and normalized duplicates without resolving conflicts."""
    valid = frame.dropna(subset=["tweet_text", "cyberbullying_type"]).copy()
    valid["tweet_text"] = valid["tweet_text"].astype(str).str.strip()
    valid = valid[valid["tweet_text"].ne("")]
    normalized = valid["tweet_text"].map(preprocess_text)
    valid = valid.assign(normalized_text=normalized)
    exact_counts = valid.groupby("tweet_text", dropna=False)
    normalized_counts = valid.groupby("normalized_text", dropna=False)
    exact_conflicts = exact_counts["cyberbullying_type"].nunique()
    normalized_conflicts = normalized_counts["cyberbullying_type"].nunique()
    originals_per_normalized = normalized_counts["tweet_text"].nunique()
    return {
        "source_rows": int(len(frame)),
        "valid_nonblank_rows": int(len(valid)),
        "exact_duplicate_rows": int(valid.duplicated(
            subset=["tweet_text", "cyberbullying_type"]).sum()),
        "unique_exact_texts": int(valid["tweet_text"].nunique()),
        "unique_normalized_text_groups": int(valid["normalized_text"].nunique()),
        "normalization_collision_groups": int((originals_per_normalized > 1).sum()),
        "normalized_empty_text_rows": int(valid["normalized_text"].eq("").sum()),
        "exact_conflicting_label_groups": int((exact_conflicts > 1).sum()),
        "normalized_conflicting_label_groups": int((normalized_conflicts > 1).sum()),
        "rows_in_normalized_conflicting_groups": int(valid["normalized_text"].isin(
            normalized_conflicts[normalized_conflicts > 1].index).sum()),
        "duplicate_normalization_only_groups": int((originals_per_normalized > 1).sum()),
    }


def stratified_group_holdout(y, groups, *, seed: int = 42,
                             n_splits: int = 10,
                             test_folds: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic approximately 70:30 holdout with disjoint groups."""
    y = np.asarray(y)
    groups = np.asarray(groups, dtype=object)
    if len(y) != len(groups):
        raise ValueError("y and groups must have the same length")
    if not 1 <= test_folds < n_splits:
        raise ValueError("test_folds must be between 1 and n_splits - 1")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed)
    fold_pairs = list(splitter.split(np.zeros(len(y)), y, groups=groups))
    test_indices = np.sort(np.concatenate(
        [fold_pairs[index][1] for index in range(test_folds)]))
    train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
    overlap = set(groups[train_indices]).intersection(groups[test_indices])
    if overlap:
        raise AssertionError(f"Normalized-text groups crossed holdout: {len(overlap)}")
    return train_indices, test_indices


def group_class_counts(y, groups) -> dict[str, dict[str, int]]:
    """Count class rows and distinct text groups for each class."""
    frame = pd.DataFrame({"label": np.asarray(y), "group": np.asarray(groups)})
    return {
        str(label): {
            "rows": int(len(part)), "groups": int(part["group"].nunique())
        }
        for label, part in frame.groupby("label", sort=True)
    }
