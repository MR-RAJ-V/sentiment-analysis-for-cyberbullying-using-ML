import pandas as pd

from src.splitting import (prepare_text_groups, stratified_group_holdout,
                           text_group_audit)


def test_normalized_equivalent_texts_stay_in_one_holdout_group():
    rows = [("Hello &amp; world!", "age"), ("Hello & world!", "religion")]
    rows.extend((f"unique {label} example {index}", label)
                for label in ("age", "religion") for index in range(8))
    frame = prepare_text_groups(pd.DataFrame(rows, columns=["tweet_text", "cyberbullying_type"]))
    assert frame.iloc[0].normalized_text == frame.iloc[1].normalized_text
    train, test = stratified_group_holdout(frame.cyberbullying_type,
        frame.normalized_text, seed=7, n_splits=4, test_folds=1)
    assert not set(frame.normalized_text.iloc[train]).intersection(
        frame.normalized_text.iloc[test])


def test_conflicting_labels_are_retained_and_reported():
    frame = pd.DataFrame([
        ("hello &amp; world", "age"),
        ("hello &amp; world", "religion"),
        ("another unique text", "age"),
    ], columns=["tweet_text", "cyberbullying_type"])
    prepared = prepare_text_groups(frame)
    audit = text_group_audit(frame)
    assert len(prepared) == 3
    assert audit["normalized_conflicting_label_groups"] == 1
    assert audit["rows_in_normalized_conflicting_groups"] == 2
