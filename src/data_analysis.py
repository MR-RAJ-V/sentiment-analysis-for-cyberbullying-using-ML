"""Inspect the supplied CSV and emit quality and class-balance reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "cyberbullying_tweets.csv"


def analyze(path: Path = DEFAULT_DATA, reports_dir: Path = ROOT / "reports") -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(path)
    required = {"tweet_text", "cyberbullying_type"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Expected columns {sorted(required)}; found {list(frame.columns)}")
    counts = frame["cyberbullying_type"].value_counts().sort_index()
    report = {
        "source": str(path), "rows": int(len(frame)), "columns": list(frame.columns),
        "dtypes": {key: str(value) for key, value in frame.dtypes.items()},
        "missing_by_column": {key: int(value) for key, value in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_text_rows": int(frame.duplicated(subset=["tweet_text"]).sum()),
        "repeated_text_groups_with_conflicting_labels": int(
            (frame.groupby("tweet_text")["cyberbullying_type"].nunique() > 1).sum()),
        "blank_text_rows": int(frame["tweet_text"].fillna("").astype(str).str.strip().eq("").sum()),
        "label_counts": {str(key): int(value) for key, value in counts.items()},
        "labels": [str(value) for value in counts.index],
        "malformed_rows": int(frame[["tweet_text", "cyberbullying_type"]].isna().any(axis=1).sum()),
    }
    (reports_dir / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports_dir / "data_analysis.txt").write_text(
        "Dataset quality report\n" + "=" * 24 + "\n" +
        f"Rows: {report['rows']}\nColumns: {', '.join(report['columns'])}\n"+
        f"Missing values: {report['missing_by_column']}\nDuplicate rows: {report['duplicate_rows']}\n"+
        f"Repeated text rows: {report['duplicate_text_rows']}\nConflicting-label text groups: "
        f"{report['repeated_text_groups_with_conflicting_labels']}\n"+
        f"Blank text rows: {report['blank_text_rows']}\nLabels: {report['label_counts']}\n",
        encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        counts.sort_values().plot(kind="barh", color="#70d4ff")
        plt.title("Cyberbullying dataset class distribution")
        plt.xlabel("Posts")
        plt.tight_layout()
        plt.savefig(reports_dir / "class_distribution.png", dpi=160)
        plt.close()
    except ImportError:
        report["chart_status"] = "Matplotlib unavailable; chart not generated"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    print(json.dumps(analyze(args.data), indent=2))
