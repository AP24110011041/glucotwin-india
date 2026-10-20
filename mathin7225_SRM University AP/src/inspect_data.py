"""Inspect CSV files locally without changing their contents."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


COLUMN_GROUPS: dict[str, tuple[str, ...]] = {
    "cgm": (r"cgm", r"glucose", r"dexcom", r"libre", r"interstitial"),
    "meal": (r"meal", r"food", r"nutrition", r"carb", r"calorie", r"diet"),
    "wearable": (
        r"wearable",
        r"heart.?rate",
        r"\bhr\b",
        r"step",
        r"activity",
        r"acceler",
        r"sleep",
        r"wrist",
        r"electrodermal",
        r"skin.?temp",
    ),
    "static_clinical": (
        r"age",
        r"sex",
        r"gender",
        r"bmi",
        r"hba1c",
        r"diagnos",
        r"medication",
        r"clinical",
        r"static",
        r"height",
        r"weight",
        r"diabetes",
    ),
}
PARTICIPANT_PATTERN = re.compile(
    r"participant|subject|patient|person|study[_ ]?id|record[_ ]?id|^id$",
    re.IGNORECASE,
)
TIMESTAMP_PATTERN = re.compile(r"timestamp|datetime|date[_ ]?time|recorded[_ ]?at", re.IGNORECASE)


def _sampling_interval_minutes(dataframe: pd.DataFrame) -> dict[str, Any] | None:
    time_columns = [column for column in dataframe.columns if TIMESTAMP_PATTERN.search(str(column))]
    if not time_columns:
        return None

    timestamp_column = time_columns[0]
    timestamps = pd.to_datetime(dataframe[timestamp_column], errors="coerce").dropna().drop_duplicates()
    if len(timestamps) < 2:
        return {
            "column": str(timestamp_column),
            "valid_unique_timestamps": len(timestamps),
            "median_interval_minutes": None,
        }

    intervals = timestamps.sort_values().diff().dropna().dt.total_seconds() / 60
    return {
        "column": str(timestamp_column),
        "valid_unique_timestamps": len(timestamps),
        "median_interval_minutes": float(intervals.median()),
    }


def inspect_csv(path: Path) -> dict[str, Any]:
    """Return schema, missingness, likely domain columns, IDs, and timing for one CSV."""
    try:
        dataframe = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError(f"Could not read CSV '{path}': {error}") from error

    columns_by_group: dict[str, list[str]] = {}
    for group, patterns in COLUMN_GROUPS.items():
        columns_by_group[group] = [
            str(column)
            for column in dataframe.columns
            if any(re.search(pattern, str(column), re.IGNORECASE) for pattern in patterns)
        ]

    participant_columns = [
        column for column in dataframe.columns if PARTICIPANT_PATTERN.search(str(column))
    ]
    participant_ids = {
        str(column): sorted(dataframe[column].dropna().astype(str).unique().tolist())
        for column in participant_columns
    }

    return {
        "file": str(path),
        "rows": len(dataframe),
        "columns": {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()},
        "missing_values": {str(column): int(count) for column, count in dataframe.isna().sum().items()},
        "participant_ids": participant_ids,
        "columns_by_group": columns_by_group,
        "cgm_sampling": _sampling_interval_minutes(dataframe),
    }


def inspect_directory(directory: Path) -> list[dict[str, Any]]:
    """Inspect every CSV below a directory recursively."""
    if not directory.exists():
        raise FileNotFoundError(f"Input path does not exist: '{directory}'")
    if not directory.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: '{directory}'")

    csv_paths = sorted(directory.rglob("*.csv"))
    if not csv_paths:
        archives = sorted(directory.rglob("*.zip"))
        if archives:
            archive_list = ", ".join(str(path) for path in archives)
            raise FileNotFoundError(
                f"No CSV files found under '{directory}', but ZIP archive(s) are present: "
                f"{archive_list}. Extract the archive manually, then inspect the extracted folder."
            )
        raise FileNotFoundError(f"No CSV files found under '{directory}'. Add CSV files and try again.")
    return [inspect_csv(path) for path in csv_paths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("data/raw"),
        help="directory to scan recursively (default: data/raw)",
    )
    arguments = parser.parse_args()

    try:
        reports = inspect_directory(arguments.directory)
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
