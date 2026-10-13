"""Helpers for selecting predictor columns from generated feature tables."""

from collections.abc import Iterable


IDENTIFIER_COLUMNS = frozenset({"patient_id", "timestamp"})
TARGET_COLUMNS = frozenset({"glucose_over_180_next_2h"})


def get_predictor_columns(
    columns: Iterable[str], target_columns: Iterable[str] = TARGET_COLUMNS
) -> list[str]:
    """Return columns suitable as predictors, excluding IDs, time, and targets."""
    excluded = IDENTIFIER_COLUMNS.union(TARGET_COLUMNS, target_columns)
    return [column for column in columns if column not in excluded]