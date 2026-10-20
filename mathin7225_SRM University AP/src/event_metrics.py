"""Sampled Libre event definitions and measured-observation exposure helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


EVENT_THRESHOLD_MG_DL = 180.0
MINIMUM_IN_RANGE_MINUTES = 60
MAX_EVENT_OBSERVATION_GAP_MINUTES = 1.0
ALERT_REFRACTORY_MINUTES = 60
EVENT_FORECAST_HORIZON_MINUTES = 120


def _validated_observations(observations: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "timestamp", "libre_gl_mg_dl"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Observation series is missing columns: {sorted(missing)}")
    if observations["patient_id"].isna().any():
        raise ValueError("Observation series contains missing patient IDs")

    frame = observations[["patient_id", "timestamp", "libre_gl_mg_dl"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("Observation series contains invalid timestamps")
    frame["libre_gl_mg_dl"] = pd.to_numeric(
        frame["libre_gl_mg_dl"], errors="raise"
    )
    if frame.duplicated(["patient_id", "timestamp"]).any():
        raise ValueError("Observation series contains duplicate patient timestamps")
    return frame.sort_values(
        ["patient_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)


def detect_libre_event_onsets(
    observations: pd.DataFrame,
    threshold_mg_dl: float = EVENT_THRESHOLD_MG_DL,
    minimum_in_range_minutes: int = MINIMUM_IN_RANGE_MINUTES,
    max_observation_gap_minutes: float = MAX_EVENT_OBSERVATION_GAP_MINUTES,
) -> pd.DataFrame:
    """Find first observed high readings after a contiguous sampled low run.

    A qualifying onset is a Libre reading strictly above ``threshold_mg_dl``
    after at least ``minimum_in_range_minutes`` from the first to last low
    readings, with every intervening source row observed at or below threshold
    and every adjacent timestamp gap no greater than ``max_observation_gap_minutes``.
    A missing value or larger gap resets the low run. High values already present
    at a patient's first observation, or after an observation gap, are not new
    events. This is a sampled-data rule, not proof of continuous physiological
    time-in-range.
    """
    if minimum_in_range_minutes <= 0 or max_observation_gap_minutes <= 0:
        raise ValueError("Event duration and maximum observation gap must be positive")
    frame = _validated_observations(observations)
    max_gap = pd.Timedelta(minutes=max_observation_gap_minutes)
    event_rows: list[dict[str, Any]] = []

    for patient_id, patient in frame.groupby("patient_id", sort=True):
        low_run_start: pd.Timestamp | None = None
        low_run_last: pd.Timestamp | None = None
        previous_timestamp: pd.Timestamp | None = None
        for timestamp, glucose in patient[
            ["timestamp", "libre_gl_mg_dl"]
        ].itertuples(index=False, name=None):
            gap_is_observed = (
                previous_timestamp is not None
                and pd.Timedelta(0) < timestamp - previous_timestamp <= max_gap
            )
            if previous_timestamp is not None and not gap_is_observed:
                low_run_start = None
                low_run_last = None

            if pd.isna(glucose):
                low_run_start = None
                low_run_last = None
            elif glucose <= threshold_mg_dl:
                if low_run_start is None:
                    low_run_start = timestamp
                low_run_last = timestamp
            else:
                if (
                    low_run_start is not None
                    and low_run_last is not None
                    and gap_is_observed
                    and low_run_last - low_run_start
                    >= pd.Timedelta(minutes=minimum_in_range_minutes)
                ):
                    event_rows.append(
                        {"patient_id": patient_id, "event_onset": timestamp}
                    )
                low_run_start = None
                low_run_last = None
            previous_timestamp = timestamp

    return pd.DataFrame(event_rows, columns=["patient_id", "event_onset"])


def merge_alert_episodes(
    alert_timestamps: Iterable[pd.Timestamp],
    refractory_minutes: int = ALERT_REFRACTORY_MINUTES,
) -> list[pd.Timestamp]:
    """Merge alerts no more than the refractory duration after an episode start.

    The first alert starts an episode. Further alerts at or before 60 minutes
    after that first alert are suppressed; an alert after that boundary starts a
    new episode. The window is anchored at the first alert, not extended by
    repeated alerts.
    """
    if refractory_minutes < 0:
        raise ValueError("refractory_minutes cannot be negative")
    alerts = sorted({pd.Timestamp(timestamp) for timestamp in alert_timestamps})
    if not alerts:
        return []

    refractory = pd.Timedelta(minutes=refractory_minutes)
    episodes = [alerts[0]]
    for timestamp in alerts[1:]:
        if timestamp - episodes[-1] > refractory:
            episodes.append(timestamp)
    return episodes


def match_alert_episodes_to_events(
    alert_episodes: Iterable[pd.Timestamp],
    event_onsets: Iterable[pd.Timestamp],
    forecast_horizon_minutes: int = EVENT_FORECAST_HORIZON_MINUTES,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """One-to-one match alert episodes to the earliest eligible future events.

    An alert can match an event only when the event onset is in
    ``(alert_time, alert_time + horizon]``. Each alert and each event can be
    used at most once; unmatched alert episodes are false alerts.
    """
    if forecast_horizon_minutes <= 0:
        raise ValueError("forecast_horizon_minutes must be positive")
    alerts = sorted(pd.Timestamp(timestamp) for timestamp in alert_episodes)
    events = sorted(pd.Timestamp(timestamp) for timestamp in event_onsets)
    horizon = pd.Timedelta(minutes=forecast_horizon_minutes)
    matched_events: set[int] = set()
    matches: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for alert in alerts:
        for index, event in enumerate(events):
            if index in matched_events:
                continue
            if alert < event <= alert + horizon:
                matched_events.add(index)
                matches.append((alert, event))
                break
    return matches


def measure_observed_minutes(
    observations: pd.DataFrame,
    patient_id: object,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_observation_gap_minutes: float = MAX_EVENT_OBSERVATION_GAP_MINUTES,
) -> float:
    """Sum only bounded intervals between adjacent, observed Libre rows.

    A contributing interval requires both adjacent source rows to have Libre
    values and their timestamp gap to be at most the configured maximum. Missing
    rows and larger timestamp gaps contribute zero; the function does not
    extrapolate before the first or after the last observation.
    """
    if max_observation_gap_minutes <= 0:
        raise ValueError("max_observation_gap_minutes must be positive")
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    if end_time <= start_time:
        return 0.0

    frame = _validated_observations(observations)
    patient = frame.loc[frame["patient_id"].astype(str).eq(str(patient_id))]
    if len(patient) < 2:
        return 0.0

    timestamps = patient["timestamp"].tolist()
    glucose = patient["libre_gl_mg_dl"].tolist()
    max_gap = pd.Timedelta(minutes=max_observation_gap_minutes)
    observed_duration = pd.Timedelta(0)
    for index in range(1, len(patient)):
        left_time, right_time = timestamps[index - 1], timestamps[index]
        left_value, right_value = glucose[index - 1], glucose[index]
        gap = right_time - left_time
        if (
            pd.isna(left_value)
            or pd.isna(right_value)
            or gap <= pd.Timedelta(0)
            or gap > max_gap
        ):
            continue
        overlap_start = max(left_time, start_time)
        overlap_end = min(right_time, end_time)
        if overlap_end > overlap_start:
            observed_duration += overlap_end - overlap_start
    return float(observed_duration.total_seconds() / 60)


def score_event_alerts(
    alert_timestamps: Iterable[pd.Timestamp],
    event_onsets: Iterable[pd.Timestamp],
    observations: pd.DataFrame | None,
    patient_id: object,
    start: pd.Timestamp,
    end: pd.Timestamp,
    forecast_horizon_minutes: int = EVENT_FORECAST_HORIZON_MINUTES,
    refractory_minutes: int = ALERT_REFRACTORY_MINUTES,
    max_observation_gap_minutes: float = MAX_EVENT_OBSERVATION_GAP_MINUTES,
) -> dict[str, Any]:
    """Score event recall and false-alert episodes per measured observation day."""
    episodes = merge_alert_episodes(alert_timestamps, refractory_minutes)
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    event_end = end_time + pd.Timedelta(minutes=forecast_horizon_minutes)
    in_scope_events = sorted(
        timestamp
        for timestamp in map(pd.Timestamp, event_onsets)
        if start_time < timestamp <= event_end
    )
    matches = match_alert_episodes_to_events(
        episodes, in_scope_events, forecast_horizon_minutes
    )
    matched_alerts = len(matches)
    false_alerts = len(episodes) - matched_alerts
    observed_minutes = (
        measure_observed_minutes(
            observations,
            patient_id,
            start_time,
            end_time,
            max_observation_gap_minutes,
        )
        if observations is not None and "patient_id" in observations
        else None
    )

    event_recall = (
        matched_alerts / len(in_scope_events) if in_scope_events else None
    )
    event_precision = matched_alerts / len(episodes) if episodes else None
    exposure_days = observed_minutes / (60 * 24) if observed_minutes is not None else None
    false_alerts_per_day = (
        false_alerts / exposure_days if exposure_days is not None and exposure_days > 0 else None
    )
    exposure_reason = None
    if observations is None or "patient_id" not in observations:
        exposure_reason = "Libre observation timestamps are unavailable."
    elif not exposure_days:
        exposure_reason = "No measurable eligible Libre observation interval in the evaluation window."

    return {
        "actual_events": len(in_scope_events),
        "alert_episodes": len(episodes),
        "matched_events": matched_alerts,
        "matched_alert_episodes": matched_alerts,
        "false_alert_episodes": false_alerts,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "measured_observation_minutes": observed_minutes,
        "measured_observation_days": exposure_days,
        "false_alert_episodes_per_day": false_alerts_per_day,
        "false_alert_exposure_unavailable_reason": exposure_reason,
        "event_recall_unavailable_reason": (
            None if in_scope_events else "No qualifying event onset in the evaluation horizon."
        ),
        "matches": matches,
    }