from __future__ import annotations

import pandas as pd
import pytest

from src.event_metrics import (
    detect_libre_event_onsets,
    match_alert_episodes_to_events,
    measure_observed_minutes,
    merge_alert_episodes,
    score_event_alerts,
)


def _observations(values: list[float | None]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": "001",
            "timestamp": pd.date_range("2025-01-01", periods=len(values), freq="min"),
            "libre_gl_mg_dl": values,
        }
    )


def test_detects_one_event_for_a_prolonged_high_run() -> None:
    values: list[float | None] = [100.0] * 61 + [190.0] * 20

    events = detect_libre_event_onsets(_observations(values))

    assert events["event_onset"].tolist() == [pd.Timestamp("2025-01-01 01:01")]


def test_detects_two_events_only_after_two_observed_low_runs() -> None:
    values: list[float | None] = [100.0] * 61 + [190.0] * 5 + [100.0] * 61 + [190.0]

    events = detect_libre_event_onsets(_observations(values))

    assert events["event_onset"].tolist() == [
        pd.Timestamp("2025-01-01 01:01"),
        pd.Timestamp("2025-01-01 02:07"),
    ]


def test_exactly_60_minutes_in_range_qualifies_but_59_minutes_does_not() -> None:
    qualifying = _observations([100.0] * 61 + [181.0])
    too_short = _observations([100.0] * 60 + [181.0])

    assert detect_libre_event_onsets(qualifying)["event_onset"].tolist() == [
        pd.Timestamp("2025-01-01 01:01")
    ]
    assert detect_libre_event_onsets(too_short).empty


def test_exactly_180_is_low_and_only_a_later_value_above_180_is_an_onset() -> None:
    qualifying = _observations([100.0] * 60 + [180.0, 181.0])
    no_onset_at_threshold = _observations([100.0] * 60 + [180.0])

    assert detect_libre_event_onsets(qualifying)["event_onset"].tolist() == [
        pd.Timestamp("2025-01-01 01:01")
    ]
    assert detect_libre_event_onsets(no_onset_at_threshold).empty


def test_ignores_event_already_in_progress_at_series_start() -> None:
    values: list[float | None] = [190.0] * 10 + [100.0] * 61 + [190.0]

    events = detect_libre_event_onsets(_observations(values))

    assert events["event_onset"].tolist() == [pd.Timestamp("2025-01-01 01:11")]


def test_missing_observation_breaks_the_required_low_run() -> None:
    values: list[float | None] = [100.0] * 40 + [None] + [100.0] * 20 + [190.0]

    events = detect_libre_event_onsets(_observations(values))

    assert events.empty


def test_timestamp_gap_breaks_the_required_low_run() -> None:
    before_gap = pd.DataFrame(
        {
            "patient_id": ["001"] * 41,
            "timestamp": pd.date_range("2025-01-01", periods=41, freq="min"),
            "libre_gl_mg_dl": [100.0] * 41,
        }
    )
    after_gap = pd.DataFrame(
        {
            "patient_id": ["001"],
            "timestamp": [pd.Timestamp("2025-01-01 00:42")],
            "libre_gl_mg_dl": [190.0],
        }
    )

    events = detect_libre_event_onsets(pd.concat([before_gap, after_gap], ignore_index=True))

    assert events.empty


def test_low_runs_do_not_cross_participant_boundaries() -> None:
    observations = pd.DataFrame(
        {
            "patient_id": ["001"] * 40 + ["002"] * 21,
            "timestamp": list(pd.date_range("2025-01-01", periods=40, freq="min"))
            + list(pd.date_range("2025-01-01 00:40", periods=21, freq="min")),
            "libre_gl_mg_dl": [100.0] * 60 + [190.0],
        }
    )

    assert detect_libre_event_onsets(observations).empty


def test_duplicate_patient_timestamps_are_rejected() -> None:
    observations = pd.concat(
        [_observations([100.0]), _observations([190.0])], ignore_index=True
    )

    with pytest.raises(ValueError, match="duplicate patient timestamps"):
        detect_libre_event_onsets(observations)


def test_refractory_merges_alerts_at_or_within_60_minutes() -> None:
    start = pd.Timestamp("2025-01-01")
    alerts = [start + pd.Timedelta(minutes=value) for value in (0, 30, 60, 61, 122)]

    episodes = merge_alert_episodes(alerts)

    assert episodes == [
        start,
        start + pd.Timedelta(minutes=61),
        start + pd.Timedelta(minutes=122),
    ]


def test_one_alert_cannot_match_two_events() -> None:
    start = pd.Timestamp("2025-01-01")
    matches = match_alert_episodes_to_events(
        [start],
        [start + pd.Timedelta(minutes=30), start + pd.Timedelta(minutes=60)],
    )

    assert matches == [(start, start + pd.Timedelta(minutes=30))]


def test_alert_at_onset_does_not_match_but_matches_later_future_event() -> None:
    start = pd.Timestamp("2025-01-01")
    later_event = start + pd.Timedelta(minutes=60)

    matches = match_alert_episodes_to_events(
        [start], [start, later_event]
    )

    assert matches == [(start, later_event)]


def test_multiple_alerts_cannot_count_one_future_event_twice() -> None:
    start = pd.Timestamp("2025-01-01")
    alerts = merge_alert_episodes(
        [start, start + pd.Timedelta(minutes=61)]
    )
    event = start + pd.Timedelta(minutes=70)

    matches = match_alert_episodes_to_events(alerts, [event])

    assert len(matches) == 1
    assert matches == [(start, event)]


def test_alerts_inside_and_outside_forecast_horizon_match_separately() -> None:
    start = pd.Timestamp("2025-01-01")
    alerts = [start, start + pd.Timedelta(minutes=200)]
    events = [start + pd.Timedelta(minutes=120), start + pd.Timedelta(minutes=319)]

    matches = match_alert_episodes_to_events(alerts, events)

    assert matches == [
        (alerts[0], events[0]),
        (alerts[1], events[1]),
    ]
    assert not match_alert_episodes_to_events(
        [alerts[1]], [start + pd.Timedelta(minutes=321)]
    )


def test_exposure_excludes_missing_rows_and_gaps_over_one_minute() -> None:
    start = pd.Timestamp("2025-01-01")
    observations = pd.DataFrame(
        {
            "patient_id": ["001"] * 13,
            "timestamp": [
                start + pd.Timedelta(minutes=value)
                for value in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 21)
            ],
            "libre_gl_mg_dl": [100.0] * 13,
        }
    )
    observations.loc[5, "libre_gl_mg_dl"] = None

    observed_minutes = measure_observed_minutes(
        observations, "001", start, start + pd.Timedelta(minutes=21)
    )

    assert observed_minutes == 9.0


def test_exposure_clips_partial_intervals_to_irregular_anchor_bounds() -> None:
    start = pd.Timestamp("2025-01-01")
    observations = _observations([100.0] * 11)
    scoring_start = start + pd.Timedelta(minutes=1, seconds=30)
    scoring_end = start + pd.Timedelta(minutes=7, seconds=15)

    observed_minutes = measure_observed_minutes(
        observations, "001", scoring_start, scoring_end
    )
    score = score_event_alerts(
        [scoring_start], [], observations, "001", scoring_start, scoring_end
    )

    assert observed_minutes == 5.75
    assert score["measured_observation_minutes"] == 5.75
    assert score["measured_observation_days"] == pytest.approx(5.75 / 1440)


def test_false_alerts_per_day_uses_measured_time_or_is_explicitly_unavailable() -> None:
    start = pd.Timestamp("2025-01-01")
    observations = _observations([100.0] * 61)
    score = score_event_alerts(
        [start],
        [],
        observations,
        "001",
        start,
        start + pd.Timedelta(minutes=60),
    )

    assert score["actual_events"] == 0
    assert score["event_recall"] is None
    assert score["measured_observation_minutes"] == 60.0
    assert score["false_alert_episodes_per_day"] == 24.0

    no_interval = score_event_alerts(
        [start], [], observations, "001", start, start
    )
    assert no_interval["false_alert_episodes_per_day"] is None
    assert no_interval["false_alert_exposure_unavailable_reason"]
