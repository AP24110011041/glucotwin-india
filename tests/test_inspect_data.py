import pandas as pd
import pytest

from src.inspect_data import inspect_directory


def test_inspects_nested_cgmacros_columns_and_sampling_interval(tmp_path) -> None:
    participant_dir = tmp_path / "participant-files"
    participant_dir.mkdir()
    csv_path = participant_dir / "readings.csv"
    pd.DataFrame(
        {
            "Timestamp": ["01/01/2025 00:00", "01/01/2025 00:05"],
            "Libre GL": [110, 112],
            "Dexcom GL": [109, 111],
            "HR": [65, 66],
        }
    ).to_csv(csv_path, index=False)

    reports = inspect_directory(tmp_path)

    assert len(reports) == 1
    assert reports[0]["columns_by_group"]["cgm"] == ["Libre GL", "Dexcom GL"]
    assert reports[0]["columns_by_group"]["wearable"] == ["HR"]
    assert reports[0]["cgm_sampling"]["median_interval_minutes"] == 5.0


def test_zip_only_directory_explains_manual_extraction(tmp_path) -> None:
    (tmp_path / "CGMacros_dateshifted365.zip").write_bytes(b"not opened by the inspector")

    with pytest.raises(FileNotFoundError, match="Extract the archive manually"):
        inspect_directory(tmp_path)
