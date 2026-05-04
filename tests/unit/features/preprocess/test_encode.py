import pandas as pd
import pytest

import encode


def test_encode_extracts_date_parts_and_drops_date() -> None:
    df = pd.DataFrame(
        {
            "Date": ["10/08/2024"],
            "HomeTeam": ["TeamA"],
            "AwayTeam": ["TeamB"],
        }
    )

    result = encode.main(df.copy())

    assert "Date" not in result.columns
    assert result.loc[0, "Day"] == 10
    assert result.loc[0, "Month"] == 8
    assert result.loc[0, "DayOfWeek"] == 5


def test_encode_handles_invalid_dates() -> None:
    df = pd.DataFrame({"Date": ["not-a-date"]})

    result = encode.main(df.copy())

    assert "Date" not in result.columns
    assert result.loc[0, "Day"] is pd.NA or pd.isna(result.loc[0, "Day"])
    assert result.loc[0, "Month"] is pd.NA or pd.isna(result.loc[0, "Month"])
    assert result.loc[0, "DayOfWeek"] is pd.NA or pd.isna(result.loc[0, "DayOfWeek"])
