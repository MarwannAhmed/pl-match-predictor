import pandas as pd

import check_dates


def test_check_date_format_rejects_invalid_date() -> None:
    df = pd.DataFrame({"Date": ["10/08/2024", "31/02/2024"]})

    assert check_dates.check_date_format(df) is False


def test_check_season_dates_reports_out_of_range_dates(capsys) -> None:
    df = pd.DataFrame(
        {
            "Season": ["2024-25"],
            "Date": ["01/06/2024"],
        }
    )

    check_dates.check_season_dates(df)

    captured = capsys.readouterr().out
    assert "out of expected range" in captured
