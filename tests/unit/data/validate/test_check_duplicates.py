import pandas as pd

import check_duplicates


def test_check_duplicates_reports_duplicate_rows(capsys) -> None:
    df = pd.DataFrame(
        {
            "Season": ["2024-25", "2024-25"],
            "HomeTeam": ["Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Chelsea"],
        }
    )

    check_duplicates.main(df)

    captured = capsys.readouterr().out
    assert "duplicate rows" in captured
