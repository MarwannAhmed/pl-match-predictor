import pandas as pd

import check_results


def test_check_result_consistency_reports_mismatched_scores(capsys) -> None:
    df = pd.DataFrame(
        {
            "Season": ["2024-25"],
            "Date": ["10/08/2024"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTHG": [1],
            "FTAG": [2],
            "FTR": ["H"],
        }
    )

    check_results.check_result_consistency(df)

    captured = capsys.readouterr().out
    assert "inconsistent results" in captured
