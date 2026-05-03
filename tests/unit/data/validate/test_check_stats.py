import pandas as pd

import check_stats


def test_check_stats_reports_negative_values(capsys) -> None:
    df = pd.DataFrame(
        {
            "Season": ["2024-25"],
            "Date": ["10/08/2024"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "HS": [-1],
            "AS": [8],
            "HST": [2],
            "AST": [2],
            "HC": [4],
            "AC": [5],
            "HY": [1],
            "AY": [2],
            "HR": [0],
            "AR": [0],
            "B365H": [2.1],
            "B365D": [3.2],
            "B365A": [3.7],
            "H_ELO": [1600.0],
            "A_ELO": [1580.0],
            "HxG": [1.2],
            "AxG": [0.9],
        }
    )

    check_stats.check_stats(df)

    captured = capsys.readouterr().out
    assert "negative stats" in captured


def test_check_shots_reports_inconsistent_counts(capsys) -> None:
    df = pd.DataFrame(
        {
            "Season": ["2024-25"],
            "Date": ["10/08/2024"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "HS": [5],
            "AS": [4],
            "HST": [6],
            "AST": [2],
        }
    )

    check_stats.check_shots(df)

    captured = capsys.readouterr().out
    assert "inconsistent shot counts" in captured
