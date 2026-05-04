import pandas as pd

import drop_unwanted


def test_drop_unwanted_removes_leakage_columns() -> None:
    df = pd.DataFrame(
        {
            "ID": [1],
            "Season": ["2024-25"],
            "FTR": ["H"],
            "FTHG": [2],
            "FTAG": [1],
            "HS": [10],
            "AS": [8],
            "HST": [5],
            "AST": [3],
            "HxG": [1.2],
            "AxG": [0.9],
            "Date": ["10/08/2024"],
            "HomeTeam": ["TeamA"],
            "AwayTeam": ["TeamB"],
            "ELO_DIFF": [50.0],
        }
    )

    result = drop_unwanted.main(df.copy())

    for col in [
        "ID",
        "Season",
        "FTHG",
        "FTAG",
        "HS",
        "AS",
        "HST",
        "AST",
        "HxG",
        "AxG",
    ]:
        assert col not in result.columns

    assert "Date" in result.columns
    assert "HomeTeam" in result.columns
    assert "AwayTeam" in result.columns
    assert "ELO_DIFF" in result.columns
