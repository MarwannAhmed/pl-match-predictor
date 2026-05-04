import pandas as pd

import preprocess


def test_preprocess_pipeline_drops_and_encodes() -> None:
    df = pd.DataFrame(
        {
            "ID": [1],
            "Season": ["2024-25"],
            "Date": ["10/08/2024"],
            "HomeTeam": ["TeamA"],
            "AwayTeam": ["TeamB"],
            "FTR": ["H"],
            "FTHG": [2],
            "FTAG": [1],
            "HS": [10],
            "AS": [8],
            "HST": [5],
            "AST": [3],
            "HxG": [1.2],
            "AxG": [0.9],
            "ELO_DIFF": [50.0],
        }
    )

    result = preprocess.main(df.copy())

    for col in [
        "ID",
        "Season",
        "FTR",
        "FTHG",
        "FTAG",
        "HS",
        "AS",
        "HST",
        "AST",
        "HxG",
        "AxG",
        "Date",
    ]:
        assert col not in result.columns

    for col in ["Day", "Month", "DayOfWeek", "HomeTeam", "AwayTeam", "ELO_DIFF"]:
        assert col in result.columns
