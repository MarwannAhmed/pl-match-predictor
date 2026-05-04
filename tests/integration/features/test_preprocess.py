import pandas as pd
import pytest

import preprocess


def test_preprocess_pipeline_drops_and_encodes() -> None:
    rows = []
    for idx in range(9):
        season_start = 2015 + idx
        season_label = f"{season_start}-{str(season_start + 1)[-2:]}"
        rows.append(
            {
                "ID": idx + 1,
                "Season": season_label,
                "Date": "10/08/2024",
                "HomeTeam": f"Team{idx}",
                "AwayTeam": f"Team{idx + 1}",
                "FTR": "H",
                "FTHG": 2,
                "FTAG": 1,
                "HS": 10,
                "AS": 8,
                "HST": 5,
                "AST": 3,
                "HxG": 1.2,
                "AxG": 0.9,
                "ELO_DIFF": float(idx + 1),
            }
        )

    df = pd.DataFrame(rows)

    train_df, test_df = preprocess.main(df.copy())

    assert len(train_df) == 8
    assert len(test_df) == 1

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
        "Date",
    ]:
        assert col not in train_df.columns
        assert col not in test_df.columns

    for col in [
        "Day",
        "Month",
        "DayOfWeek",
        "HomeTeam",
        "AwayTeam",
        "ELO_DIFF",
        "FTR",
    ]:
        assert col in train_df.columns
        assert col in test_df.columns

    assert train_df["ELO_DIFF"].mean() == pytest.approx(0.0, abs=1e-7)
