import pandas as pd
import pytest

import engineer


def _build_sample_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Season": ["2024-25", "2024-25"],
            "Date": ["10/08/2024", "17/08/2024"],
            "HomeTeam": ["TeamA", "TeamB"],
            "AwayTeam": ["TeamB", "TeamA"],
            "FTHG": [2, 0],
            "FTAG": [1, 0],
            "FTR": ["H", "D"],
            "HxG": [1.5, 0.8],
            "AxG": [0.7, 0.9],
            "HS": [10, 7],
            "AS": [8, 9],
            "HST": [5, 2],
            "AST": [3, 4],
            "B365H": [2.0, 2.5],
            "B365D": [3.0, 3.1],
            "B365A": [4.0, 2.9],
            "H_ELO": [1600.0, 1550.0],
            "A_ELO": [1500.0, 1620.0],
        }
    )


def test_engineer_pipeline_adds_core_features() -> None:
    df = _build_sample_matches()

    result = engineer.main(df.copy())

    for column in [
        "ELO_DIFF",
        "H_PROB_ELO",
        "ELO_RATIO",
        "OVERROUND",
        "IMP_H_PROB_ODDS",
        "IMP_A_PROB_ODDS",
        "ODDS_LOG_H",
        "ODDS_LOG_D",
        "ODDS_LOG_A",
        "ODDS_RATIO_HA",
        "ELO_MARKET_DIFF",
        "HPTS_FORM",
        "APTS_FORM",
        "HxG_FORM",
        "AxG_FORM",
        "xG_FORM_DIFF",
        "xG_HOME_AWAY_FORM_DIFF",
    ]:
        assert column in result.columns

    assert result.loc[0, "ELO_DIFF"] == pytest.approx(100.0)
    assert result.loc[1, "HPTS_FORM"] == pytest.approx(0.0)
    assert result.loc[1, "APTS_FORM"] == pytest.approx(3.0)
    assert result.loc[0, "xG_FORM_DIFF"] == pytest.approx(0.0)
