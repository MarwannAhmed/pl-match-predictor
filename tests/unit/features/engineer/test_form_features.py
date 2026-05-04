import pandas as pd
import pytest

import form_features


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
        }
    )


def test_compute_form_generates_rollups_and_positions() -> None:
    df = _build_sample_matches()

    result = form_features.compute_form(df.copy())

    first_match = result.iloc[0]
    second_match = result.iloc[1]

    assert first_match["HPTS_FORM"] == pytest.approx(0.0)
    assert first_match["APTS_FORM"] == pytest.approx(0.0)

    assert second_match["HPTS_FORM"] == pytest.approx(0.0)
    assert second_match["HG_FORM"] == pytest.approx(1.0)
    assert second_match["HGC_FORM"] == pytest.approx(2.0)
    assert second_match["HxG_FORM"] == pytest.approx(0.7)
    assert second_match["HxGC_FORM"] == pytest.approx(1.5)
    assert second_match["HS_FORM"] == pytest.approx(8.0)
    assert second_match["HSC_FORM"] == pytest.approx(10.0)
    assert second_match["HST_FORM"] == pytest.approx(3.0)
    assert second_match["HSTC_FORM"] == pytest.approx(5.0)

    assert second_match["APTS_FORM"] == pytest.approx(3.0)
    assert second_match["AG_FORM"] == pytest.approx(2.0)
    assert second_match["AGC_FORM"] == pytest.approx(1.0)
    assert second_match["AxG_FORM"] == pytest.approx(1.5)
    assert second_match["AxGC_FORM"] == pytest.approx(0.7)
    assert second_match["AS_FORM"] == pytest.approx(10.0)
    assert second_match["ASC_FORM"] == pytest.approx(8.0)
    assert second_match["AST_FORM"] == pytest.approx(5.0)
    assert second_match["ASTC_FORM"] == pytest.approx(3.0)

    assert second_match["H_POS"] == 2
    assert second_match["A_POS"] == 1
    assert second_match["H_PTS_UP"] == pytest.approx(3.0)
    assert second_match["H_PTS_DOWN"] == pytest.approx(-1.0)
    assert second_match["A_PTS_UP"] == pytest.approx(-1.0)
    assert second_match["A_PTS_DOWN"] == pytest.approx(3.0)
