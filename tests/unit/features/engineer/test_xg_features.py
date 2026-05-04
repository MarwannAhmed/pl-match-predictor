import pandas as pd
import pytest

import xg_features


def test_xg_features_computes_differences_and_overperformance() -> None:
    df = pd.DataFrame(
        {
            "HxG_FORM": [1.2],
            "AxG_FORM": [0.8],
            "HxG_HOME_FORM": [1.1],
            "AxG_AWAY_FORM": [0.7],
            "HG_FORM": [1.0],
            "AG_FORM": [0.6],
            "HG_HOME_FORM": [0.9],
            "AG_AWAY_FORM": [0.5],
        }
    )

    result = xg_features.main(df.copy())

    assert result.loc[0, "xG_FORM_DIFF"] == pytest.approx(0.4)
    assert result.loc[0, "xG_HOME_AWAY_FORM_DIFF"] == pytest.approx(0.4)
    assert result.loc[0, "HxG_FORM_OVERPERF"] == pytest.approx(0.2)
    assert result.loc[0, "AxG_FORM_OVERPERF"] == pytest.approx(0.2)
    assert result.loc[0, "HxG_HOME_FORM_OVERPERF"] == pytest.approx(0.2)
    assert result.loc[0, "AxG_AWAY_FORM_OVERPERF"] == pytest.approx(0.2)
