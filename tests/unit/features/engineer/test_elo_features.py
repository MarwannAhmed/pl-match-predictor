import pandas as pd
import pytest

import elo_features


def test_elo_features_adds_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "H_ELO": [1600.0],
            "A_ELO": [1500.0],
        }
    )

    result = elo_features.main(df.copy())

    assert result.loc[0, "ELO_DIFF"] == pytest.approx(100.0)
    expected_prob = 1 / (1 + 10 ** (-(100.0 + 65) / 400))
    assert result.loc[0, "H_PROB_ELO"] == pytest.approx(expected_prob)
    assert result.loc[0, "ELO_RATIO"] == pytest.approx(1600.0 / 1500.0)
