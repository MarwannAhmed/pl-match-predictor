import numpy as np
import pandas as pd
import pytest

import odds_features


def test_odds_features_computes_probabilities_and_logs() -> None:
    df = pd.DataFrame(
        {
            "B365H": [2.0],
            "B365D": [3.0],
            "B365A": [4.0],
            "H_PROB_ELO": [0.6],
        }
    )

    result = odds_features.main(df.copy())

    raw_h = 1 / 2.0
    raw_d = 1 / 3.0
    raw_a = 1 / 4.0
    overround = raw_h + raw_d + raw_a

    assert result.loc[0, "OVERROUND"] == pytest.approx(overround)
    assert result.loc[0, "IMP_H_PROB_ODDS"] == pytest.approx(raw_h / overround)
    assert result.loc[0, "IMP_A_PROB_ODDS"] == pytest.approx(raw_a / overround)
    assert result.loc[0, "ODDS_LOG_H"] == pytest.approx(np.log(2.0))
    assert result.loc[0, "ODDS_LOG_D"] == pytest.approx(np.log(3.0))
    assert result.loc[0, "ODDS_LOG_A"] == pytest.approx(np.log(4.0))
    assert result.loc[0, "ODDS_RATIO_HA"] == pytest.approx(2.0 / 4.0)
    assert result.loc[0, "ELO_MARKET_DIFF"] == pytest.approx(0.6 - (raw_h / overround))
