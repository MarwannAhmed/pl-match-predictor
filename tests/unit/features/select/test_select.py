import pandas as pd

import _select as select


def test_correlation_filter_prefers_ranked_feature() -> None:
    df = pd.DataFrame(
        {
            "ELO_DIFF": [1.0, 2.0, 3.0, 4.0],
            "IMP_H_PROB_ODDS": [1.0, 2.0, 3.0, 4.0],
            "OTHER": [4.0, 3.0, 2.0, 1.0],
        }
    )

    filtered, removed = select.correlation_filter(
        df,
        threshold=0.85,
        preferred_order=["ELO_DIFF", "IMP_H_PROB_ODDS"],
    )

    assert "ELO_DIFF" in filtered.columns
    assert "IMP_H_PROB_ODDS" not in filtered.columns
    assert "IMP_H_PROB_ODDS" in removed


def test_drop_bottom_importance_removes_expected_features() -> None:
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [1, 1, 1],
            "C": [2, 2, 2],
            "D": [3, 3, 3],
        }
    )
    perm_df = pd.DataFrame(
        {
            "Feature": ["A", "B", "C", "D"],
            "Importance_Mean": [0.4, 0.3, 0.2, 0.1],
        }
    )

    filtered, removed = select.drop_bottom_importance(df, perm_df, drop_fraction=0.25)

    assert "D" in removed
    assert "D" not in filtered.columns
