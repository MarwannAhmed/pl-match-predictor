import pandas as pd

import check_missing


def test_check_missing_reports_columns_with_missing_values(capsys) -> None:
    df = pd.DataFrame({"FTR": ["H", None], "B365H": [1.9, 2.1]})

    check_missing.main(df)

    captured = capsys.readouterr().out
    assert "Column 'FTR' has 1" in captured
