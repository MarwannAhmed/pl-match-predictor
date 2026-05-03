import pandas as pd
import pytest

import check_types


def test_validate_type_accepts_expected_string_dtype() -> None:
    df = pd.DataFrame({"Season": ["2024-25", "2023-24"]})

    check_types.validate_type(df, "Season", "string")


def test_validate_type_raises_for_wrong_dtype() -> None:
    df = pd.DataFrame({"FTHG": [1.5, 2.0]})

    with pytest.raises(TypeError):
        check_types.validate_type(df, "FTHG", "int")
