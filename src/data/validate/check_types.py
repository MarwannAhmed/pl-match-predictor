import pandas as pd


def _is_string(series: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(series)


def _is_integer(series: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(series)


def _is_float(series: pd.Series) -> bool:
    return pd.api.types.is_float_dtype(series)


def validate_type(df: pd.DataFrame, column: str, kind: str) -> None:
    series = df[column]
    if kind == "string" and not _is_string(series):
        raise TypeError(
            f"Column '{column}' has dtype {series.dtype}, expected string"
        )
    if kind == "int" and not _is_integer(series):
        raise TypeError(
            f"Column '{column}' has dtype {series.dtype}, expected integer"
        )
    if kind == "float" and not _is_float(series):
        raise TypeError(
            f"Column '{column}' has dtype {series.dtype}, expected float"
        )


def validate_types(df: pd.DataFrame, expected_kinds: dict[str, str]) -> None:
    for column, kind in expected_kinds.items():
        try:
            validate_type(df, column, kind)
        except TypeError as exc:
            print(f"Error in column '{column}': {exc}")


def main(df: pd.DataFrame) -> None:
    expected_kinds = {
        "ID": "int",
        "Season": "string",
        "Date": "string",
        "HomeTeam": "string",
        "AwayTeam": "string",
        "FTR": "string",
        "FTHG": "int",
        "FTAG": "int",
        "HS": "int",
        "AS": "int",
        "HST": "int",
        "AST": "int",
        "B365H": "float",
        "B365D": "float",
        "B365A": "float",
        "HxG": "float",
        "AxG": "float",
        "H_ELO": "float",
        "A_ELO": "float",
    }
    validate_types(df, expected_kinds)
    print("Type checking completed.")


if __name__ == "__main__":
    df = pd.read_csv("data/matches/collected.csv")
    main(df)