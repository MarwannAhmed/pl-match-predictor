import pandas as pd


def _build_alpha_mapping(df: pd.DataFrame) -> dict[str, int]:
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel("K"))
    return {team: idx for idx, team in enumerate(sorted(teams))}


def main(
    df: pd.DataFrame,
    home_mapping: dict[str, int] | None = None,
    away_mapping: dict[str, int] | None = None,
    return_mappings: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    print("Encoding features...")
    print("Columns before encoding:", df.columns.tolist())

    # Extract day and month from the Date column
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df = df.drop(columns=["Date"])

    if "FTR" in df.columns:
        df["FTR"] = df["FTR"].map({"H": 2, "D": 1, "A": 0}).astype("Int64")

    # Encode HomeTeam and AwayTeam using label encoding
    if home_mapping is None:
        home_mapping = _build_alpha_mapping(df)
    if away_mapping is None:
        away_mapping = _build_alpha_mapping(df)

    df["HomeTeam"] = df["HomeTeam"].map(home_mapping).fillna(-1).astype(int)
    df["AwayTeam"] = df["AwayTeam"].map(away_mapping).fillna(-1).astype(int)

    print("Columns after encoding:", df.columns.tolist())
    if return_mappings:
        return df, home_mapping, away_mapping
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/matches/engineered_features.csv")
    df = main(df)
    df.to_csv("data/matches/encoded_features.csv", index=False)