import pandas as pd
from sklearn.preprocessing import StandardScaler

import drop_unwanted as drop_unwanted
import transform as transform


def split_by_season(df: pd.DataFrame, train_seasons: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_seasons is None:
        seasons = sorted(df["Season"].dropna().unique().tolist())
        train_seasons = seasons[:8]
    train_df = df[df["Season"].isin(train_seasons)].copy()
    test_df = df[~df["Season"].isin(train_seasons)].copy()
    return train_df, test_df


def _points_from_results(df: pd.DataFrame, is_home: bool) -> pd.Series:
    win_mask = df["FTR"] == ("H" if is_home else "A")
    draw_mask = df["FTR"] == "D"
    return win_mask.astype(int) * 3 + draw_mask.astype(int)


def build_team_rankings(train_df: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    home_points = train_df.assign(points=_points_from_results(train_df, True))
    away_points = train_df.assign(points=_points_from_results(train_df, False))

    home_totals = home_points.groupby("HomeTeam")["points"].sum()
    away_totals = away_points.groupby("AwayTeam")["points"].sum()

    all_home = pd.Index(train_df["HomeTeam"].unique())
    all_away = pd.Index(train_df["AwayTeam"].unique())

    home_totals = home_totals.reindex(all_home, fill_value=0)
    away_totals = away_totals.reindex(all_away, fill_value=0)

    home_sorted = home_totals.sort_values(ascending=True)
    away_sorted = away_totals.sort_values(ascending=True)

    home_mapping = {team: idx for idx, team in enumerate(home_sorted.index.tolist())}
    away_mapping = {team: idx for idx, team in enumerate(away_sorted.index.tolist())}
    return home_mapping, away_mapping


def scale_numeric(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if exclude_cols is None:
        exclude_cols = []
    numeric_cols = train_df.select_dtypes(include="number").columns
    scale_cols = [col for col in numeric_cols if col not in exclude_cols]
    scaler = StandardScaler()
    train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
    test_df[scale_cols] = scaler.transform(test_df[scale_cols])
    return train_df, test_df


def main(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = split_by_season(df)

    home_mapping, away_mapping = build_team_rankings(train_df)

    train_df = drop_unwanted.main(train_df)
    test_df = drop_unwanted.main(test_df)

    train_df = transform.main(train_df, home_mapping=home_mapping, away_mapping=away_mapping)
    test_df = transform.main(test_df, home_mapping=home_mapping, away_mapping=away_mapping)

    train_df, test_df = scale_numeric(
        train_df,
        test_df,
        exclude_cols=["HomeTeam", "AwayTeam", "FTR"],
    )
    return train_df, test_df

if __name__ == "__main__":
    df = pd.read_csv("data/matches/engineered_features.csv")
    train_df, test_df = main(df)
    train_df.to_csv("data/matches/preprocessed_train.csv", index=False)
    test_df.to_csv("data/matches/preprocessed_test.csv", index=False)