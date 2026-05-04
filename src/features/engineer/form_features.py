import pandas as pd


def _build_team_rows(season_df: pd.DataFrame, is_home: bool) -> pd.DataFrame:
    if is_home:
        team_col = "HomeTeam"
        goals_for_col = "FTHG"
        goals_against_col = "FTAG"
        xg_for_col = "HxG"
        xg_against_col = "AxG"
        shots_for_col = "HS"
        shots_against_col = "AS"
        shots_on_target_for_col = "HST"
        shots_on_target_against_col = "AST"
        win_mask = season_df["FTR"] == "H"
    else:
        team_col = "AwayTeam"
        goals_for_col = "FTAG"
        goals_against_col = "FTHG"
        xg_for_col = "AxG"
        xg_against_col = "HxG"
        shots_for_col = "AS"
        shots_against_col = "HS"
        shots_on_target_for_col = "AST"
        shots_on_target_against_col = "HST"
        win_mask = season_df["FTR"] == "A"

    draw_mask = season_df["FTR"] == "D"
    points = win_mask.astype(int) * 3 + draw_mask.astype(int)

    return pd.DataFrame(
        {
            "match_id": season_df["_match_id"],
            "Date": season_df["Date"],
            "team": season_df[team_col],
            "is_home": is_home,
            "points": points,
            "goals_for": season_df[goals_for_col],
            "goals_against": season_df[goals_against_col],
            "xg_for": season_df[xg_for_col],
            "xg_against": season_df[xg_against_col],
            "shots_for": season_df[shots_for_col],
            "shots_against": season_df[shots_against_col],
            "shots_on_target_for": season_df[shots_on_target_for_col],
            "shots_on_target_against": season_df[shots_on_target_against_col],
            "win": win_mask.astype(int),
            "draw": draw_mask.astype(int),
        }
    )


def _rolling_means(
    long_df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]
) -> pd.DataFrame:
    ordered = long_df.sort_values(["Date", "match_id"])
    rolled = (
        ordered.groupby(group_cols, sort=False)[metric_cols]
        .apply(lambda group: group.shift(1).rolling(5, min_periods=1).mean())
        .reset_index(level=group_cols, drop=True)
    )
    return rolled.fillna(0.0)


def compute_season_form(season_df: pd.DataFrame) -> pd.DataFrame:
    season_df = season_df.sort_values("Date").reset_index().rename(columns={"index": "_orig_index"})
    season_df["_match_id"] = season_df.index

    home_rows = _build_team_rows(season_df, True)
    away_rows = _build_team_rows(season_df, False)
    long_df = pd.concat([home_rows, away_rows], ignore_index=True)

    season_df["H_POS"] = 0
    season_df["A_POS"] = 0
    season_df["H_PTS_UP"] = 0
    season_df["H_PTS_DOWN"] = 0
    season_df["A_PTS_UP"] = 0
    season_df["A_PTS_DOWN"] = 0
    teams = pd.unique(pd.concat([season_df["HomeTeam"], season_df["AwayTeam"]], ignore_index=True))
    for match_date in season_df["Date"].drop_duplicates().sort_values():
        prior_matches = long_df[long_df["Date"] < match_date]
        standings = (
            prior_matches.groupby("team")[["points", "goals_for", "goals_against"]]
            .sum()
            .reindex(teams, fill_value=0)
        )
        standings["goal_diff"] = standings["goals_for"] - standings["goals_against"]
        standings = standings.reset_index().sort_values(
            ["points", "goal_diff", "goals_for", "team"],
            ascending=[False, False, False, True],
        )
        standings["pos"] = range(1, len(standings) + 1)
        if prior_matches.empty:
            standings["pts_up"] = 0.0
            standings["pts_down"] = 0.0
        else:
            standings = standings.set_index("pos")
            points_by_pos = standings["points"]
            standings["pts_up"] = points_by_pos.shift(1) - standings["points"]
            standings["pts_down"] = standings["points"] - points_by_pos.shift(-1)
            standings.loc[1, "pts_up"] = -1.0
            standings.loc[len(standings), "pts_down"] = -1.0
            standings[["pts_up", "pts_down"]] = standings[["pts_up", "pts_down"]].fillna(0.0)
            standings = standings.reset_index()
        pos_map = standings.set_index("team")["pos"].to_dict()
        pts_up_map = standings.set_index("team")["pts_up"].to_dict()
        pts_down_map = standings.set_index("team")["pts_down"].to_dict()
        date_mask = season_df["Date"] == match_date
        season_df.loc[date_mask, "H_POS"] = season_df.loc[date_mask, "HomeTeam"].map(pos_map)
        season_df.loc[date_mask, "A_POS"] = season_df.loc[date_mask, "AwayTeam"].map(pos_map)
        season_df.loc[date_mask, "H_PTS_UP"] = season_df.loc[date_mask, "HomeTeam"].map(pts_up_map)
        season_df.loc[date_mask, "H_PTS_DOWN"] = season_df.loc[date_mask, "HomeTeam"].map(pts_down_map)
        season_df.loc[date_mask, "A_PTS_UP"] = season_df.loc[date_mask, "AwayTeam"].map(pts_up_map)
        season_df.loc[date_mask, "A_PTS_DOWN"] = season_df.loc[date_mask, "AwayTeam"].map(pts_down_map)

    metric_cols = [
        "points",
        "goals_for",
        "goals_against",
        "xg_for",
        "xg_against",
        "shots_for",
        "shots_against",
        "shots_on_target_for",
        "shots_on_target_against",
    ]

    overall_roll = _rolling_means(long_df, ["team"], metric_cols)
    venue_roll = _rolling_means(long_df, ["team", "is_home"], metric_cols)

    ordered_long = long_df.sort_values(["Date", "match_id"]).copy()
    overall_roll.columns = [f"overall_{col}" for col in metric_cols]
    venue_roll.columns = [f"venue_{col}" for col in metric_cols]
    ordered_long = ordered_long.join(overall_roll)
    ordered_long = ordered_long.join(venue_roll)

    home_long = ordered_long[ordered_long["is_home"]].set_index("match_id")
    away_long = ordered_long[~ordered_long["is_home"]].set_index("match_id")

    feature_map = {
        "points": "PTS",
        "goals_for": "G",
        "goals_against": "GC",
        "xg_for": "xG",
        "xg_against": "xGC",
        "shots_for": "S",
        "shots_against": "SC",
        "shots_on_target_for": "ST",
        "shots_on_target_against": "STC",
    }

    season_df = season_df.set_index("_match_id")
    for metric, suffix in feature_map.items():
        season_df[f"H{suffix}_FORM"] = home_long[f"overall_{metric}"]
        season_df[f"A{suffix}_FORM"] = away_long[f"overall_{metric}"]
        season_df[f"H{suffix}_HOME_FORM"] = home_long[f"venue_{metric}"]
        season_df[f"A{suffix}_AWAY_FORM"] = away_long[f"venue_{metric}"]

    season_df = season_df.reset_index(drop=True)
    season_df = season_df.set_index("_orig_index").sort_index()
    return season_df.drop(columns=["_match_id"], errors="ignore")

def compute_form(df: pd.DataFrame) -> pd.DataFrame:

    FORM_FEATURES = ['PTS', 'G', 'GC', 'xG', 'xGC', 'S', 'SC', 'ST', 'STC']

    HOME_FORM_FEATURES = [f"H{feature}_FORM" for feature in FORM_FEATURES]
    AWAY_FORM_FEATURES = [f"A{feature}_FORM" for feature in FORM_FEATURES]
    HOME_HOME_FORM_FEATURES = [f"H{feature}_HOME_FORM" for feature in FORM_FEATURES]
    AWAY_AWAY_FORM_FEATURES = [f"A{feature}_AWAY_FORM" for feature in FORM_FEATURES]

    df[HOME_FORM_FEATURES + AWAY_FORM_FEATURES + HOME_HOME_FORM_FEATURES + AWAY_AWAY_FORM_FEATURES] = 0.0
    df["H_POS"] = 0
    df["A_POS"] = 0
    df["H_PTS_UP"] = 0
    df["H_PTS_DOWN"] = 0
    df["A_PTS_UP"] = 0
    df["A_PTS_DOWN"] = 0

    for season in df['Season'].unique():
        season_mask = df['Season'] == season
        season_df = df[season_mask]

        season_df = compute_season_form(season_df)
        df.loc[season_mask] = season_df

    return df

def main(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering form features...")
    df = compute_form(df)
    print("Form features engineered successfully.")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/matches/collected.csv")
    engineered_df = main(df)
    engineered_df.to_csv("data/matches/form_features.csv", index=False)