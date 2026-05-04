import pandas as pd

def check_season_length(df: pd.DataFrame) -> None:
    # Check that each season has the expected number of matches (380)
    season_counts = df["Season"].value_counts()
    for season, count in season_counts.items():
        if count != 380:
            print(f"Season '{season}' has {count} matches, expected 380.")

def check_season_teams(df: pd.DataFrame) -> None:
    # Check that each season has the expected number of unique teams (20)
    season_teams = df.groupby("Season").apply(lambda x: set(x["HomeTeam"]).union(set(x["AwayTeam"])), include_groups=False)
    for season, teams in season_teams.items():
        if len(teams) != 20:
            print(f"Season '{season}' has {len(teams)} unique teams, expected 20.")

def check_fixture_consistency(df: pd.DataFrame) -> None:
    # Check that each team plays every other team exactly twice per season (once at home, once away)
    seasons = df["Season"].unique()
    for season in seasons:
        season_df = df[df["Season"] == season]
        teams = set(season_df["HomeTeam"]).union(set(season_df["AwayTeam"]))
        for team1 in teams:
            for team2 in teams:
                if team1 != team2:
                    matches = season_df[((season_df["HomeTeam"] == team1) & (season_df["AwayTeam"] == team2)) |
                                        ((season_df["HomeTeam"] == team2) & (season_df["AwayTeam"] == team1))]
                    if len(matches) != 2:
                        print(f"Teams '{team1}' and '{team2}' in season '{season}' have {len(matches)} matches, expected 2.")

def main(df: pd.DataFrame) -> None:
    check_season_length(df)
    check_season_teams(df)
    check_fixture_consistency(df)
    print("Season validation completed.")

if __name__ == "__main__":
    df = pd.read_csv("data/matches/collected.csv")
    main(df)