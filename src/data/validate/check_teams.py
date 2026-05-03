import pandas as pd

def check_home_away_teams(df: pd.DataFrame) -> None:
    # Check that HomeTeam and AwayTeam are not the same
    invalid_rows = df[df["HomeTeam"] == df["AwayTeam"]]
    if not invalid_rows.empty:
        print(f"Found {len(invalid_rows)} rows where HomeTeam and AwayTeam are the same:")
        print(invalid_rows[["Season", "Date", "HomeTeam", "AwayTeam"]])

def check_team_duplicates(df: pd.DataFrame) -> None:
    # Check that no team plays more than once on the same date, whether home or away
    duplicate_rows = df[df.duplicated(subset=["Season", "Date", "HomeTeam"], keep=False) |
                        df.duplicated(subset=["Season", "Date", "AwayTeam"], keep=False)]
    if not duplicate_rows.empty:
        print(f"Found {len(duplicate_rows)} rows where teams play more than once on the same day:")
        print(duplicate_rows[["Season", "Date", "HomeTeam", "AwayTeam"]])

def main(df: pd.DataFrame) -> None:
    check_home_away_teams(df)
    check_team_duplicates(df)
    print("Team validation completed.")