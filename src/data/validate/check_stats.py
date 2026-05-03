import pandas as pd

def check_stats(df: pd.DataFrame) -> None:
    # Check that all stats columns are non-negative
    stats_columns = ["HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR", "B365H", "B365D", "B365A", "H_ELO", "A_ELO", "HxG", "AxG"]
    invalid_rows = df[(df[stats_columns] < 0).any(axis=1)]
    if not invalid_rows.empty:
        print(f"Found {len(invalid_rows)} rows with negative stats:")
        print(invalid_rows[["Season", "Date", "HomeTeam", "AwayTeam"] + stats_columns])

def check_shots(df: pd.DataFrame) -> None:
    # Check that shots on target are not greater than total shots
    invalid_rows = df[(df["HST"] > df["HS"]) | (df["AST"] > df["AS"])]
    if not invalid_rows.empty:
        print(f"Found {len(invalid_rows)} rows with inconsistent shot counts:")
        print(invalid_rows[["Season", "Date", "HomeTeam", "AwayTeam", "HS", "AS", "HST", "AST"]])

def main(df: pd.DataFrame) -> None:
    check_stats(df)
    check_shots(df)
    print("Stats validation completed.")

if __name__ == "__main__":
    df = pd.read_csv("data/matches/collected.csv")
    main(df)