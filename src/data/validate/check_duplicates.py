import pandas as pd

def main(df: pd.DataFrame) -> None:
    duplicate_rows = df[df.duplicated(subset=["Season", "HomeTeam", "AwayTeam"])]
    if not duplicate_rows.empty:
        print(f"Found {len(duplicate_rows)} duplicate rows:")
        print(duplicate_rows)
    else:
        print("Duplicate row checking completed.")

if __name__ == "__main__":
    df = pd.read_csv("data/matches/collected.csv")
    main(df)