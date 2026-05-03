import pandas as pd

def check_goals(df: pd.DataFrame) -> None:
    invalid_rows = df[(df["FTHG"] < 0) | (df["FTAG"] < 0)]
    if not invalid_rows.empty:
        print(f"Found {len(invalid_rows)} rows with negative goal counts:")
        print(invalid_rows[["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]])

def check_result_consistency(df: pd.DataFrame) -> None:
    inconsistent_rows = df[(df["FTR"] == "H") & (df["FTHG"] <= df["FTAG"]) |
                           (df["FTR"] == "A") & (df["FTHG"] >= df["FTAG"]) |
                           (df["FTR"] == "D") & (df["FTHG"] != df["FTAG"])]
    if not inconsistent_rows.empty:
        print(f"Found {len(inconsistent_rows)} rows with inconsistent results:")
        print(inconsistent_rows[["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]])

def check_target_imbalance(df: pd.DataFrame) -> None:
    result_counts = df["FTR"].value_counts()
    imbalance_ratio = result_counts.max() / result_counts.min()
    if imbalance_ratio >= 2:
        print(f"Warning: Target variable 'FTR' is imbalanced with ratio {imbalance_ratio:.2f} (max: {result_counts.idxmax()} - {result_counts.max()} | min: {result_counts.idxmin()} - {result_counts.min()}).")

def main(df: pd.DataFrame) -> None:
    check_goals(df)
    check_result_consistency(df)
    check_target_imbalance(df)
    print("Result validation completed.")

if __name__ == "__main__":
    df = pd.read_csv("data/matches/collected.csv")
    main(df)