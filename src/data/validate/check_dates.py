import pandas as pd

def check_date_format(df: pd.DataFrame) -> bool:
    # Check if all dates in the DataFrame are in the correct format
    for date_str in df["Date"]:
        try:
            pd.to_datetime(date_str, format="%d/%m/%Y")
        except ValueError:
            return False
    return True

def check_season_dates(df: pd.DataFrame) -> None:
    # Check that for any season, all dates are within the expected range
    seasons = df["Season"].unique()
    for season in seasons:
        season_df = df[df["Season"] == season]
        start_year, end_year = map(int, season.split("-"))
        season_start = pd.Timestamp(f"{start_year}-08-01")
        if end_year == 2020: # Special case for 2019-20 season which ended later due to COVID-19
            season_end = pd.Timestamp(f"{end_year}-07-31")
        else:
            season_end = pd.Timestamp(f"{end_year}-05-31")
        
        for date_str in season_df["Date"]:
            date = pd.to_datetime(date_str, format="%d/%m/%Y")
            if not (season_start <= date <= season_end):
                print(f"Date '{date_str}' in season '{season}' is out of expected range ({season_start.date()} to {season_end.date()})")

def main(df: pd.DataFrame) -> None:
    check_date_format(df)
    check_season_dates(df)
    print("Date validation completed.")

if __name__ == "__main__":
    df = pd.read_csv("data/matches/collected.csv")
    main(df)