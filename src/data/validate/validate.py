import pandas as pd
import check_types as check_types
import check_missing as check_missing
import check_duplicates as check_duplicates
import check_seasons as check_seasons
import check_dates as check_dates
import check_teams as check_teams
import check_results as check_results
import check_stats as check_stats

if __name__ == "__main__":
    df = pd.read_csv("data/matches/collected.csv")
    check_types.main(df)
    check_missing.main(df)
    check_duplicates.main(df)
    check_seasons.main(df)
    check_dates.main(df)
    check_teams.main(df)
    check_results.main(df)
    check_stats.main(df)