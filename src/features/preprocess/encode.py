import pandas as pd

def main(df: pd.DataFrame) -> pd.DataFrame:
    print("Encoding features...")
    print("Columns before encoding:", df.columns.tolist())

    # Extract day and month from the Date column
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df = df.drop(columns=["Date"])

    # Encode HomeTeam and AwayTeam using label encoding

    print("Columns after encoding:", df.columns.tolist())
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/matches/engineered_features.csv")
    df = main(df)
    df.to_csv("data/matches/encoded_features.csv", index=False)