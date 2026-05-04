import pandas as pd

def main(df: pd.DataFrame) -> pd.DataFrame:
    print("Dropping leakage columns...")
    df = df.drop(columns=["FTR", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HxG", "AxG"])
    print("Columns after dropping leakage features:", df.columns.tolist())
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/matches/engineered_features.csv")
    df = main(df)
    df.to_csv("data/matches/dropped_leakage_features.csv", index=False)