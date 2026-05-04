import pandas as pd

def main(df: pd.DataFrame) -> pd.DataFrame:
    print("Dropping leakage and unnecessary columns...")
    df = df.drop(columns=["ID", "Season", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HxG", "AxG"])
    print("Columns after dropping leakage and unnecessary features:", df.columns.tolist())
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/matches/engineered_features.csv")
    df = main(df)
    df.to_csv("data/matches/dropped_unwanted_features.csv", index=False)