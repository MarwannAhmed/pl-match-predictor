import pandas as pd
import drop_leakage as drop_leakage

def main(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_leakage.drop_leakage_features(df)
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/matches/engineered_features.csv")
    df = main(df)
    df.to_csv("data/matches/preprocessed_features.csv", index=False)