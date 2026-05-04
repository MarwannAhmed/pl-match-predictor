import pandas as pd
import drop_unwanted as drop_unwanted
import encode as encode

def main(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_unwanted.main(df)
    df = encode.main(df)
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/matches/engineered_features.csv")
    df = main(df)
    df.to_csv("data/matches/preprocessed_features.csv", index=False)