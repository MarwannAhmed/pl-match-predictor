import pandas as pd
import elo_features as elo_features
import odds_features as odds_features
import form_features as form_features
import xg_features as xg_features

def main(df: pd.DataFrame) -> pd.DataFrame:
    df = elo_features.main(df)
    df = odds_features.main(df)
    df = form_features.main(df)
    df = xg_features.main(df)
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/matches/collected.csv")
    df = main(df)
    df.to_csv("data/matches/engineered_features.csv", index=False)