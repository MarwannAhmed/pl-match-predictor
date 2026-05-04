import pandas as pd

def main(df: pd.DataFrame) -> pd.DataFrame:

    print("Engineering ELO features...")

    df["ELO_DIFF"] = df["H_ELO"] - df["A_ELO"]

    home_advantage = 65
    df["H_PROB_ELO"] = 1 / (1 + 10 ** (-( df['ELO_DIFF'] + home_advantage) / 400))
    print("ELO features engineered successfully.")
    
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/matches/collected.csv")
    df = main(df)
    df.to_csv("data/matches/elo_features.csv", index=False)