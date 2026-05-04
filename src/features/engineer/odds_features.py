import pandas as pd
import numpy as np

def main(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering odds features...")

    RAW_H = 1 / df['B365H']
    RAW_D = 1 / df['B365D']
    RAW_A = 1 / df['B365A']
    
    # Bookmaker's margin
    df['OVERROUND'] = RAW_H + RAW_D + RAW_A
    
    # Normalized probabilities
    df['IMP_H_PROB_ODDS'] = RAW_H / df['OVERROUND']
    df['IMP_A_PROB_ODDS'] = RAW_A / df['OVERROUND']

    # Log transforms to capture non-linear relationships
    df['ODDS_LOG_H'] = np.log(df['B365H'])
    df['ODDS_LOG_D'] = np.log(df['B365D'])
    df['ODDS_LOG_A'] = np.log(df['B365A'])

    df['ODDS_RATIO_HA'] = df['B365H'] / df['B365A']
    df['ELO_MARKET_DIFF'] = df['H_PROB_ELO'] - df['IMP_H_PROB_ODDS']

    print("Odds features engineered successfully.")

    return df

if __name__ == "__main__":
    df = pd.read_csv("data/matches/collected.csv")
    df = main(df)
    df.to_csv("data/matches/odds_features.csv", index=False)