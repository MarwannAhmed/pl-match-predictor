import pandas as pd

def main(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering xG features...")

    df['xG_FORM_DIFF'] = df['HxG_FORM'] - df['AxG_FORM']
    df['xG_HOME_AWAY_FORM_DIFF'] = df['HxG_HOME_FORM'] - df['AxG_AWAY_FORM']
    df['HxG_FORM_OVERPERF'] = df['HxG_FORM'] - df['HG_FORM']
    df['AxG_FORM_OVERPERF'] = df['AxG_FORM'] - df['AG_FORM']
    df['HxG_HOME_FORM_OVERPERF'] = df['HxG_HOME_FORM'] - df['HG_HOME_FORM']
    df['AxG_AWAY_FORM_OVERPERF'] = df['AxG_AWAY_FORM'] - df['AG_AWAY_FORM']

    print("xG features engineered successfully.")

    return df

if __name__ == "__main__":
    df = pd.read_csv("data/matches/collected.csv")
    df = main(df)
    df.to_csv("data/matches/xg_features.csv", index=False)