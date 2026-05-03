import pandas as pd

def main(df: pd.DataFrame) -> None:
    missing_counts = df.isna().sum()
    for column, count in missing_counts.items():
        if count > 0:
            print(f"Column '{column}' has {count} ({count / len(df) * 100:.2f}%) missing values")
    print("Missing value checking completed.")

if __name__ == "__main__":
    df = pd.read_csv("data/matches/collected.csv")
    main(df)