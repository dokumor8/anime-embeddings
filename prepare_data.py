import pandas as pd


def main():
    df = pd.read_csv("./data/reviews_100.csv")
    df_subset = df.head(50)
    df_subset.to_csv("./data/reviews_100_2.csv", index=False)


main()
