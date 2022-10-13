import pandas as pd

if __name__ == "__main__":
    df1 = pd.read_csv("labels/f6month-slopes.csv", index_col=0)
    df2 = pd.read_csv("labels/label2outof3slopes.csv", index_col=0)
    df3 = pd.read_csv("labels/last6m-median.csv", index_col=0)
    df4 = pd.read_csv("labels/monthly-rev-per-visit.csv", index_col=0)
    df5 = pd.read_csv("labels/monthly-trx-count.csv", index_col=0)
    df6 = pd.read_csv("labels/yearly-slopes.csv", index_col=0)

    df2.columns = ["label2outof3slopes"]
    df3.columns = ["last6m-median"]
