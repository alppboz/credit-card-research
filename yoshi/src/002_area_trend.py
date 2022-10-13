import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

if __name__ == '__main__':
    poi_df = pd.read_csv('data/poi-Istanbul-filtered.csv')
    df = pd.read_csv("output/000_agg_df.csv", index_col=[0, 1], header=[0, 1])

    sum_df = df['ISLEM_TUTARI']['sum'].unstack(1)
    sum_df = sum_df.dropna(subset=sum_df.columns)

    first_sum_df = sum_df[sum_df.columns[0:6].tolist()].sum(axis=1)
    second_sum_df = sum_df[sum_df.columns[7:].tolist()].sum(axis=1)

    # increase_df = (second_sum_df - first_sum_df) / first_sum_df
    # increase_df.hist(bins=50)

    ##
    merge_df = pd.merge(sum_df,
                        poi_df[["merchant_id", "district_id", "mcc"]],
                        left_index=True, right_on="merchant_id")
    dist_sum_df = merge_df.groupby("district_id").sum().drop(["merchant_id",
                                                              "mcc"], axis=1)
    dist_relative_sum_df = dist_sum_df.div(dist_sum_df.mean(axis=1), axis=0)
    matplotlib.style.use('ggplot')
    for index, row in dist_relative_sum_df.iterrows():
        row.plot(kind="bar")
        plt.axhline(1, color='red')
        plt.savefig("fig/002_each_area_relative_sum/{}.png".format(index))
        plt.close()

    dist_seasonal_sum_df = dist_sum_df.div(dist_sum_df.mean(axis=0), axis=1)
    for index, row in dist_seasonal_sum_df.iterrows():
        row.plot(kind="bar")
        plt.savefig("fig/002_each_area_seasonal_sum/{}.png".format(index))
        plt.close()

    dist_seasonal_relative_sum_df = dist_seasonal_sum_df.div(
        dist_seasonal_sum_df.mean(axis=1), axis=0)
    for index, row in dist_seasonal_relative_sum_df.iterrows():
        row.plot(kind="bar")
        plt.axhline(1, color='red')
        plt.savefig("fig/002_each_area_seasonal_relative_sum/{}.png".format(index))
        plt.close()


    dist_normalized_sum_df = dist_sum_df / dist_sum_df.sum().sum()
    for index, row in dist_normalized_sum_df.iterrows():
        row.plot(kind="bar")
        plt.savefig("fig/002_each_area_normalized_sum/{}.png".format(index))
        plt.close()


