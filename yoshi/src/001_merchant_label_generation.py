import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("output/000_agg_df.csv", index_col=[0, 1], header=[0, 1])
    sum_df = df['ISLEM_TUTARI']['sum'].unstack(1)
    sum_df = sum_df.dropna(subset=sum_df.columns)

    first_sum_df = sum_df[sum_df.columns[0:6].tolist()].sum(axis=1)
    second_sum_df = sum_df[sum_df.columns[7:].tolist()].sum(axis=1)

    increase_df = (second_sum_df - first_sum_df) / first_sum_df
    increase_df.hist(bins=50)
    plt.xlim([-2, 4])
    plt.savefig("fig/001_revenue_sum_increaserate_dist_hist50bins.png")
    plt.close()

    bins = np.linspace(-2, 4, 50)
    top25 = increase_df[increase_df >= increase_df.quantile(0.75)]
    bottom25 = increase_df[increase_df >= increase_df.quantile(0.25)]

    plt.hist(top25, bins, alpha=0.5, label='top 25%')
    plt.hist(bottom25, bins, alpha=0.5, label='bottom 25%')
    plt.savefig("fig/001_top25-bottom25_hist.png")
    plt.close()


    relative_sum_df = sum_df.div(sum_df.mean(axis=1), axis=0)
    matplotlib.style.use('ggplot')
    for index, row in relative_sum_df.iterrows():
        row.plot(kind="bar")
        plt.axhline(1, color='red')
        plt.savefig("fig/001_each_merchant_relative_sum/{}.png".format(index))
        plt.close()

