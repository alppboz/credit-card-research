import os
import sys
sys.path.append("lib")
import utils

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns
from matplotlib import pyplot as plt


if __name__ == '__main__':
    df_filepath = "output/000_filtered_merchant_trans_df.csv"    
    if os.path.exists(df_filepath):
        filtered_merchant_trans_df = pd.read_csv(df_filepath, index_col=0)
    else:
        trans_df = utils.load_data("transaction")
        filtered_trans_df = trans_df[trans_df["UYEISYERI_ID_MASK"] != 999999]
        num_total = len(trans_df)
        num_filtered = len(filtered_trans_df)
        count_s = filtered_trans_df["UYEISYERI_ID_MASK"].value_counts()
        filtered_merchantid_list = count_s[count_s >= 100].index.tolist()

        filtered_merchant_trans_df = filtered_trans_df[filtered_trans_df.UYEISYERI_ID_MASK.isin(filtered_merchantid_list)]
        filtered_merchant_trans_df["yyyymmdd"] = filtered_merchant_trans_df["ISLEM_TARIHI"].apply(lambda x: dateparser.parse(x.split(":")[0]))
        filtered_merchant_trans_df["yyyymm"] = filtered_merchant_trans_df["yyyymmdd"].apply(lambda x: str(x)[0:7])
        filtered_merchant_trans_df.to_csv(df_filepath, index=True)

    agg_df = filtered_merchant_trans_df.groupby(["UYEISYERI_ID_MASK", "yyyymm"]).agg({"MUSTERI_ID_MASK": ['count', 'nunique'], 'ISLEM_TUTARI': ['sum', 'mean']})
    agg_df.to_csv("output/000_agg_df.csv")
    #agg_df["MUSTERI_ID_MASK"]["count"].unstack()

    # Create cross table
    cross_df = pd.crosstab(filtered_merchant_trans_df["UYEISYERI_ID_MASK"], filtered_merchant_trans_df["MUSTERI_ID_MASK"])
    customer_count_s = cross_df.sum(axis=0)
    filtered_cross_df = cross_df[customer_count_s[customer_count_s >= 30].index.tolist()]
    n_topics = 10
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=100, learning_method='batch',
                                    verbose=0, learning_offset=50, random_state=19999,
                                    n_jobs=1)
    all_Xtopic = lda.fit_transform(filtered_cross_df.as_matrix())
    all_Xtopic_norm = all_Xtopic / all_Xtopic.sum(axis=1)[:, np.newaxis]

    all_Xtopic_df = pd.DataFrame(all_Xtopic_norm)
    all_Xtopic_df.index = filtered_cross_df.index

    maxtopic_s = all_Xtopic_df.apply(lambda x: x.argmax(), axis=1)
    merchant_info_df = filtered_merchant_trans_df.drop_duplicates(subset=["UYEISYERI_ID_MASK"]).set_index("UYEISYERI_ID_MASK")[["ISYERI_TURU", "X", "Y", "DISTRICT_ID"]]
    cur_merchant_info_df = merchant_info_df.ix[maxtopic_s.index]
    cur_merchant_info_df["maxtopic"] = maxtopic_s
    plt.figure(figsize=(96, 96))
    # NOTE: X and Y are swapped in the original data
    sns.lmplot('Y', 'X', data=cur_merchant_info_df, hue='maxtopic', fit_reg=False, scatter_kws={"s": 1})
    plt.savefig("fig/000_topicdist.png", dpi=300)
    plt.close()

    # focused
    plt.figure(figsize=(128, 128))
    sns.lmplot('Y', 'X', data=cur_merchant_info_df, hue='maxtopic', fit_reg=False, scatter_kws={"s": 1})
    #plt.xlim([28.0, 29.5])
    #plt.ylim([40.5, 41.5])
    plt.xlim([28.2, 29.4])
    plt.ylim([40.3, 41.5])
    plt.savefig("fig/000_topicdist_focused.png", dpi=300)
    plt.close()
    
