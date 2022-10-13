import json
import os
import sys
sys.path.append("lib")
import utils

import numpy as np
import pandas as pd

import dateparser


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Input conf filepath")
        sys.exit(1)
        
    with open(sys.argv[1]) as fin:
        all_conf = json.load(fin)

    conf = all_conf['merchant_filtering']
    df_filepath = conf['output_filepath']
    output_agg_filepath = conf['output_agg_filepath']
    min_transcount = conf['min_transcount']
    remove_online = bool(conf['remove_online'])
    mcc_list = conf['mcc_list']
    min_merchants_mcc = conf['min_merchants_mcc']
    min_month_trans = conf["min_month_trans"]

    if os.path.exists(df_filepath):
        filtered_merchant_trans_df = pd.read_csv(df_filepath, index_col=0)

    else:
        trans_df = utils.load_data("transaction")

        # 1. Remove invalid merchant ID
        filtered_trans_df = trans_df[trans_df["UYEISYERI_ID_MASK"] != 999999]

        # 2. Remove online transaction
        if remove_online:
            filtered_trans_df = filtered_trans_df.query("ONLINE_ISLEM == 0")

        # 3. Filter merchants in the selected merhcnat categories
        filtered_trans_df = filtered_trans_df[
            filtered_trans_df["ISYERI_TURU"].isin(mcc_list)]

        # 4. Filter merchants that do not have sufficient transactions
        num_total = len(trans_df)
        num_filtered = len(filtered_trans_df)

        count_s = filtered_trans_df["UYEISYERI_ID_MASK"].value_counts()
        filtered_merchantid_list = count_s[count_s >= min_transcount].index.tolist()        
        filtered_merchant_trans_df = filtered_trans_df[filtered_trans_df.UYEISYERI_ID_MASK.isin(filtered_merchantid_list)]

        # 5. Filter out merchant categories that do not have 
        #      sufficient amount of merchants.
        mcc_count_s = filtered_trans_df["ISYERI_TURU"].value_counts()
        filtered_mcc_list = mcc_count_s[mcc_count_s >= min_merchants_mcc].index.tolist()
        filtered_merchant_trans_df = filtered_merchant_trans_df[
            filtered_merchant_trans_df["ISYERI_TURU"].isin(filtered_mcc_list)]

        print("Parse yyyymmdd...")
        filtered_merchant_trans_df["yyyymmdd"] = filtered_merchant_trans_df["ISLEM_TARIHI"].apply(lambda x: dateparser.parse(x.split(":")[0]))
        filtered_merchant_trans_df["yyyymm"] = filtered_merchant_trans_df["yyyymmdd"].apply(lambda x: str(x)[0:7])
        print("done.")

        # 6. Filter merchants that have at least one transaction every month
        monthly_agg_df = filtered_merchant_trans_df.groupby(["UYEISYERI_ID_MASK", "yyyymm"]).agg({"MUSTERI_ID_MASK": ['count', 'nunique'],
                                                                                                  'ISLEM_TUTARI': ['sum', 'mean']}).unstack()
        merchants_every_month = monthly_agg_df[monthly_agg_df["MUSTERI_ID_MASK"]["count"].isnull().sum(axis=1) <= (12 - min_month_trans)].index.tolist()
        filtered_merchant_trans_df = filtered_merchant_trans_df[filtered_merchant_trans_df["UYEISYERI_ID_MASK"].isin(merchants_every_month)]

        filtered_merchant_trans_df.to_csv(df_filepath, index=True)

    agg_df = filtered_merchant_trans_df.groupby(["UYEISYERI_ID_MASK", "yyyymm"]).agg({"MUSTERI_ID_MASK": ['count', 'nunique'], 'ISLEM_TUTARI': ['sum', 'mean']})
    agg_df.to_csv(output_agg_filepath)
    #agg_df["MUSTERI_ID_MASK"]["count"].unstack()

    
