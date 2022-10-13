import numpy as np
import pandas as pd

import utils

def calc_entropy(s):
    count_s = s.value_counts()
    prob_s = count_s / count_s.sum()
    return (prob_s * np.log(1.0 / prob_s)).sum()

if __name__ == '__main__':
    df = pd.read_csv("output/000_filtered_merchant_trans_df.csv")
    demographic_df = utils.load_data('demographic')

    # set period
    first6m_list = ['2014-07',
                    '2014-08',
                    '2014-09',
                    '2014-10',
                    '2014-11',
                    '2014-12']

    for yyyy_mm in first6m_list:
        yyyymm = yyyy_mm.replace('-', '')
        demographic_df.loc[:, "risk_score_{}".format(yyyymm)] = \
            demographic_df["KK_RISK_KODU_{}".format(yyyymm)].apply(
                lambda x: x.split(")")[0]).replace("\.", np.nan,
                                                   regex=True)
    filtered_df = df[df['yyyymm'].isin(first6m_list)]

    # TODO
    # spending amount diversity
    # location diversity
    # time diversity

    # extract demographic features for each merchant
    merchantid_list = []
    feature_dict_list = []
    for merchantid, group_df in filtered_df.groupby(["UYEISYERI_ID_MASK"]):
        customerid_set = set(group_df["MUSTERI_ID_MASK"].tolist())
        cur_df = demographic_df[demographic_df["MUSTERI_ID_MASK"].isin(customerid_set)]

        feature_dict = {
            # income
            "income_median": cur_df["GELIR"].median(),
            "income_mean": cur_df["GELIR"].mean(),
            "income_std": cur_df["GELIR"].std(),

            # age
            "age_median": cur_df["YAS"].median(),
            "age_mean": cur_df["YAS"].mean(),
            "age_std": cur_df["YAS"].std(),

            # home district entropy
            "home_dist_ent": calc_entropy(cur_df["HOME_DISTRICT_ID"]),
            "work_dist_ent": calc_entropy(cur_df["WORK_DISTRICT_ID"]),

            # demographic entropy
            "gender_ent": calc_entropy(cur_df["CINSIYETI"]),
            "education_ent": calc_entropy(cur_df['EGITIM_DRM_ACK']),
            "marital_ent": calc_entropy(cur_df["MEDENI_DRM_ACK"]),
            "employment_ent": calc_entropy(cur_df["IS_TURU_ACK"])}
        merchantid_list.append(merchantid)
        feature_dict_list.append(feature_dict)

    feature_df = pd.DataFrame(feature_dict_list)
    feature_df.index = merchantid_list
    feature_df.to_csv("features/demographic.csv")
