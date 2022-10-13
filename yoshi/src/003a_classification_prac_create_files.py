import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
    GBClassifier = XGBClassifier
except ImportError:
    print(("WARNING: xgboost not installed. ",
           "Use sklearn.ensemble.GradientBoostingClassifier instead."))
    from sklearn.ensemble import GradientBoostingClassifier
    GBClassifier = GradientBoostingClassifier



if __name__ == '__main__':
    df = pd.read_csv("output/000_agg_df.csv", index_col=[0, 1], header=[0, 1])
    sum_df = df['ISLEM_TUTARI']['sum'].unstack(1)
    sum_df = sum_df.dropna(subset=sum_df.columns)

    first_sum_df = sum_df[sum_df.columns[0:6].tolist()].sum(axis=1)
    second_sum_df = sum_df[sum_df.columns[7:].tolist()].sum(axis=1)

    increase_s = (second_sum_df - first_sum_df) / first_sum_df
    pos_flag = (increase_s >= increase_s.median())
    label_df = pd.DataFrame(pos_flag.astype('int'),
                            columns=['label'])
    label_df.to_csv("labels/last6m-median.csv")

    poi_df = pd.read_csv('data/poi-Istanbul-filtered.csv')
    merchant_id_list = filter(lambda x: x in poi_df['merchant_id'].tolist(),
                              label_df.index.tolist())
    y = label_df.ix[merchant_id_list].as_matrix().ravel()
    X_rev_df = sum_df[sum_df.columns[0:6].tolist()].ix[merchant_id_list]
    X_mean_df = sum_df[sum_df.columns[0:6].tolist()].ix[merchant_id_list].mean(
        axis=1)
    X_mean_df = pd.DataFrame(X_mean_df, columns=["6m-rev-mean"])
    X_std_df = sum_df[sum_df.columns[0:6].tolist()].ix[merchant_id_list].std(
        axis=1)
    X_std_df = pd.DataFrame(X_std_df, columns=["6m-rev-std"])
    pd.concat([X_rev_df, X_mean_df, X_std_df], axis=1).to_csv("features/first6m-rev.csv")
    onehot_enc = OneHotEncoder()
    X_distid = onehot_enc.fit_transform(
        poi_df.set_index('merchant_id').ix[merchant_id_list]['district_id'].as_matrix().reshape(-1, 1)).todense()
    X_distid_df = pd.DataFrame(X_distid,
                               columns=onehot_enc.active_features_)
    X_distid_df.index = X_rev_df.index
    X_distid_df.to_csv("features/distid.csv")

