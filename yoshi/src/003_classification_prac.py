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
    poi_df = pd.read_csv('data/poi-Istanbul-filtered.csv')
    merchant_id_list = filter(lambda x: x in poi_df['merchant_id'].tolist(),
                              label_df.index.tolist())
    y = label_df.ix[merchant_id_list].as_matrix().ravel()
    X_rev = sum_df[sum_df.columns[0:6].tolist()].ix[merchant_id_list].as_matrix()
    X_mean = sum_df[sum_df.columns[0:6].tolist()].ix[merchant_id_list].mean(
        axis=1).as_matrix().reshape(-1, 1)
    X_std = sum_df[sum_df.columns[0:6].tolist()].ix[merchant_id_list].std(
        axis=1).as_matrix().reshape(-1, 1)
    onehot_enc = OneHotEncoder()
    X_distid = onehot_enc.fit_transform(
        poi_df.set_index('merchant_id').ix[merchant_id_list]['district_id'].as_matrix().reshape(-1, 1)).todense()

    for clf_name, clf in [('lr', GridSearchCV(LogisticRegression(),
                                              param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
                                              scoring='roc_auc')),
                          ('xgboost',  GridSearchCV(XGBClassifier(),
                                                    param_grid={'n_estimators': [10, 100],
                                                                'learning_rate': [0.01, 0.05],
                                                                'max_depth': [2, 5]},
                                                    scoring='roc_auc'))]:
        test_auc_dict = {}
        for name, X in [('rev', X_rev),
                        ('rev-mean-std',
                         np.concatenate([X_rev, X_mean, X_std], axis=1)),
                        ('rev-mean-std-distid',
                         np.concatenate([X_rev, X_mean, X_std, X_distid], axis=1))]:
            skf = StratifiedKFold(n_splits=5,
                                  shuffle=True,
                                  random_state=1)
            test_auc_list = []
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                test_auc = roc_auc_score(y_test,
                                         clf.best_estimator_.predict_proba(X_test)[:, 1])
                test_auc_list.append(test_auc)
            test_auc_dict[name] = test_auc_list

        test_auc_df = pd.DataFrame(test_auc_dict)
        eval_df = pd.DataFrame([test_auc_df.mean(axis=0), test_auc_df.std(axis=0)]).T
        eval_df.columns = ['Mean', 'Std.']
        eval_df.to_csv("output/003_classification_{}.csv".format(clf_name))
