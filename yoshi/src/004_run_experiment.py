"""
TODO:
  - Feature importance calculation
"""

import argparse
from logging import getLogger
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
try:
    from xgboost import XGBClassifier
    GBClassifier = XGBClassifier
except ImportError:
    print(("WARNING: xgboost not installed. ",
           "Use sklearn.ensemble.GradientBoostingClassifier instead."))
    from sklearn.ensemble import GradientBoostingClassifier
    GBClassifier = GradientBoostingClassifier


import warnings
warnings.filterwarnings("ignore")


logger = getLogger(__name__)


def load_data(label_filepath,
              feature_filepath_list):

    assert os.path.exists(label_filepath)
    for feature_filepath in feature_filepath_list:
        assert os.path.exists(feature_filepath)

    label_df = pd.read_csv(label_filepath,
                           index_col=0)
    print(label_df.shape, label_filepath)
    feature_df_list = []
    for feature_filepath in feature_filepath_list:
        f_df = pd.read_csv(feature_filepath,
                           index_col=0)
        filename = os.path.basename(feature_filepath).split('.')[0]
        f_df.columns = map(lambda x: "{}__{}".format(filename, x),
                           f_df.columns.tolist())
        feature_df_list.append(f_df)
    feature_df = pd.concat(feature_df_list, axis=1)
    label_merchantid_set = set(label_df.index.tolist())
    feature_merchantid_set = set(feature_df.index.tolist())

    instance_ind = list(label_merchantid_set & feature_merchantid_set)
    assert len(instance_ind) > 0

    logger.debug("#label_merchantid_set={}".format(
        len(label_merchantid_set)))
    logger.debug('#feature_merchantid_set={}'.format(
        len(feature_merchantid_set)))
    logger.debug('#intersection='.format(
        len(label_merchantid_set & feature_merchantid_set)))

    filtered_feature_df = feature_df.loc[instance_ind]
    X = filtered_feature_df.fillna(0).values
    y = label_df.loc[instance_ind].values.ravel()
    assert len(np.unique(y)) == 2
    assert X.shape[0] == len(y)
    return X, y, filtered_feature_df


def run_cross_validation(X, y,
                         feature_df=None):
    test_auc_dict = {}
    fimp_dict = {}
    for clf_name, clf in [('lr', GridSearchCV(
                                LogisticRegression(),
                                param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
                                scoring='roc_auc'
                            )),
                            ('xgboost', GridSearchCV(
                                XGBClassifier(),
                                param_grid={'n_estimators': [10, 100],
                                            'learning_rate': [0.01, 0.05],
                                            'max_depth': [2, 5]},
                                scoring='roc_auc'
                            )),
                            ('rf', GridSearchCV(
                                RandomForestClassifier(),
                                    param_grid={'n_estimators': [10, 100],
                                                'max_depth': [3, 5, 10, 20]
                                },
                                scoring='roc_auc'
                            )),
                            ('svm', GridSearchCV(
                                SVC(),
                                param_grid={
                                    'kernel': ['linear', 'rbf', 'sigmoid'],
                                    'probability': [True]
                                },
                                scoring='roc_auc'
                            ))
                              ]:
        skf = StratifiedKFold(n_splits=5,
                              shuffle=True,
                              random_state=1)
        test_auc_list = []
        fimp_list = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            clf.fit(X_train, y_train)
            test_auc = roc_auc_score(
                y_test,
                clf.best_estimator_.predict_proba(X_test)[:, 1])
            test_auc_list.append(test_auc)
            # feature importance
            if clf_name == 'lr':
                fimp = clf.best_estimator_.coef_[0]
            elif clf_name == 'xgboost':
                fimp = clf.best_estimator_.feature_importances_
            fimp_list.append(fimp)

        test_auc_dict[clf_name] = test_auc_list
        fimp_df = pd.DataFrame(fimp_list, columns=feature_df.columns)
        fimp_dict[clf_name] = fimp_df

    test_auc_df = pd.DataFrame(test_auc_dict)
    mean_df = pd.DataFrame(test_auc_df.mean(axis=0)).T
    mean_df.index = ["mean"]
    std_df = pd.DataFrame(test_auc_df.std(axis=0)).T
    std_df.index = ["std"]

    eval_df = pd.concat([test_auc_df,
                         mean_df,
                         std_df],
                        axis=0)

    return eval_df, fimp_dict


def create_filename(label_filepath,
                    feature_filepath_list):
    prefix = os.path.basename(label_filepath).split(".")[0]
    suffix = "_".join(list(
        map(lambda x: os.path.basename(x).split(".")[0],
            feature_filepath_list)))
    return "{}__{}.csv".format(prefix, suffix)


def run_experiment(label_filepath,
                   feature_filepath_list,
                   output_dirpath):
    print(output_dirpath)
    assert os.path.exists(output_dirpath)
    output_filepath = os.path.join(output_dirpath,
                                   create_filename(label_filepath,
                                                   feature_filepath_list))
    X, y, feature_df = load_data(label_filepath,
                                 feature_filepath_list)
    eval_df, fimp_dict = run_cross_validation(X, y, feature_df)
    eval_df.to_csv(output_filepath)
    for clf_name, fimp_df in fimp_dict.items():
        cur_filepath = output_filepath.replace(".csv",
                                               "_fimp_{}.csv".format(clf_name))
        fimp_mean_s = fimp_df.mean(axis=0)
        fimp_std_s = fimp_df.std(axis=0)
        fimp_info_df = pd.DataFrame({'fimp_mean': fimp_mean_s,
                                     'fimp_std': fimp_std_s})
        fimp_info_df.sort_values('fimp_mean',
                                 ascending=False).to_csv(cur_filepath)

    return eval_df, fimp_info_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('-F', '--features',
                        nargs='*',
                        type=str,
                        required=True,
                        help='feature file names')
    parser.add_argument('-L', '--label',
                        type=str,
                        required=True,
                        help='Label file name')
    parser.add_argument('-O', '--outputdir',
                        type=str,
                        default="eval",
                        help='Output directory path')
    args = parser.parse_args()
    feature_filepath_list = sorted(args.features)
    label_filepath = args.label
    output_dirpath = args.outputdir

    eval_df, fimp_info_df = run_experiment(label_filepath,
                                           feature_filepath_list,
                                           output_dirpath)
