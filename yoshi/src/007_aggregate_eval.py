import argparse
import os
import re

import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='eval command')
    parser.add_argument('-D', '--dirpath',
                        type=str,
                        default="eval/",
                        help='Base directory path')
    parser.add_argument('-L', '--labelname',
                        type=str,
                        required=True,
                        help='Label name for aggregation')
    args = parser.parse_args()
    labelname = args.labelname
    dirpath = args.dirpath
    mean_dict = {}
    for filename in os.listdir(dirpath):
        if not re.search(labelname, filename):
            continue
        if re.search("_fimp_", filename):
            continue
        filepath = os.path.join(dirpath, filename)
        df = pd.read_csv(filepath, index_col=0)
        featurename = filename.split("__")[1].split('.')[0]
        mean_dict[featurename] = df.loc["mean"]
    agg_df = pd.DataFrame(mean_dict)
    result_df = agg_df.T.sort_values('xgboost')
    result_df.to_csv("table/{}.csv".format(labelname))