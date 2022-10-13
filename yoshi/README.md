#

## How to run the script

`src/004_run_experiment.py` will run experiments given settings. You can use the script with your favorite parameters to explore the performance of additional features.

You need to prepare label and feature data in the CSV format. Each CSV file must contain merchant ID in the first column. If merchant ID sets in these files are inconsistent, the script will use the merchantID set of input label data as master. Note that the experimental results are not directly comparable if you use different label files whose merchant IDs are inconsistent.

```
$ pwd
creditcard-research/yoshi

$ ls labels
last6m-median.csv

$ ls features
distid.csv      first6m-rev.csv
```

To run the command, use `-L` to point a label data file and `-F` to point feature data file(s). You can use multiple feature name files. 

```
$ python src/004_run_experiment.py -L labels/last6m-median.csv -F features/distid.csv features/first6m-rev.csv
```

The result will be stored in `eval/` directory. You can change the output directory with `-O` option. You may see that the file name is the concatenation of label name and feature name(s).

```
$ ls eval
last6m-median__distid_first6m-rev.csv
```

This evaluation file contains ROC-AUC measure for each fold in addition to mean and std.

```
$ cat eval/last6m-median__distid_first6m-rev.csv
,lr,xgboost
0,0.628285780378,0.647917242624
1,0.652284580499,0.634155328798
2,0.598673469388,0.624427437642
3,0.655317460317,0.675433673469
4,0.585907029478,0.622760770975
mean,0.624093664012,0.640938890702
std,0.0311914748748,0.0217236982625
```

