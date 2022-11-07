python src\004_run_experiment.py -O output\last6m-median-None-Full\ -L labels\last6m-median-None-full.csv -F features\demographic.csv
python src\004_run_experiment.py -O output\last6m-median-None-Full\ -L labels\last6m-median-None-full.csv -F features\first6m-rev-mod.csv
python src\004_run_experiment.py -O output\last6m-median-None-Full\ -L labels\last6m-median-None-full.csv -F features\unweighted_network_features.csv
python src\004_run_experiment.py -O output\last6m-median-None-Full\ -L labels\last6m-median-None-full.csv -F features\weighted_network_features.csv
python src\004_run_experiment.py -O output\last6m-median-None-Full\ -L labels\last6m-median-None-full.csv -F features\demographic.csv features\first6m-rev-mod.csv
python src\004_run_experiment.py -O output\last6m-median-None-Full\ -L labels\last6m-median-None-full.csv -F features\unweighted_network_features.csv features\first6m-rev-mod.csv features\demographic.csv
python src\004_run_experiment.py -O output\last6m-median-None-Full\ -L labels\last6m-median-None-full.csv -F features\weighted_network_features.csv features\first6m-rev-mod.csv features\demographic.csv
python src\004_run_experiment.py -O output\last6m-mcc-median-None-full\ -L labels\last6m-mcc-median-None-full.csv -F features\demographic.csv
python src\004_run_experiment.py -O output\last6m-mcc-median-None-full\ -L labels\last6m-mcc-median-None-full.csv -F features\first6m-rev-mod.csv
python src\004_run_experiment.py -O output\last6m-mcc-median-None-full\ -L labels\last6m-mcc-median-None-full.csv -F features\unweighted_network_features.csv
python src\004_run_experiment.py -O output\last6m-mcc-median-None-full\ -L labels\last6m-mcc-median-None-full.csv -F features\weighted_network_features.csv
python src\004_run_experiment.py -O output\last6m-mcc-median-None-full\ -L labels\last6m-mcc-median-None-full.csv -F features\demographic.csv features\first6m-rev-mod.csv
python src\004_run_experiment.py -O output\last6m-mcc-median-None-full\ -L labels\last6m-mcc-median-None-full.csv -F features\unweighted_network_features.csv features\first6m-rev-mod.csv features\demographic.csv
python src\004_run_experiment.py -O output\last6m-mcc-median-None-full\ -L labels\last6m-mcc-median-None-full.csv -F features\weighted_network_features.csv features\first6m-rev-mod.csv features\demographic.csv