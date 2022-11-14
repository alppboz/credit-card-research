python run_experiment.py -O results/with-mcc-dist/ -L labels/labels_x.csv -F features/demographics_x.csv
python run_experiment.py -O results/with-mcc-dist/ -L labels/labels_x.csv -F features/revenue_x.csv
python run_experiment.py -O results/with-mcc-dist/ -L labels/labels_x.csv -F features/filtered_network_features_weight_x.csv
python run_experiment.py -O results/with-mcc-dist/ -L labels/labels_x.csv -F features/demographics_x.csv features/revenue_x.csv
python run_experiment.py -O results/with-mcc-dist/ -L labels/labels_x.csv -F features/demographics_x.csv features/revenue_x.csv features/filtered_network_features_weight_x.csv
python run_experiment.py -O results/with-mcc-dist/ -L labels/labels_y.csv -F features/demographics_y.csv
python run_experiment.py -O results/with-mcc-dist/ -L labels/labels_y.csv -F features/revenue_y.csv
python run_experiment.py -O results/with-mcc-dist/ -L labels/labels_y.csv -F features/filtered_network_features_weight_y.csv
python run_experiment.py -O results/with-mcc-dist/ -L labels/labels_y.csv -F features/demographics_y.csv features/revenue_y.csv
python run_experiment.py -O results/with-mcc-dist/ -L labels/labels_y.csv -F features/demographics_y.csv features/revenue_y.csv features/filtered_network_features_weight_y.csv
