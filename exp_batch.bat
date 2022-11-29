python spatial_filter.py
python filter_records.py --bank x
python construct_network.py --bank x
python generate_features_labels.py --bank x --weight weight
python run_experiment.py -O results/ -E only_5411 -L labels/labels_x.csv -F features/demographics_x.csv
python run_experiment.py -O results/ -E only_5411 -L labels/labels_x.csv -F features/revenue_x.csv
python run_experiment.py -O results/ -E only_5411 -L labels/labels_x.csv -F features/filtered_network_features_weight_x.csv
python run_experiment.py -O results/ -E only_5411 -L labels/labels_x.csv -F features/demographics_x.csv features/revenue_x.csv
python run_experiment.py -O results/ -E only_5411 -L labels/labels_x.csv -F features/demographics_x.csv features/revenue_x.csv features/filtered_network_features_weight_x.csv
