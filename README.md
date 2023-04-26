# Merchant Financial Well-Being with Networks

### Preparing features and label sets

- `spatial_filter.py`: Filters raw transaction data with respect to Greater Istanbul Area (~ 10km) and creates `data/bank_\[type\]_transaction.csv`.
- `filter_records.py`: Filter merchants based on the parameters in `trans_filter` attribute in `config.yaml`. It creates `data/filtered_data/filtered_bank_\[type\]_transaction.csv`.
- `generate_features_labels.py`: Constructs demographic, revenue and network features for the merchants in filtered transactions. In addition, creates labels based on merchant revenues.
- `run_experiment.py`: Creates and runs models on the given feature set(s) and label values.
