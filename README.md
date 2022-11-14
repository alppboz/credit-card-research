# Merchant Financial Well-Being with Networks

### Preparing features and label sets

- `spatial_filter.py`: Filters raw transaction data with respect to Greater Istanbul Area (~ 10km) and creates `data/bank_\[type\]_transaction.csv`.
- `filter_records.py`: Filter merchants based on the parameters in `trans_filter` attribute in `config.yaml`. It creates `data/filtered_data/filtered_bank_\[type\]_transaction.csv`.
- `generate_features_labels.py`: Constructs demographic, revenue and network features for the merchants in filtered transactions. In addition, creates labels based on merchant revenues.
- `run_experiment.py`: Creates and runs models on the given feature set(s) and label values.

#### Merchants at the border

In our implementation, we consider the shared customers between merchants inside the selected administrative boundaries and the ones that are close them but happens to be located in another region. We increase Istanbul shapefile with a buffer of 10km and consider the merchants (that adhere the merchant selection procedure) in the resulting networks. During feature generation, we solely focus on the merchants falling within the selected administrative region.