import pandas as pd
import numpy as np
from os.path import join, exists
import os
import argparse
import yaml
from datetime import datetime
import igraph as ig
from tqdm import tqdm
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler('.netlogfile')
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s ~~ %(message)s', datefmt='%d-%b-%y %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)


def const_trans_net(config, bank, overwrite=False):
    '''
    construct merchant networks from transaction records
    '''
    output_file = join('data', 'networks', f'filtered_bank_{bank}.pickle')
    if exists(output_file):
        if not overwrite:
            print(f'{output_file} already exists')
            return
        else:
            os.remove(output_file)

    customer_min_trans = config['network_conf']['customer_min_trans']
    min_customer_num = config['network_conf']['min_customer_num']
    trans_cols = config['tran_cols'][f'bank_{bank}']

    #trans_df = pd.read_csv(join('data', f'bank_{bank}_transactions.csv'), dtype={trans_cols['merchant_id']: int})
    trans_df = pd.read_csv(join('data', 'filtered_data', f'filtered_bank_{bank}_trans.csv'), dtype={trans_cols['merchant_id']: int})

    date_format = config['break_date']['date_format']
    break_date = datetime.strptime(config['break_date'][f'bank_{bank}'], date_format)

    # filter by break date
    # bank_date_format = config['break_date'][f'bank_{bank}_date_format']
    # trans_df[trans_cols['tran_date']] = pd.to_datetime(trans_df[trans_cols['tran_date']], format=bank_date_format)
    trans_df[trans_cols['tran_date']] = pd.to_datetime(trans_df[trans_cols['tran_date']], format=date_format)
    trans_df = trans_df[trans_df[trans_cols['tran_date']] <= break_date].set_index(trans_cols['merchant_id'])

    # merchant districts and mcc
    mcc_districts = pd.read_csv(join('data', 'filtered_data', f'bank_{bank}_merchant_districts.csv')).set_index(trans_cols['merchant_id'])
    mcc_districts_lookup = set(mcc_districts.index)

    mid_cs_list = []
    for merchant in tqdm(np.unique(trans_df.index), desc='finding customers'):
        customers = trans_df.loc[merchant, trans_cols['customer_id']]
        if not isinstance(customers, pd.Series):
            customers = pd.Series(customers)
        cust_freq = customers.value_counts()
        cur_s = cust_freq[cust_freq >= customer_min_trans]
        if len(cur_s) > 0:
            customer_set = set(cur_s.index.tolist())
        else:
            customer_set = set([])
        mid_cs_list.append([merchant, customer_set])

    edge_list = []
    node_list = []
    mcc_list = []
    district_id_list = []
    weights = []
    for i in tqdm(range(len(mid_cs_list)), desc='creating edges'):
        mid_i, cs_i = mid_cs_list[i]
        mcc = mcc_districts.loc[mid_i, trans_cols['mcc']] if mid_i in mcc_districts_lookup else 'unk'
        mcc_list.append(mcc)
        district_id = mcc_districts.loc[mid_i, 'district_id'] if mid_i in mcc_districts_lookup else 'unk'
        district_id_list.append(district_id)
        node_list.append(mid_i)
        for j in range(i + 1, len(mid_cs_list)):
            mid_j, cs_j = mid_cs_list[j]
            count = len(cs_i & cs_j)  # find the number of shared customers
            if count > min_customer_num:
                edge_list.append((mid_i, mid_j))
                weights.append(count)

    g = ig.Graph(directed=False)
    # store merchant ids as str
    g.add_vertices([str(node) for node in node_list])
    g.add_edges([(str(i), str(j)) for i,j in edge_list])
    g.vs['mcc'] = mcc_list
    g.vs['district_id'] = district_id_list
    g.es['weight'] = weights

    g.write_pickle(output_file)

    print('network created')

    num_merchants = len(node_list)
    unk_merchants = len([n for n in mcc_list if n == 'unk'])
    logger.debug('bank {}, unk merchants count: {}, pct: {}'.format(bank, unk_merchants, unk_merchants/num_merchants*100))
    logger.debug('bank {}, graph # of nodes: {}, # of edges {}, density: {}'.format(bank, g.vcount(), g.ecount(), g.density()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create revenue/demographic/network features and well-being labels')

    parser.add_argument('-B', '--bank',
                        type=str,
                        required=True,
                        help='bank name ("x", "y" or custom)')

    args = parser.parse_args()
    bank = args.bank.lower()

    with open(join('config.yaml')) as f:
        config = yaml.safe_load(f)

    const_trans_net(config, bank)