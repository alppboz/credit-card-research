import pandas as pd
import numpy as np
import geopandas as gpd
from os.path import join, exists
import os
import yaml
import logging

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler('.spatialfiltlog', 'w')
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s ~~ %(message)s', datefmt='%d-%b-%y %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

def geo_filter_merchants(trans_fname, bank, geom, config, overwrite=False):
    '''
    filter merchants by their locations
    '''
    output_file = join('data', f'bank_{bank}_transactions.csv')
    if exists(output_file):
        if not overwrite:
            print(f'{output_file} already exists')
            return
        else:
            os.remove(output_file)
    
    print(f'extracting transactions for bank {bank}')
    dfs = pd.read_csv(trans_fname, chunksize=10**6 * 3, escapechar='\\', na_values='N')
    header = True

    lat_col = config['tran_cols'][f'bank_{bank}']['merchant_lat']
    lng_col = config['tran_cols'][f'bank_{bank}']['merchant_lng']

    original_trans = 0
    original_num_merchants = 0
    n_trans = 0
    num_merchants = 0
    for df in dfs:
        original_trans += df.shape[0]
        original_num_merchants += df[config['tran_cols'][f'bank_{bank}']['merchant_id']].nunique()

        df = df.dropna(subset=[lng_col, lat_col])
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lng_col], df[lat_col]), crs='EPSG:4326')
        df = gpd.sjoin(df, geom, how='inner', op='within')

        n_trans += df.shape[0]
        num_merchants += df[config['tran_cols'][f'bank_{bank}']['merchant_id']].nunique()

        df.drop(['geometry', 'index_right'], axis=1).to_csv(output_file, header=header, index=False, mode='a')
        header=False

    logger.debug('bank {} -> original # of transactions: {}'.format(bank, original_trans))
    logger.debug('bank {} -> original # of merchants: {}'.format(bank, original_num_merchants))

    logger.debug('bank {} -> filtered # of transactions: {}'.format(bank, n_trans))
    logger.debug('bank {} -> filtered # of merchants: {}'.format(bank, num_merchants))
    print(f'spatial filtering bank_{bank}: done')


if __name__ == '__main__':
    greater_ist_shp = gpd.read_file(join('data', 'greater-istanbul-area.geojson'))
    banks = ['x', 'y']
    trans_fnames = [(join('data', f'bank_{bank}_transactions_raw.csv'), bank) for bank in banks]

    with open(join('config.yaml')) as f:
        config = yaml.safe_load(f)

    for trans_fname, bank in trans_fnames:
        geo_filter_merchants(trans_fname, bank, greater_ist_shp, config)