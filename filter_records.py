import pandas as pd
import numpy as np
import yaml
from os.path import join, exists
import argparse
import geopandas as gpd
import logging
import os

logfname = '.filterlogfile'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfname, mode='w')
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s ~~ %(message)s', datefmt='%d-%b-%y %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)


def filter_trans_records(config, bank, fname_prefix):
    '''
    filter credit transactions for the given bank
    '''
    fname = config['trans_file_names'][f'bank_{bank}']
    if fname_prefix:
        fname = '{}_{}'.format(fname_prefix, fname)

    output_fname = join('data', 'filtered_data', fname)

    print(f'filtering transactions for bank: {bank}')

    #df = pd.read_csv(join('data', f'bank_{bank}_transactions.csv'))
    df = pd.read_csv(join('data', f'bank_{bank}_transactions_customer_filters.csv'))

    cols = config['tran_cols'][f'bank_{bank}']
    filters = config['trans_filter'][f'bank_{bank}']

    logger.debug('bank {}, original merchant count: {}'.format(bank, df[cols["merchant_id"]].nunique()))
    logger.debug('bank {}, # of transactions: {}'.format(bank, df.shape[0]))

    # remove invalid mcc entries
    df = df[df[cols['merchant_id']] != 999999]

    # remove online transactions
    if filters['remove_online']:
        df = df[df[cols['online_flag']] == 0]

    # filter mcc
    df = df[df[cols['mcc']].isin(filters['mcc_list'])]

    # filter out merchants by transaction counts
    tcount = df[cols['merchant_id']].value_counts()
    filtered_merchantid_list = tcount[tcount >= filters['min_trans_count']].index.tolist()        
    df = df[df[cols['merchant_id']].isin(filtered_merchantid_list)]

    # filter out merchant categories that do not have sufficient amount of merchants.
    mcc_count_s = df[cols['mcc']].value_counts()
    filtered_mcc_list = mcc_count_s[mcc_count_s >= filters['min_merchants_mcc']].index.tolist()
    df = df[df[cols['mcc']].isin(filtered_mcc_list)]

    date_format = config['break_date']['date_format']
    bank_date_format = config['break_date'][f'bank_{bank}_date_format']
    df[cols['tran_date']] = pd.to_datetime(df[cols['tran_date']], format=bank_date_format)

    logger.debug('bank {}, time range: {} - {}'.format(bank, df[cols["tran_date"]].min(), df[cols["tran_date"]].max()))

    df['yyyymm'] = df[cols['tran_date']].dt.strftime('%Y-%m')
    df[cols['tran_date']] = df[cols['tran_date']].dt.strftime(date_format)

    monthly_agg_df = df.groupby([cols['merchant_id'], 'yyyymm']).agg({cols['customer_id']: ['count', 'nunique'],
                                                                        cols['tran_amount']: ['sum', 'mean']}).unstack()
    merchants_every_month = monthly_agg_df[monthly_agg_df[cols['customer_id']]['count'].isnull().sum(axis=1) <= (12 - filters['min_month_trans'])].index.tolist()
    df = df[df[cols['merchant_id']].isin(merchants_every_month)]

    logger.debug('bank {}, [BEFORE SPATIAL FILTER] # of merchants: {}'.format(bank, df[cols["merchant_id"]].nunique()))
    logger.debug('bank {}, [BEFORE SPATIAL FILTER] # of transactions: {}'.format(bank, df.shape[0]))

    print('extracting merchant districts')

    # extract merchant districts
    shp = gpd.read_file(join('data', config['shpfiles'][f'bank_{bank}']))
    latlngs = df[[cols['merchant_id'], cols['mcc'], cols['merchant_lat'], cols['merchant_lng']]].drop_duplicates(subset=[cols['merchant_id']])
    latlngs = gpd.GeoDataFrame(latlngs, geometry=gpd.points_from_xy(latlngs[cols['merchant_lng']], latlngs[cols['merchant_lat']]), crs='EPSG:4326')

    # spatial join
    latlngs = gpd.sjoin(latlngs, shp, how='inner', op='within')
    latlngs[[cols['merchant_id'], cols['mcc'], 'district_id']].to_csv(join('data', 'filtered_data', f'bank_{bank}_merchant_districts.csv'), index=False)

    df = df.loc[df[cols['merchant_id']].isin(latlngs[cols['merchant_id']])]

    logger.debug('bank {}, [AFTER SPATIAL FILTER] # of merchants: {}'.format(bank, df[cols['merchant_id']].nunique()))
    logger.debug('bank {}, [AFTER SPATIAL FILTER] # of transactions: {}'.format(bank, df.shape[0]))        

    logger.debug('bank {}, nan districts: {}'.format(bank, latlngs.isna().sum()))

    df.to_csv(output_fname, index=False)

    print('merchant district extraction done')
    print('obtaining aggregate transaction summaries')

    # record aggregated summaries
    agg_df = df.groupby([cols['merchant_id'], cols['mcc'], cols['tran_date']]).agg({cols['customer_id']: ['count', 'nunique'], cols['tran_amount']: ['sum', 'mean']})

    agg_fname = config['trans_file_names']['agg'][f'bank_{bank}']
    if fname_prefix:
        agg_fname = '{}_{}'.format(fname_prefix, agg_fname)
    agg_df.to_csv(join('data', 'filtered_data', agg_fname), index=True)


def assign_customer_district_ids(config, bank):
    '''
    assign district ids to customer home and work locations based on their lat/lngs
    '''
    customer_df_fname = join('data', f'bank_{bank}_customers.csv')
    output_fname = join('data', 'filtered_data', f'bank_{bank}_customers_districts.csv')

    print('assigning customer home and work districts')

    customer_df = pd.read_csv(customer_df_fname)
    shp = gpd.read_file(join('data', config['shpfiles'][f'bank_{bank}']))

    cols = config['customer_cols'][f'bank_{bank}']

    def assing_district(dt, customers):
        '''
        find 'home' and 'work' districts (dt) for the given customers
        '''
        assert dt == 'home' or dt == 'work', 'invalid district type...'

        cust_geo = gpd.GeoDataFrame(customers, geometry=gpd.points_from_xy(customers[cols[f'{dt}_lng']], customers[cols[f'{dt}_lat']]), crs='EPSG:4326')
        result = gpd.sjoin(cust_geo, shp, how='left', op='within')

        # change the name of district column and remove join columns as well
        return result.rename({'district_id': f'{dt}_district_id'}, axis=1).drop([c for c in result.columns if '_left' in c or '_right' in c], axis=1)

    home_districts = assing_district('home', customer_df)
    # combine them with work districts
    combined_districts = assing_district('work', home_districts)
    
    result = combined_districts.drop('geometry', axis=1)
    logger.debug('bank {}, customer districts nan value percentages: {}'.format(bank, result[['home_district_id', 'work_district_id']].isna().sum() / result.shape[0]))
    result.to_csv(output_fname, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filtering credit card transactions for the given bank')

    parser.add_argument(
        '-B', 
        '--bank',
        type=str,
        required=True,
        help='bank name ("x", "y" or custom)'
    )

    parser.add_argument(
        '-P',
        '--prefix',
        type=str,
        required=False,
        help='output file name prefix'
    )

    parser.add_argument(
        '-M',
        '--mcc',
        type=int,
        required=False,
        nargs='*',
        help='mcc list to be considered'
    )

    args = parser.parse_args()
    bank = args.bank.lower()
    fname_prefix = args.prefix
    mcc_list = args.mcc

    with open(join('config.yaml')) as f:
        config = yaml.safe_load(f)

    if mcc_list:
        config['trans_filter'][f'bank_{bank}']['mcc_list'] = mcc_list

    filter_trans_records(config, bank, fname_prefix)
    assign_customer_district_ids(config, bank)