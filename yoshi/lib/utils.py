# -*- coding: utf-8 -*-

import pandas as pd
import pyproj

#name_filename_dict = {'demographic': 'SU_MUSTERI_KITLE_ORNEKLEM_100K.txt',
#name_filename_dict = {'demographic': 'SU_MUSTERI_KITLE_ORNEKLEM_60K.txt',
name_filename_dict = {'demographic': 'SU_MUSTERI_KITLE_ORNEKLEM_60K.txt.districtid',
                      'atm': 'SU_ORNEKLEM_ATM_ISLEM_XY.txt',
                      'product': 'SU_ORNEKLEM_CAPRAZ_SATIS.txt',
                      'callcenter' : 'SU_ORNEKLEM_CM_ISL_VE_BAGLANMA.txt',
                      'eft_transfer': 'SU_ORNEKLEM_GIDEN_EFT_BIL.txt',
                      'internal_transfer': 'SU_ORNEKLEM_HAVALE_BIL.txt',
                      'campaign_rt': 'SU_ORNEKLEM_KAMPANYA_VERILERI_RT.txt',
                      'campaign_batch': 'SU_ORNEKLEM_KAMPANYA_VERI_BATCH.txt',
                      'credit': 'SU_ORNEKLEM_KKB_VERILERI.txt',
                      'ekstre_bill': 'SU_ORNEKLEM_KK_EKSTRE_BIL.txt',
                      'odm_bill': 'SU_ORNEKLEM_KK_EKSTRE_ODM_BIL.txt',
#                      'transaction': 'SU_ORNEKLEM_KK_HAR_BILGI.txt',
                      'transaction': 'SU_ORNEKLEM_KK_HAR_BILGI.txt.districtid',
                      'mobile': 'SU_ORNEKLEM_MOBIL_INTERNET_BIL.txt',
                      'autopayment': 'SU_ORNEKLEM_OFO_BILGI.txt',
                      'responsescore': 'SU_ORNEKLEM_RESPONSE_MODEL_TREND.txt',
                      'riskscore': 'SU_ORNEKLEM_RISK_SKOR.txt',
                      'branchvisit': 'SU_ORNEKLEM_SUBE_ISL_ZIY.txt',
                      'district': 'DISTRICT_ID.txt'}


mcccode_name_dict = {5411: 'Grocery Stores, Supermarkets (Retail Stores)',
                     5541: 'Service Stations (with or without Ancillary Services) (Automobiles and Vehicles)',
                     5691: "Men's and Women's Clothing Stores (Clothing Stores)",
                     5812: 'Eating Places, Restaurants (Miscellaneous Stores)',
                     5960: 'Direct Marketing-Insurance Services (Mail Order/Telephone OrderProviders)',
                     5499: 'Miscellaneous Food Stores-Convenience Stores, Markets,Specialty Stores, and Vending Machines (Retail Stores)'}

districtid_name_dict = {1: "Arnavutköy",
                        2: "Güngören",
                        4: "Ataşehir",
                        9: "Beykoz",
                        12: "Zeytinburnu",
                        15: "Çatalca",
                        17: "Beyoğlu",
                        31: "Silivri",
                        33: "Beşiktaş",
                        34: "Bakırköy",
                        38: "Sancaktepe",
                        42: "Esenyurt",
                        50: "Kadıköy",
                        53: "Başakşehir",
                        59: "Sarıyer",
                        60: "Büyükçekmece",
                        61: "Üsküdar",
                        63: "Ümraniye",
                        65: "Pendik",
                        70: "Bayrampaşa",
                        77: "Bağcılar",
                        79: "Esenler",
                        81: "Bahçelievler",
                        82: "Eyüp",
                        84: "Fatih",
                        86: "Sultangazi",
                        87: "Tuzla",
                        88: "Avcılar",
                        96: "Kartal",
                        98: "Sultanbeyli",
                        100: "Gaziosmanpaşa",
                        111: "Adalar",
                        115: "Şişli",
                        120: "Beylikdüzü",
                        121: "Küçükçekmece",
                        123: "Kağıthane",
                        131: "Şile",
                        141: "Çekmeköy",
                        160: "Maltepe"}

def calc_distance (lon1, lat1, lon2, lat2):
    """
    ## Not Tested Yet! ##

    :param float lon1:
    :param float lat1:
    :param float lon2:
    :param float lat2:

    :return: distance (meter)
    :rtype: float

    Returns distance between given geocoordinates
    Hubeny? Vincentiy?
    meter
    """
    q = pyproj.Geod(ellps='WGS84')
    fa, ba, d = q.inv( lon1, lat1, lon2, lat2 )
    return d



def get_districtname(districtid):
    if districtid in districtid_name_dict:
        return districtid_name_dict[districtid].decode('utf-8')
    else:
        return None


def get_categoryname(mcccode):
    if mcccode in mcccode_name_dict:
        return mcccode_name_dict[mcccode]
    else:
        return None


def get_filename(name):
    if name in name_filename_dict:
        return name_filename_dict[name]
    return None

def load_data(name=None,
              basedir='data/new_data',
              encoding='latin5'):

    if name is None:
        data_dict = {}
        for name, filename in name_filename_dict.items():
            filepath = "%s/%s" % (basedir, filename)
            data_dict[name] = pd.read_csv(filepath, encoding=encoding)
        return data_dict
    else:
        return pd.read_csv('%s/%s' % (basedir, name_filename_dict[name]),
                           encoding=encoding)


def load_filtered_data(min_transaction=10,
                       merchant_min_uniqcustomer=2,
                       merchant_min_transaction=10):
    """
    #transaction	categoryid	categoryname

    1089614	5411    Grocery Stores, Supermarkets    Retail Stores
    482178	5541    Service Stations (with or without Ancillary Services)   Automobiles and Vehicles
    437760	5691    Men<92>s and Women<92>s Clothing Stores Clothing Stores
    185595	5812    Eating Places, Restaurants      Miscellaneous Stores
    # 138713	5960    Direct Marketing-Insurance Services     Mail Order/Telephone OrderProviders
    138575	5499    Miscellaneous Food Stores-Convenience Stores, Markets,Specialty Stores, and Vending Machines    Retail Stores
    """
    transaction_df = load_data('transaction')
    category_list = [5411, 5541, 5691, 5812, 5499]
    customerid_list_list = []
    for category in category_list:
        count_s = transaction_df[transaction_df['ISYERI_TURU'] == category]['MUSTERI_ID_MASK'].value_counts()
        customerid_list = count_s[count_s >= min_transaction].index.tolist()
        print(customerid_list)
        customerid_list_list.append(customerid_list)
        """
        filtered_transaction_df = transaction_df[transaction_df['MUSTERI_ID_MASK'].isin(customerid_list)]
        agg_df = filtered_transaction_df.groupby(['UYEISYERI_ID_MASK']).agg({'MUSTERI_ID_MASK': ['count', 'nunique']})
        filtered_agg_df = agg_df[(agg_df['MUSTERI_ID_MASK']['nunique'] >= merchant_min_uniqcustomer) & \
                                 (agg_df['MUSTERI_ID_MASK']['count'] >= merchant_min_transaction)]
        filtered_transaction_df = filtered_transaction_df[filtered_transaction_df['UYEISYERI_ID_MASK'].isin(filtered_agg_df.index)]
        district_agg_df = filtered_transaction_df.groupby(['DISTRICT_ID']).\
                          agg({'MUSTERI_ID_MASK': ['count', 'nunique']})
        district_agg_df.to_csv('output/002-001/002-001_categoryid-%d_district_numcustomer.csv' % (category))
        """
    return customerid_list_list
