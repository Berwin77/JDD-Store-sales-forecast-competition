import numpy as np
import pandas as pd
import datetime


ads_path = './data/t_ads.csv'
ads = pd.read_csv(ads_path)
ads['create_dt'] = pd.to_datetime(ads['create_dt'])

date_list = ['2016/6/30', '2016/7/31', '2016/8/31', '2016/9/30', '2016/10/31', '2016/11/30',
              '2016/12/31', '2017/1/31', '2017/2/28', '2017/3/31', '2017/4/30']
delta = datetime.timedelta(days=30)
date_list = pd.to_datetime(date_list)

feature = pd.DataFrame()
for d in date_list:
    ad = ads[(d - delta < ads['create_dt']) & (ads['create_dt'] <= d)]

    #consume  一定注意as_index这个值 用False代表 groupby的值不做index
    consume_num = ad.groupby('shop_id', as_index=False).agg({'consume':['sum']})

    consume_num.columns = ['shop_id', 'consume_num_sum']
    #charge
    charge_num = ad.groupby('shop_id', as_index=False).agg({'charge':['sum']})
    charge_num.columns = ['shop_id', 'charge_num_sum']

    all_feature = pd.merge(consume_num, charge_num, on='shop_id')
    all_feature['dt'] = d
    feature = pd.concat([feature, all_feature])
    print(d)

#输出
# fea_mean = feature['consume_num_sum'].mean()
# fea_max = feature['consume_num_sum'].max()
# fea_min = feature['consume_num_sum'].min()
# fea_median = feature['consume_num_sum'].median()
# fea_std = feature['consume_num_sum'].std()
# feature['consume_num_sum'] = feature['consume_num_sum']/fea_max
# print(fea_mean, fea_max,fea_median,fea_std, fea_min)
#feature['consume_num_sum'] = (feature['consume_num_sum']-feature['consume_num_sum'].mean())/feature['consume_num_sum'].std(ddof=0)

#print(feature['consume_num_sum'].max, feature['consume_num_sum'].min)
feature.to_csv('./cache/ads_feature_all.csv', index=False)