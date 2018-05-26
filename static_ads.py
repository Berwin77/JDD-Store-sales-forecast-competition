import numpy as np
import pandas as pd
import datetime


ads_path = './data/t_ads.csv'
ads = pd.read_csv(ads_path)
ads['create_dt'] = pd.to_datetime(ads['create_dt'])


ads['dt'] = pd.to_datetime(ads['create_dt'])

date_list = ['2016/6/30', '2016/7/31', '2016/8/31', '2016/9/30', '2016/10/31', '2016/11/30',
              '2016/12/31', '2017/1/31', '2017/2/28', '2017/3/31', '2017/4/30']
delta = datetime.timedelta(days=30)
date_list = pd.to_datetime(date_list)

feature = pd.DataFrame()
for d in date_list:
    ad = ads[(date_list[0]-delta < ads['create_dt']) & (ads['create_dt'] <= d)]

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


#重命名列名
for i in feature.columns[0:]:
    feature.rename(columns={i:i+'_history'}, inplace=True)
feature.rename(columns={'dt_history':'dt', 'shop_id_history':'shop_id'}, inplace=True)
#输出
feature.to_csv('./cache/static_ads_feature.csv', index=False)
