import numpy as np
import pandas as pd
import datetime


sales_path = './data/t_sales_sum.csv'
sales = pd.read_csv(sales_path)
sales['dt'] = pd.to_datetime(sales['dt'])

date_list = ['2016/6/30', '2016/7/31', '2016/8/31', '2016/9/30', '2016/10/31', '2016/11/30',
              '2016/12/31', '2017/1/31', '2017/2/28', '2017/3/31', '2017/4/30']
delta = datetime.timedelta(days=30)
date_list = pd.to_datetime(date_list)

feature = pd.DataFrame()
all_feature = pd.DataFrame()
i = 0
for d in date_list[4:]:
    sale = sales[sales['dt'] <= date_list[i]]
    sale_sum = sale.groupby('shop_id', as_index=False).agg({'sale_amt_3m': ['sum']})
    sale_sum.columns = ['shop_id', 'sale_amt_3m']
    sale_sum['dt'] = d
    feature = pd.concat([feature, sale_sum])
    print(d)
    i = i+1
    print(i)
#输出
feature.rename(columns={'sale_amt_3m':'sale_feature_history'}, inplace=True)
feature.to_csv('./cache/static_sale_feature.csv', index=False)