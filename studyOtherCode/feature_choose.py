import pandas as pd
import math
t_order = pd.read_csv('../data/t_order.csv')
t_order['year_month'] = t_order['ord_dt'].map(lambda x:str(x) [:7])
t_order_sum = t_order[t_order.columns.drop(['ord_dt','pid'])].groupby(['shop_id','year_month'],as_index=False).sum()
month_4_sale = t_order_sum[t_order_sum['year_month']=='2016-11'][['sale_amt','shop_id','offer_amt','rtn_amt']]
month_4_sale['sale_amt'] = month_4_sale['sale_amt']
print(len(month_4_sale))
print('------------------')
true_12_sale = pd.read_csv('../data/t_sales_sum.csv')
true_12_sale['year_month'] = true_12_sale['dt'].map(lambda x:str(x)[:7])
true_12_sale = true_12_sale[true_12_sale['year_month']=='2016-11'].drop_duplicates('shop_id')
print(len(true_12_sale))
print(true_12_sale)
print('++++++++++++++++++++++')
all = pd.merge(month_4_sale, true_12_sale, on='shop_id').drop_duplicates('shop_id')#_drop_duplicates()去重操作
print(all)
all['bizhi'] = all['sale_amt_3m'] / all['sale_amt']
print (len(all['bizhi']))
print('-----------------')
print(all['bizhi'])
print(sum(all['sale_amt'] / all['sale_amt_3m'])/3000.0)