import pandas as pd
import datetime
import numpy as np

product_feature_path = './data/t_product.csv'
products = pd.read_csv(product_feature_path)

products['on_dt'] = pd.to_datetime(products['on_dt'])

date_list = ['2016/8/31', '2016/9/30', '2016/10/31', '2016/11/30',
              '2016/12/31', '2017/1/31', '2017/2/28', '2017/3/31', '2017/4/30']
delta = datetime.timedelta(days=30)
date_list = pd.to_datetime(date_list)

feature = pd.DataFrame()
for d in date_list:
    product = products[(date_list[0] - delta < products['on_dt'])&(products['on_dt'] <= d)]
    #consume  一定注意as_index这个值 用False代表 groupby的值不做index

    consume_num = product.groupby('shop_id', as_index=False).agg({'pid':['count']})

    consume_num.columns = ['shop_id', 'count']
    consume_num['dt'] = d
    feature = pd.concat([feature, consume_num])
    print(d)

# fea = [x for x in feature.columns if x not in ['dt', 'shop_id']]
# for col in feature[fea].columns:
#     feature[col] = (feature[col] - feature[col].mean()) / feature[col].std(ddof=0)


#重命名列名
for i in feature.columns[0:]:
    feature.rename(columns={i:i+'_history'}, inplace=True)
feature.rename(columns={'dt_history':'dt', 'shop_id_history':'shop_id'}, inplace=True)


feature.to_csv('./cache/static_product_feature.csv', index=False)
print('finish!')