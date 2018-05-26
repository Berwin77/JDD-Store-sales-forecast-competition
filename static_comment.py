import numpy as np
import pandas as pd
import datetime as dt

commentall = pd.read_csv('./data/t_comment.csv')
commentall['create_dt'] = pd.to_datetime(commentall['create_dt'])

date_list = ['2016/6/30','2016/7/31','2016/8/31','2016/9/30','2016/10/31','2016/11/30','2016/12/31','2017/1/31','2017/2/28','2017/3/31','2017/4/30']
date_list = pd.to_datetime(date_list)
feature = pd.DataFrame()
for d in date_list:
    time_interval = commentall[commentall['create_dt'] <= d]
    #bad_amt
    bad_num = time_interval.groupby('shop_id', as_index=False).agg({'bad_num':['sum']})
    bad_num.columns = ['shop_id', 'bad_num_sum']
    #cmmt_num
    cmmt_num = time_interval.groupby('shop_id', as_index=False).agg({'cmmt_num':['sum']})
    cmmt_num.columns = ['shop_id', 'cmmt_num_sum']
    all_feature = pd.merge(bad_num, cmmt_num, on='shop_id')
    #dis_num
    dis_num = time_interval.groupby('shop_id', as_index=False).agg({'dis_num': ['sum']})
    dis_num.columns = ['shop_id', 'dis_num_sum']
    all_feature = pd.merge(all_feature, dis_num, on='shop_id')
    #good_num
    good_num = time_interval.groupby('shop_id', as_index=False).agg({'good_num': ['sum']})
    good_num.columns = ['shop_id', 'good_num_sum']
    all_feature = pd.merge(all_feature, good_num, on='shop_id')
    #mid_num
    mid_num = time_interval.groupby('shop_id', as_index=False).agg({'mid_num':['sum']})
    mid_num.columns = ['shop_id', 'mid_num_sum']
    all_feature = pd.merge(all_feature, mid_num, on='shop_id')

    all_feature['dt'] = d
    feature = pd.concat([feature, all_feature])
    print(d)

# 计算评价率
feature['cmmt_num_sum'].replace(0, 0.01)
# 差评率
feature['bad/cmmt'] = feature['bad_num_sum'] / feature['cmmt_num_sum']
# 晒单率
feature['dis/cmmt'] = feature['dis_num_sum'] / feature['cmmt_num_sum']
# 好评率
feature['good/cmmt'] = feature['good_num_sum'] / feature['cmmt_num_sum']
# 中评率
feature['mid/cmmt'] = feature['mid_num_sum'] / feature['cmmt_num_sum']
#差/好评率
#feature['bad/good'] = feature['bad_num_sum'] / feature['good_num_sum']
#评价总数：做归一化

#feature['cmmt_num_sum_f'] = feature['cmmt_num_sum'] - feature['cmmt_num_sum'].mean()/feature['cmmt_num_sum'].std(ddof=0)
feature['cmmt_num_sum_f'] = feature['cmmt_num_sum']
#整理
feature = feature[['shop_id', 'dt', 'bad/cmmt', 'dis/cmmt', 'good/cmmt', 'mid/cmmt', 'cmmt_num_sum_f']]

#重命名列名
for i in feature.columns[1:]:
    feature.rename(columns={i:i+'_history'}, inplace=True)
feature.rename(columns={'dt_history':'dt', 'shop_id_history':'shop_id'}, inplace=True)
#输出
feature.to_csv('./cache/static_comment.csv', index=False)