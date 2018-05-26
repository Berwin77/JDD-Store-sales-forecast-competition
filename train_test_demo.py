import pandas as pd
import numpy as np
import datetime

##我们要获取所有数据特征
#主数据特征
feature_all_path = '/Users/liu_bowen/PycharmProjects/Pycharm/JDD/cache/feature_all.csv'
feature_all = pd.read_csv(feature_all_path)
feature_all['dt'] = pd.to_datetime(feature_all['dt'])
#评论数据特征
comment_path = '/Users/liu_bowen/PycharmProjects/Pycharm/JDD/cache/comment_feature_all.csv'
comment_all = pd.read_csv(comment_path)
comment_all['dt'] = pd.to_datetime(comment_all['dt'])

#广告数据特征
ads_path = '/Users/liu_bowen/PycharmProjects/Pycharm/JDD/cache/ads_feature_all.csv'
ads_all = pd.read_csv(ads_path)
ads_all['dt'] = pd.to_datetime(ads_all['dt'])

#label
labels_path = '/Users/liu_bowen/PycharmProjects/Pycharm/JDD/data/t_sales_sum.csv'
labels = pd.read_csv(labels_path)
labels['dt'] = pd.to_datetime(labels['dt'])
labels['sale_amt_3m'] = np.log(labels['sale_amt_3m'])

labels.rename(columns={'sale_amt_3m':'label'}, inplace=True)

#构造测试集和测试集

#确定时间段
train_begin = datetime.date(2016, 8, 31)
train_end = datetime.date(2016, 10, 31)

test_begin = datetime.date(2016,11,30)
test_end = datetime.date(2017,1,31)


#构造集合
# train = pd.DataFrame()
# test = pd.DataFrame()
#构造训练集
train = feature_all[(train_begin<= feature_all['dt']) & (feature_all['dt']<=train_end)]
#销售特征和评论特征
train = pd.merge(train, comment_all, how='left', on=['shop_id', 'dt'])
#加入广告投放特征
#train = pd.merge(train, ads_all, how='left', on=['shop_id', 'dt'])
#加入label
train = pd.merge(train, labels, how='left', on=['shop_id', 'dt'])

#构造测试集

test = feature_all[(test_begin<=feature_all['dt'])&(feature_all['dt']<=test_end)]
#销售特征和评论特征
test = pd.merge(test, comment_all, how='left', on=['shop_id', 'dt'])
#加入广告投放特征
test = pd.merge(test, ads_all, how='left', on=['shop_id', 'dt'])
#加入label
test = pd.merge(test, labels, how='left', on=['shop_id', 'dt'])

train.to_csv('/Users/liu_bowen/PycharmProjects/Pycharm/JDD/demo/underline_train_8_10.csv', index=False)
test.to_csv('/Users/liu_bowen/PycharmProjects/Pycharm/JDD/demo/underline_test_11_1.csv', index=False)

print('Finish!!!')