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

#商品特征
product_path = '/Users/liu_bowen/PycharmProjects/Pycharm/JDD/cache/product_feature_all.csv'
product = pd.read_csv(product_path)
product['dt'] = pd.to_datetime(product['dt'])
#测试和训练月份的之前三个月销售额
sale_path = '/Users/liu_bowen/PycharmProjects/Pycharm/JDD/cache/sale_feature.csv'
sale_feature = pd.read_csv(sale_path)
sale_feature['dt'] = pd.to_datetime(sale_feature['dt'])

#广告数据特征
ads_path = '/Users/liu_bowen/PycharmProjects/Pycharm/JDD/cache/ads_feature_all.csv'
ads_all = pd.read_csv(ads_path)
ads_all['dt'] = pd.to_datetime(ads_all['dt'])



#增量feature
static_feature_all_path = './cache/static_feature.csv'
static_feature = pd.read_csv(static_feature_all_path)
static_feature['dt'] = pd.to_datetime(static_feature['dt'])

#增量comment
static_comment_all_path = './cache/static_comment.csv'
static_comment = pd.read_csv(static_comment_all_path)
static_comment['dt'] = pd.to_datetime(static_comment['dt'])

#增量商品特征
static_product_path = '/Users/liu_bowen/PycharmProjects/Pycharm/JDD/cache/static_product_feature.csv'
static_product = pd.read_csv(static_product_path)
static_product['dt'] = pd.to_datetime(product['dt'])
#增量sale特性
static_sale_path = '/Users/liu_bowen/PycharmProjects/Pycharm/JDD/cache/static_sale_feature.csv'
static_sale_feature = pd.read_csv(static_sale_path)
static_sale_feature['dt'] = pd.to_datetime(static_sale_feature['dt'])
#增量广告特征
static_ads_path = '/Users/liu_bowen/PycharmProjects/Pycharm/JDD/cache/static_ads_feature.csv'
static_ads_all = pd.read_csv(static_ads_path)
static_ads_all['dt'] = pd.to_datetime(ads_all['dt'])

#label
labels_path = '/Users/liu_bowen/PycharmProjects/Pycharm/JDD/data/t_sales_sum.csv'
labels = pd.read_csv(labels_path)
labels['dt'] = pd.to_datetime(labels['dt'])

labels.rename(columns={'sale_amt_3m':'label'}, inplace=True)

#构造测试集和测试集

#确定时间段

valid_time = datetime.date(2016, 10, 31)



#构造训练集
valid = feature_all[(valid_time == feature_all['dt'])]
#销售特征和评论特征
valid = pd.merge(valid, comment_all, how='left', on=['shop_id', 'dt'])
#加product特征
valid = pd.merge(valid, product, how='left', on=['shop_id', 'dt'])
#加入广告投放特征
valid = pd.merge(valid, ads_all, how='left', on=['shop_id', 'dt'])

#加入3个月前的sale
valid = pd.merge(valid, sale_feature, how='left', on=['shop_id', 'dt'])

#加入static feature
valid = pd.merge(valid, static_feature, how='left', on=['shop_id', 'dt'])
#加入static comment
valid = pd.merge(valid, static_comment, how='left', on=['shop_id', 'dt'])
#加入static product
valid = pd.merge(valid, static_product, how='left', on=['shop_id', 'dt'])
#加入static sale
valid = pd.merge(valid, static_sale_feature, how='left', on=['shop_id', 'dt'])
#加入static ads
valid = pd.merge(valid, static_ads_all, how='left', on=['shop_id', 'dt'])

#加入label
valid = pd.merge(valid, labels, how='left', on=['shop_id', 'dt'])


# for i in train.columns[0:]:
#     train[i] = train[i].fillna(0)
valid = valid.fillna(0)

#print(np.isfinite(train.any()))
#print(np.isnan(train.any()))
print(len(valid))

valid.to_csv('/Users/liu_bowen/PycharmProjects/Pycharm/JDD/demo/valid_10_31.csv', index=False)
print('Finish!!!')