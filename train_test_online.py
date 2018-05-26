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
train_begin = datetime.date(2016,11, 30)
train_end = datetime.date(2017, 1, 31)

test_begin = datetime.date(2017,2,28)
test_end = datetime.date(2017,4,30)


#构造集合
# train = pd.DataFrame()
# test = pd.DataFrame()
#构造训练集
train = feature_all[(train_begin<= feature_all['dt']) & (feature_all['dt']<=train_end)]
#销售特征和评论特征
train = pd.merge(train, comment_all, how='left', on=['shop_id', 'dt'])
#加product特征
train = pd.merge(train, product, how='left', on=['shop_id', 'dt'])
#加入广告投放特征
train = pd.merge(train, ads_all, how='left', on=['shop_id', 'dt'])

#加入3个月前的sale
train = pd.merge(train, sale_feature, how='left', on=['shop_id', 'dt'])

#加入static feature
train = pd.merge(train, static_feature, how='left', on=['shop_id', 'dt'])
#加入static comment
train = pd.merge(train, static_comment, how='left', on=['shop_id', 'dt'])
#加入static product
train = pd.merge(train, static_product, how='left', on=['shop_id', 'dt'])
#加入static sale
train = pd.merge(train, static_sale_feature, how='left', on=['shop_id', 'dt'])
#加入static ads
train = pd.merge(train, static_ads_all, how='left', on=['shop_id', 'dt'])

#加入label
train = pd.merge(train, labels, how='left', on=['shop_id', 'dt'])

#构造测试集

test = feature_all[(test_begin<= feature_all['dt']) & (feature_all['dt']<=test_end)]
#销售特征和评论特征
test = pd.merge(test, comment_all, how='left', on=['shop_id', 'dt'])
#加product特征
test = pd.merge(test, product, how='left', on=['shop_id', 'dt'])

#加入广告投放特征
test = pd.merge(test, ads_all, how='left', on=['shop_id', 'dt'])

#加入3个月前的sale
test = pd.merge(test, sale_feature, how='left', on=['shop_id', 'dt'])
#加入static feature
test = pd.merge(test, static_feature, how='left', on=['shop_id', 'dt'])
#加入static comment
test = pd.merge(test, static_comment, how='left', on=['shop_id', 'dt'])
#加入static product
test = pd.merge(test, static_product, how='left', on=['shop_id', 'dt'])
#加入static sale
test = pd.merge(test, static_sale_feature, how='left', on=['shop_id', 'dt'])
#加入static ads
test = pd.merge(test, static_ads_all, how='left', on=['shop_id', 'dt'])

#加入label
test = pd.merge(test, labels, how='left', on=['shop_id', 'dt'])


# for i in train.columns[0:]:
#     train[i] = train[i].fillna(0)
train = train.fillna(0)
test = test.fillna(0)
#print(np.isfinite(train.any()))
#print(np.isnan(train.any()))
print(len(train))
print(len(test))
train.to_csv('/Users/liu_bowen/PycharmProjects/Pycharm/JDD/demo/underline_train_1_31.csv', index=False)
test.to_csv('/Users/liu_bowen/PycharmProjects/Pycharm/JDD/demo/underline_test_4_30.csv', index=False)
print('Finish!!!')