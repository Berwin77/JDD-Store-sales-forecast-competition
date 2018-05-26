import pandas as pd
import numpy as np
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


#label
labels_path = '/Users/liu_bowen/PycharmProjects/Pycharm/JDD/data/t_sales_sum.csv'
labels = pd.read_csv(labels_path)
labels['dt'] = pd.to_datetime(labels['dt'])

labels.rename(columns={'sale_amt_3m':'label'}, inplace=True)

#构造训练集

#销售特征和评论特征
all_features = pd.merge(feature_all, comment_all, how='left', on=['shop_id', 'dt'])
#加product特征
all_features = pd.merge(all_features, product, how='left', on=['shop_id', 'dt'])


#加入static feature
all_features = pd.merge(all_features, static_feature, how='left', on=['shop_id', 'dt'])
#加入static comment
all_features = pd.merge(all_features, static_comment, how='left', on=['shop_id', 'dt'])
#加入static product
all_features = pd.merge(all_features, static_product, how='left', on=['shop_id', 'dt'])
print(all_features)

#加入label
all_features = pd.merge(all_features, labels, how='left', on=['shop_id', 'dt'])
all_features = all_features.fillna(0)
for i in range(1, 3001):
    feature = pd.DataFrame()
    feat = all_features[all_features['shop_id']==i]
    feature = pd.concat([feature, feat])
    f = [x for x in feature.columns if x not in ['label', 'shop_id', 'dt']]
    for col in feature[f].columns:
        feature[col] = (feature[col] - feature[col].mean()) / feature[col].std(ddof=0)
    feature = feature.fillna(0)
    feature.to_csv('./every_shop/shop%d.csv' % i,index=False)
    print(i)
print('Finish!')
