import xgboost
import operator
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
from sklearn import cross_validation


train = pd.read_csv('./demo/underline_train_1_31.csv')
test = pd.read_csv('./demo/underline_test_4_30.csv')
valid = pd.read_csv('./demo/valid_10_31.csv')

print(len(train))
print(len(test))
print(len(valid))
feature = [x for x in train.columns if x not in ['label', 'shop_id', 'dt']]

# 进行正则化
for col in train[feature].columns:
    train[col] = (train[col] - train[col].mean()) / train[col].std(ddof=0)
for col in test[feature].columns:
    test[col] = (test[col] - test[col].mean()) / test[col].std(ddof=0)
for col in valid[feature].columns:
    valid[col] = (valid[col] - valid[col].mean()) / valid[col].std(ddof=0)
X = train[feature]
y = train['label']
Dttrain = xgboost.DMatrix(X, label=y)
x_test = test[feature]
#y_test = test['label']

valid_X = valid[feature]
valid_y = valid['label']

#进行模型
x_train, x_valid, y_train, y_valid = train_test_split(X, y,random_state=1 ,test_size=0.3)
#x1_valid, x2_valid, y1_valid, y2_valid = train_test_split(x_valid, y_valid, test_size=0.4)

xgbTrain = xgboost.DMatrix(x_train, label=y_train)
xgbVal = xgboost.DMatrix(x_valid, label=y_valid)
watchlist = [(xgbTrain, 'train'), (xgbVal, 'eval')]
xgbTest = xgboost.DMatrix(valid_X, label=valid_y)
xgbTest2 = xgboost.DMatrix(x_test)
def mapeObj(preds, dtrain):
    gaps = dtrain.get_label()
    grad = np.sign(preds - gaps) / gaps
    hess = 1 / np.abs(preds - gaps)
    return grad, hess

def huber_approx_obj(preds, dtrain):
    d = preds - dtrain.get_label() # remove .get_labels() for sklearn
    h = 10000000  # h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess

def log_cosh_obj(preds, dtrain):
    x = preds - dtrain.get_label()
    grad = np.tanh(x)
    hess = 1 / np.cosh(x)**2
    return grad, hess

def fair_obj(preds, dtrain):
    """y = c * abs(x) - c * np.log(abs(abs(x) + c))"""
    x = preds - dtrain.get_label()
    c = 100000
    den = abs(x) + c
    grad = (c * x) / den
    hess = (c * c) / (den * den)
    return grad, hess

def wmaeEval(preds, dtrain):
    label = dtrain.get_label()
    return 'wmaeEval', np.sum(np.abs(preds - label)) / np.sum(label)

param = {}
#param['eval_metric'] = 'rmse'
param['eta'] = 0.01
param['max_depth'] = 8
param['min_child_weight'] = 4
param['subsample'] = 0.8
param['colsample_bytree'] = 0.3
param['silent'] = 1
# param = {
#             'booster':'gbtree',
#             'objective':'reg:linear',
#             'eta':0.01,
#             'max_depth':10,
#             'subsample':0.8,
#             'min_child_weight':5,
#             'colsample_bytree':0.3,
#             'scale_pos_weight':0.1,
#             'eval_metric':'rmse',
#             'gamma':0.2,
#             'silent':1
# }

num_round = 1000
model = xgboost.train(param, xgbTrain, num_round, watchlist,
                  obj=fair_obj,feval=wmaeEval, early_stopping_rounds=100)
pred1 = model.predict(xgbTest)
print('-----------------------------------------')

loss = wmaeEval(pred1, xgbTest)
print('loss=', loss)
df_valid = pd.DataFrame(pred1, columns=['pred'])
df_valid.insert(0, 'shop_id', valid['shop_id'])
df_valid.insert(2, 'real', valid['label'])
df_valid.to_csv('./valid/valid_10_xgb.csv', index= False)
print('========================')
pred2 = model.predict(xgbTest2)
df = pd.DataFrame(pred2, columns=['sale_num'])
df[df['sale_num'] < 0] = 0



df.insert(0, 'shop_id', test['shop_id'])
df.insert(1, 'dt', test['dt'])
#多个月
# result_begin = datetime.date(2017,2,28)
# result_end = datetime.date(2017,4,30)
#df1 = df[(result_begin <=df['dt'])&(df['dt']<=result_end)]
#一个月
result_days = datetime.date(2017,4,30)
df['dt'] = pd.to_datetime(df['dt'])
df = df[df['dt']==result_days]
df.pop('dt')



df.to_csv('./result/shop_result_xgb.csv', columns=['shop_id', 'pred'], index=False)

print('===========================================')
importance = model.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
df.to_csv('./result/特征重要性.csv', index=False)

print('finish!!')