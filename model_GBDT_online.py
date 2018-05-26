import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV



def wmaeEval(preds, dtrain):
    label = dtrain
    return 'wmaeEval', np.sum(np.abs(preds - label)) / np.sum(label)


train = pd.read_csv('./demo/underline_train_1_31.csv')
test = pd.read_csv('./demo/underline_test_4_30.csv')
valid = pd.read_csv('./demo/valid_10_31.csv')


print(len(train))
print(len(test))

feature = [x for x in train.columns if x not in ['label', 'shop_id', 'dt']]
#print(type(feature))
#进行正则化
for col in train[feature].columns:
    train[col] = (train[col] - train[col].mean()) / train[col].std(ddof=0)
for col in test[feature].columns:
    test[col] = (test[col] - test[col].mean()) / test[col].std(ddof=0)
for col in valid[feature].columns:
    valid[col] = (valid[col] - valid[col].mean()) / valid[col].std(ddof=0)

X = train[feature]
y = train['label']
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=100)
x1_valid, x2_valid, y1_valid, y2_valid = train_test_split(x_valid, y_valid, test_size=0.4, random_state=50)
x_test = test[feature]
#y_test = test['label']

valid_X = valid[feature]
valid_y = valid['label']
params = {
          'n_estimators': 1000,
          'max_depth': 9,
          #'max_features':,
          'subsample':0.85,
          'min_samples_leaf': 7,
          'min_samples_split': 16,
          'learning_rate': 0.05,
          'loss': 'lad'}

#param_test1 ={'n_estimators':np.array(range(100,1001,50))}
#param_test1 ={'max_depth':np.array(range(3,14,2))}
#param_test1 ={'min_samples_split':np.array(range(2,17,2))}
#param_test1 ={'min_samples_leaf':np.array(range(1,18,2))}
#param_test1 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
#param_test1 = {'max_features':np.array(range(15,34,2))}
clf = GradientBoostingRegressor(**params)

#gsearch1 = GridSearchCV(estimator=clf, param_grid=param_test1, scoring='neg_mean_absolute_error', cv=3)
# clf.fit(x_train, y_train)
# pred = clf.predict(x1_valid)
# gsearch1.fit(x_train, y_train)
# print(gsearch1.grid_scores_)
# print(gsearch1.best_params_)
# print(gsearch1.best_score_)
# params['max_features'] = gsearch1.best_params_['max_features']
clf = GradientBoostingRegressor(**params)
clf.fit(x_train, y_train)
pred = clf.predict(x1_valid)
valid_pred = clf.predict(valid_X)
print('============')
print(wmaeEval(valid_pred, valid_y))
print('==========')
df_valid = pd.DataFrame(valid_pred, columns=['pred_valid'])
df_valid.insert(0, 'shop_id', valid['shop_id'])
df_valid.insert(2, 'real', valid['label'])
df_valid.to_csv('./valid/valid_10_GBDT.csv', index= False)
##########################################################

print(wmaeEval(pred, y1_valid))
pred1 = clf.predict(x2_valid)
print(wmaeEval(pred1, y2_valid))
print('finish1!!!')
pred_test = clf.predict(x_test)
df = pd.DataFrame(pred_test, columns=['sale_num'])
df[df['sale_num']< 0]= 0
df.insert(0, 'shop_id', test['shop_id'])
df.insert(1, 'dt', test['dt'])
# 取其中的4月
result_day = datetime.date(2017,4,30)
df['dt'] = pd.to_datetime(df['dt'])
df = df[df['dt'] == result_day]
df.pop('dt')

#输出
df.to_csv('./result/shop_result_GBDT.csv', index=False)#, header= False)
print('finish!!!')

