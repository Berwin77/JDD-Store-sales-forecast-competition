import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
#feature1 = pd.DataFrame({'pred':'', 'real':''},index=["0"])
#feature = pd.DataFrame({'shop_id':'', 'pred':''},index=["0"])

def wmaeEval(preds, dtrain):
    label = dtrain
    return 'wmaeEval', np.sum(np.abs(preds - label)) / np.sum(label)
pred = []
real = []
pred2 = []
shop = []
for i in range(1, 3001):
    if i%300 == 0:
        print(i,'============')


    #print(i)
    feat = pd.read_csv('./every_shop/shop%d.csv'%i)
    date = ['2017/1/31', '2016/10/31', '2017/4/30']
    date = pd.to_datetime(date)
    #print(feat)
    #print(np.isnan(feat.all()))
    feat['dt'] = pd.to_datetime(feat['dt'])
    f = [x for x in feat.columns if x not in ['label', 'shop_id', 'dt']]

    train = feat[(feat['dt'] <= date[0])&(feat['dt'] != date[1])]
    X_train = np.asarray(train[f], dtype=np.float64)
    Y_train = np.asarray(train['label'], dtype=np.float64)

    valid = feat[feat['dt'] == date[1]]
    valid_X = np.asarray(train[f], dtype=np.float64)
    valid_y = np.asarray(valid['label'], dtype=np.float64)



    test = feat[(feat['dt'] == date[2])]
    X_test = test[f]



    #print(X_test)

    model = LinearRegression()

    model.fit(X_train, Y_train)
    y_hat = model.predict(valid_X)
    pred.append(y_hat[0])
    real.append(valid_y[0])

    y_hat2 = model.predict(X_test)
    if y_hat2[0] <=0:
        y_hat2[0] = 0
    pred2.append(y_hat2[0])


#print(feature1)
# result = wmaeEval(np.array(feature1['pred']), np.array(feature1['real']))
pred = np.array(pred).reshape((len(pred), 1))
real = np.array(real).reshape((len(real), 1))
result = wmaeEval(pred, real)
print(result)

ret = np.concatenate([pred, real], axis=1)
df = pd.DataFrame(ret, columns=['pred', 'real'])
df.to_csv('./valid/valid_10_linear.csv', index=False)


pred2 = np.array(pred2).reshape((len(pred2), 1))




feature = pd.DataFrame(pred2, columns=['pred2'])

feature.to_csv('./result/shop_result_linear.csv', index=False)#, header=False)
print('Finish!')