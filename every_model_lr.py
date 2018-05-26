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
for i in range(1, 5):
    if i%300 == 0:
        print(i,'============')


    #print(i)
    feat = pd.read_csv('./every_shop/shop%d.csv'%i)
    date = ['2017/1/31', '2017/4/30']
    date = pd.to_datetime(date)
    #print(feat)
    #print(np.isnan(feat.all()))
    feat['dt'] = pd.to_datetime(feat['dt'])
    f = [x for x in feat.columns if x not in ['label', 'shop_id', 'dt']]
    # for col in feat[f].columns:
    #     feat[col] = (feat[col] - feat[col].mean()) / feat[col].std(ddof=0)
    #train = feat[feat['dt'] <=date[0]]
    #X_train = np.asarray(train[f], dtype=np.float64)
    #Y_train = np.asarray(train['label'], dtype=np.float64)
    Y_train = np.asarray(feat[feat['dt'] <= date[0]]['label'], dtype=np.float64)
    m = len(Y_train)
    Date_all = np.array(feat[f], dtype=np.float64)
    pca = PCA(n_components=10)
    Date_pca = pca.fit_transform(Date_all)

    X_train = Date_all[:m]
    X_test = Date_all[-1].reshape(1, -1)
    #print(X_train.shape)
    #print(X_test.shape)


    #test = feat[(feat['dt'] == date[1])]
    #X_test = test[f]

    #print(X_train)
    #X_test = pca.fit_transform(X_test)


    #print(X_test)
    x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, test_size=0.16, random_state=10)
    model = LinearRegression()
    print(x_valid)
    #print(x_train)
    #print(np.isnan(x_train))
    model.fit(x_train, y_train)
    y_hat = model.predict(x_valid)
    pred.append(y_hat[0])
    real.append(y_valid[0])
    #feature1.loc[i] = {'pred': y_hat[0], 'real': y_valid[0]}

    y_hat2 = model.predict(X_test)
    if y_hat2[0] <=0:
        y_hat2[0] = 0
    pred2.append(y_hat2[0])
    shop.append(i)
    #feature.loc[i] = {'shop_id':i, 'pred_y': y_hat2[0]}
    # print('===================')

#print(feature1)
# result = wmaeEval(np.array(feature1['pred']), np.array(feature1['real']))
pred = np.array(pred).reshape((len(pred), 1))
real = np.array(real).reshape((len(real), 1))
result = wmaeEval(pred, real)
print(result)

ret = np.concatenate([pred, real], axis=1)
df = pd.DataFrame(ret, columns=['pred', 'real'])
df.to_csv('./result/shop_result_lr1.csv', index=False)


pred2 = np.array(pred2).reshape((len(pred2), 1))
shop = np.array(shop).reshape((len(shop), 1))

shop = np.array((range(1, 3001)), dtype=np.int32).reshape((3000, 1))
ret1 = np.concatenate([shop, pred2], axis=1)
feature = pd.DataFrame(ret1, columns=['shop_id', 'pred2'])

feature.to_csv('./result/shop_result_linear.csv', index=False)#, header=False)
print('Finish!')