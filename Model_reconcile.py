import pandas as pd
import numpy as np
def wmaeEval(preds, dtrain):
    label = dtrain
    return 'wmaeEval', np.sum(np.abs(preds - label)) / np.sum(label)



valid1 = pd.read_csv('./valid/valid_10_GBDT.csv')
#valid2 = pd.read_csv('./valid/valid_10_linear.csv')
valid2 = pd.read_csv('./valid/valid_10_xgb.csv')
valid = pd.DataFrame()
valid['shop_id'] = valid1['shop_id']
valid['pred1'] = valid1['pred_valid']
valid['pred2'] = valid2['pred']
valid['real1'] = valid1['real']
valid['real2'] = valid2['real']
valid['pred_s'] = valid['pred1']*0.5+valid['pred2']*0.5
result1 = wmaeEval(valid['pred_s'], valid['real1'])
result2 = wmaeEval(valid['pred_s'], valid['real2'])
valid.to_csv('./valid/valid_reconcile.csv', index=False)
print(result1)
print(result2)

test1 = pd.read_csv('/Users/liu_bowen/PycharmProjects/Pycharm/JDD/result/shop_result_GBDT.csv')
test2 = pd.read_csv('/Users/liu_bowen/PycharmProjects/Pycharm/JDD/result/shop_result_xgb.csv')
test = pd.DataFrame()
test['shop_id'] = test1['shop_id']
test['pred1'] = test1['sale_num']
test['pred2'] = test2['pred']
test['pred_s'] = test['pred1']*0.5+test['pred2']*0.5
test.to_csv('./result/model_reconcile_6_4.csv', columns=['shop_id', 'pred_s'], index=False, header=False)

