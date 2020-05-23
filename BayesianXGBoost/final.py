'''
cbz
lopt
'''
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import os
from math import *
import time

'''
author: Bozhou Chen
'''

EPSILON = 0.001
EPSILON_threshold = 0.00001
INF = 100000
seg_tree = None
key_to_index = {}
index_to_key = {}
index_to_value = {}
infparams = ['max_delta_step', 'gamma', 'min_child_weight', 'max_depth', 'reg_lambda', 'reg_alpha', 'n_estimators']



def lopt(P, X, Y):
    '''
    Do LOPT to hyper-parameter
    return a Dict
    '''
    start = time.time()
    l = 1
    r = len(P)
    allkeys = P.keys()
    num = 1
    for key in allkeys:
        key_to_index[key] = num
        index_to_key[num] = key
        index_to_value[num] = P[key]
        num += 1
    nvalue = P['n_estimators']
    acc = 0
    if nvalue > 35:
        P['n_estimators'] = 10
        P = func(P, X, Y, l, r)
        P['n_estimators'] = 40
        acc = accu(P, X, Y)
    else:
        P = func(P, X, Y, l, r)
    value = P['reg_alpha']
    for i in range(41):
        P['reg_alpha'] = i * 0.005
        tmp = accu(P, X, Y)
        if tmp > acc:
            tmp = acc
            value = P['reg_alpha']
    P['reg_alpha'] = value
    if nvalue > 35:
        P['n_estimators'] = max(nvalue, 70)
    else:
        P = sMC(P, 'n_estimators', X, Y)
    return P, time.time() - start


def norm(P):
    keys = P.keys()
    for key in keys:
        p = P[key]
        if key == 'max_delta_step':
            p = parameter_clean(p, (1, 10), True, True, 'float')
        elif key == 'gamma':
            p = parameter_clean(p, (0, 30), True, True, 'float')
        elif key == 'min_child_weight':
            p = parameter_clean(p, (0, 30), True, True, 'float')
        elif key == 'max_depth':
            p = parameter_clean(p, (3, 30), True, True, 'float')
        elif key == 'reg_lambda':
            p = parameter_clean(p, (0, 2), True, True, 'float')
        elif key == 'subsample':
            p = parameter_clean(p, (0.5, 1), False, True, 'float')
        elif key == 'colsample_bytree':
            p = parameter_clean(p, (0.5, 1), False, True, 'float')
        elif key == 'colsample_bylevel':
            p = parameter_clean(p, (0.5, 1), False, True, 'float')
        elif key == 'learning_rate':
            p = parameter_clean(p, (0.01, 0.2), True, True, 'float')
        elif key == 'reg_alpha':
            p = parameter_clean(p, (0, 2), True, True, 'float')
        elif key == 'n_estimators':
            # We are uncertain about the range here
            p = parameter_clean(p, (1, 300), True, True, 'int')
        P[key] = p
        P[key] = p
    return P


def parameter_clean(parameter, range, include_min, include_max, type, delta=1e-5, inf=1e10):
    min = range[0]
    max = range[1]
    if min == 'inf':
        min = -inf
    if max == 'inf':
        max = inf
    p = parameter
    # adjust data range
    if include_min and p < min:
        p = min
    if not include_min and p <= min:
        if type == 'int':
            p = min + 1
        else:
            p = min + delta
    if include_max and p > max:
        p = max
    if not include_max and p >= max:
        if type == 'int':
            p = max - 1
        else:
            p = max - delta
    # then adjust data type
    if type == 'int':
        p = int(p)
    return p


def func(P, X, Y, l, r):
    print('recursion')
    print(l, r)
    if l == r:
        P = MC(P, index_to_key[l], X, Y)
        return P
    if l == r - 1:
        P = DMC(P, index_to_key[l], index_to_key[r], X, Y)
        return P
    mid = int((l + r) / 2)
    P = func(P, X, Y, l, mid)
    P = func(P, X, Y, mid + 1, r)
    return P


def sMC(P, key, X, Y):
    stride = 2
    threshold = 0.2
    value_x = P[key]
    value_last_x = INF
    s_value_x = value_x
    print('Start an MC process')
    print('optimized hyperparameter set is ' + str(key))
    while stride > threshold:
        P[key] = value_x - stride
        P = norm(P)
        a = accu(P, X, Y)
        print(P)
        print(a)
        print()
        P[key] = value_x
        P = norm(P)
        b = accu(P, X, Y)
        print(P)
        print(b)
        print()
        P[key] = value_x + stride
        P = norm(P)
        c = accu(P, X, Y)
        print(P)
        print(c)
        print()
        if a > b:
            P[key] = value_x - stride
        elif c > b:
            P[key] = value_x + stride
        else:
            stride = stride / 2
            P[key] = value_x
            continue
        value_last_x = value_x
        value_x = P[key]
    print('an MC process finish')
    # update(index,abs(value_x - s_value_x))
    return P


def MC(P, key, X, Y):
    stride = EPSILON
    threshold = EPSILON_threshold
    if infparams.__contains__(key):
        stride = 1
        threshold = 0.1
    if key == 'n_estimators':
        return P
    if key == 'reg_alpha':
        return P
        # stide = 40
        # threshold = n_threshold
    value_x = P[key]
    value_last_x = INF
    s_value_x = value_x
    print('start a MC training')
    print('optimized hyper-parameter set is ' + str(key))
    while stride > threshold:
        P[key] = value_x - stride
        P = norm(P)
        a = accu(P, X, Y)
        print(P)
        print(a)
        print()
        P[key] = value_x
        P = norm(P)
        b = accu(P, X, Y)
        print(P)
        print(b)
        print()
        P[key] = value_x + stride
        P = norm(P)
        c = accu(P, X, Y)
        print(P)
        print(c)
        print()
        if a > b:
            P[key] = value_x - stride
        elif c > b:
            P[key] = value_x + stride
        else:
            stride = stride / 2
            P[key] = value_x
            continue
        value_last_x = value_x
        value_x = P[key]
    print('an MC progress finish')
    # update(index,abs(value_x - s_value_x))
    return P


def DMC(P, key_st, key_nd, X, Y):
    P = MC(P, key_st, X, Y)
    P = MC(P, key_nd, X, Y)
    return P


def update(index, value):
    return


def check_over(l, r):
    return


def accu(P, X, Y):
    '''
    Compute accuracy when using P to run algorithm
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=33)
    func = "multi:softmax"
    func1 = "mlogloss"
    ssset = set(Y)
    n = len(ssset)
    if n == 2:
        func = "binary:logitraw"
        func1 = "logloss"

    model = xgb.XGBClassifier(
        booster='gbtree',
        objective=func,
        eval_metric='auc',
        tree_method='exact',
        silent=False,
        n_jobs=4,
        seed=7,
        nthread=4,
        max_delta_step=int(P["max_delta_step"]),
        gamma=P["gamma"],
        min_child_weight=int(P["min_child_weight"]),
        max_depth=int(P["max_depth"]),
        reg_lambda=P["reg_lambda"],
        reg_alpha=P["reg_alpha"],
        subsample=P["subsample"],
        colsample_bytree=P["colsample_bytree"],
        colsample_bylevel=P["colsample_bylevel"],
        learning_rate=P["learning_rate"],
        n_estimators=int(P["n_estimators"]),
    )
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=33)
    model.fit(x_train,
              y_train,
              eval_set=[(x_val, y_val)],
              eval_metric=func1,
              verbose=True)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    return accuracy


def handle(x_train, y_train, x_test, y_test, dic, n):
    func = "multi:softmax"
    func1 = "mlogloss"
    if n == 2:
        func = "binary:logitraw"
        func1 = "logloss"

    model = xgb.XGBClassifier(
        booster='gbtree',
        objective=func,
        eval_metric='auc',
        tree_method='exact',
        silent=False,
        n_jobs=4,
        seed=7,
        nthread=4,
        max_delta_step=int(dic["max_delta_step"]),
        gamma=dic["gamma"],
        min_child_weight=int(dic["min_child_weight"]),
        max_depth=int(dic["max_depth"]),
        reg_lambda=dic["reg_lambda"],
        reg_alpha=dic["reg_alpha"],
        subsample=dic["subsample"],
        colsample_bytree=dic["colsample_bytree"],
        colsample_bylevel=dic["colsample_bylevel"],
        learning_rate=dic["learning_rate"],
        n_estimators=int(dic["n_estimators"]),
    )

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=33)

    model.fit(x_train,
              y_train,
              eval_set=[(x_val, y_val)],
              eval_metric=func1,
              verbose=True)

    ### make prediction for test data
    y_pred = model.predict(x_test)

    ### model evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print("accuarcy: %.2f%%" % (accuracy * 100.0))
    return accuracy


'''
Test
'''
# id = 16
# traindatapath = 'E:/TrainData/' + str(id) + '.csv'
# testdatapath = 'E:/TestData/' + str(id) + '.csv'
# keys = ['max_delta_step', 'gamma', 'min_child_weight', 'max_depth', 'reg_lambda', 'subsample',
#                 'colsample_bytree', 'colsample_bylevel', 'learning_rate', 'reg_alpha', 'n_estimators']
#
# df = pd.read_csv(traindatapath)
# Y_train = df['Label'].values
# X_train = df.drop('Label',axis=1).values
# df = pd.read_csv(testdatapath)
# Y_test = df['Label'].values
# X_test = df.drop('Label',axis=1).values
# df = pd.read_csv('predicted_parameter.csv')[[str(id)]].values
# pcn = {}
# i = 0
# for key in keys:
#     pcn[key] = df[i][0]
#     i += 1
# df = pd.read_csv('BayesParameters/' + str(id) + '.csv').values
# pb = {}
# i = 0
# for key in keys:
#     pb[key] = df[0][i]
#     i += 1
# print(pcn)
# print(pb)
# pcn = norm(pcn)
#
# plopt,lopt_time = lopt(pcn,X_train,Y_train,l=1,r=len(pcn))
# # print('CN的输出是')
# # print(pcn)
# print('LOPT的输出是')
# print(plopt)
# print('LOPT的预测时间是')
# print(lopt_time)
# # # print('CN的预测结果是')
# # # print(handle(X_train,Y_train,X_test,Y_test,plopt,len(set(Y_train))))
# # print('LOPT的预测结果是')
# # plopt = {'max_delta_step': 3.021674156188965, 'gamma': 0, 'min_child_weight': 0.6084884405136108, 'max_depth': 30, 'reg_lambda': 1.1372474431991575, 'subsample': 0.6993720798492431, 'colsample_bytree': 0.6124504423141479, 'colsample_bylevel': 0.6392967190742492, 'learning_rate': 0.12428573429584505, 'reg_alpha': 0.35, 'n_estimators': 300}
# # plopt = {'max_delta_step': 3.896674156188965, 'gamma': 0, 'min_child_weight': 0.4834884405136109, 'max_depth': 30, 'reg_lambda': 1.5122474431991575, 'subsample': 0.6991064548492432, 'colsample_bytree': 0.611434817314148, 'colsample_bylevel': 0.6392810940742493, 'learning_rate': 0.12327010929584505, 'reg_alpha': 2, 'n_estimators': 10}
# # plopt = {'max_delta_step': 2.701248526573181, 'gamma': 0.2059753388166428, 'min_child_weight': 0.2566872835159302, 'max_depth': 30, 'reg_lambda': 0.5044683218002319, 'subsample': 0.9179360270500184, 'colsample_bytree': 0.8264171481132507, 'colsample_bylevel': 0.8274216651916504, 'learning_rate': 0.1098628118634224, 'reg_alpha': 0.38, 'n_estimators': 10}
# # plopt = {'max_delta_step': 2.1036207675933842, 'gamma': 1.104675978422165, 'min_child_weight': 0.17917755246162415, 'max_depth': 30, 'reg_lambda': 1.17205011844635, 'subsample': 0.7731020450592041, 'colsample_bytree': 0.6307834982872009, 'colsample_bylevel': 0.6296542882919312, 'learning_rate': 0.1226506158709526, 'reg_alpha': 0.2, 'n_estimators': 18}
# # pcn['max_depth'] = 18.071641851237906
# # pcn['n_estimators'] = 12
#
# # x = []
# # y = []
# # for i in range(0,20):
# #     value = i * 0.05 + 0.2
# #     plopt['reg_alpha'] = value
# #     print(value)
# #     accu = handle(X_train,Y_train,X_test,Y_test,norm(plopt),len(set(Y_train)))
# #     print(accu)
# #     x.append(value)
# #     y.append(accu)
# # plt.plot(x,y)
# # plt.show()
#
# # pcn['n_estimators'] = 129.93553834908928
# # plopt['n_estimators'] = 300
# # pb['n_estimators'] = 14
# print(handle(X_train,Y_train,X_test,Y_test,plopt,len(set(Y_train))))
#
# # pcn = norm(pcn)
# # print(handle(X_train,Y_train,X_test,Y_test,pcn,len(set(Y_train))))
# # print(pcn)
# # print()
# # x = []
# # y = []
#
# # for i in range(1,31):
# #     value = 0 + i * 10
# #     loptans['n_estimators'] = value
# #     x.append(value)
# #     accu = handle(X_train,Y_train,X_test,Y_test,loptans,len(set(Y_train)))
# #     y.append(accu)
# #     print(accu)
# #     print()
# # plt.plot(x,y)
# # plt.show()
#
#
# # x = []
# # y = []
# # for i in range(1,80):
# #     value = 0.5 + i / 10
# #     pcn['gamma'] = value
# #     tmp = accu(pcn, X_train, Y_train)
# #     x.append(value)
# #     y.append(tmp)
# #     print(tmp)
# #     print(pcn)
# #     print()
# # plt.plot(x,y)
# # plt.show()
