'''
local optimization
Interface: lopt函数，另外handle函数可以供测试用
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
    The main function of LOPT
    :param P: hyper-parameters, Dict
    :param X: Features of dataset
    :param Y: labels of dataset
    :return: LOPT's output hyper-parameters, and LOPT running time
    '''
    l = 1
    r = len(P)
    allkeys = P.keys()
    num = 1
    for key in allkeys:
        key_to_index[key] = num
        index_to_key[num] = key
        index_to_value[num] = P[key]
        num += 1
    start = time.time()
    nvalue = P['n_estimators']
    acc = 0
    if nvalue > 35:
        P['n_estimators'] = 10
        P = __func(P, X, Y, l, r)
        P['n_estimators'] = 40
        acc = __accu(P, X, Y)
    else:
        P = __func(P, X, Y, l, r)
    value = P['reg_alpha']
    for i in range(41):
        P['reg_alpha'] = i * 0.005
        tmp = __accu(P, X, Y)
        if tmp > acc:
            tmp = acc
            value = P['reg_alpha']
    P['reg_alpha'] = value
    if nvalue > 35:
        P['n_estimators'] = max(nvalue, 70)
    else:
        P = __sMC(P, 'n_estimators', X, Y)
    return P, time.time() - start


def norm(P):
    '''
    Control the hyper-parameters to the reasonable range
    :param P: hyper-parameters, Dict
    :return: hyper-parameters after range control, Dict
    '''
    from .Utils import *
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
            p = parameter_clean(p, (1, 300), True, True, 'int')
        P[key] = p
        P[key] = p
    return P


def __func(P, X, Y, l, r):
    '''
    The core function of LOPT optimization, called recursively
    :param P: Hyper-parameters, Dict
    :param X: Features of dataset
    :param Y: Labels of dataset
    :param l: The leftmost index of this optimization
    :param r: The rightmost index of this optimization
    :return: Hyper-parameters after optimization, Dict
    '''
    print('Recursion')
    print(l, r)
    if l == r:
        P = __MC(P, index_to_key[l], X, Y)
        return P
    if l == r - 1:
        P = __DMC(P, index_to_key[l], index_to_key[r], X, Y)
        return P
    mid = int((l + r) / 2)
    P = __func(P, X, Y, l, mid)
    P = __func(P, X, Y, mid + 1, r)
    return P


def __DMC(P, key_st, key_nd, X, Y):
    '''
    Dual Montain Climbing
    :param P: Hyper-parameters, Dict
    :param key_st:
    :param key_nd:
    :param X: Features of dataset
    :param Y: Labels of dataset
    :return:
    '''
    P = __MC(P, key_st, X, Y)
    P = __MC(P, key_nd, X, Y)
    return P


def __MC(P, key, X, Y):
    '''
    Implementation of Montain Climbing algorithm
    :param P: Hyper-parameters, Dict
    :param key: Name of hyper-parameter
    :param X: Features of dataset
    :param Y: Labels of dataset
    :return:
    '''
    stride = EPSILON
    threshold = EPSILON_threshold
    if infparams.__contains__(key):
        stride = 1
        threshold = 0.1
    if key == 'n_estimators':
        return P
    if key == 'reg_alpha':
        return P
    value_x = P[key]
    value_last_x = INF
    s_value_x = value_x
    print('Start one __MC process')
    print('Optimizing hyperparameter ' + str(key))
    while stride > threshold:
        P[key] = value_x - stride
        P = norm(P)
        a = __accu(P, X, Y)
        print(P)
        print(a)
        print()
        P[key] = value_x
        P = norm(P)
        b = __accu(P, X, Y)
        print(P)
        print(b)
        print()
        P[key] = value_x + stride
        P = norm(P)
        c = __accu(P, X, Y)
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
    print('One __MC process finish')
    return P


def __sMC(P, key, X, Y):
    '''
    Similar to __MC, besides, contains some special handling to some certain hyper-parameters
    :param P: Hyper-parameters, Dict
    :param key: Name of the hyper-parameter
    :param X: Features of dataset
    :param Y: Labels of dataset
    :return:
    '''
    stride = 2
    threshold = 0.2
    value_x = P[key]
    value_last_x = INF
    s_value_x = value_x
    print('Start a __MC process')
    print('Optimizing hyperparameter ' + str(key))
    while stride > threshold:
        P[key] = value_x - stride
        P = norm(P)
        a = __accu(P, X, Y)
        print(P)
        print(a)
        print()
        P[key] = value_x
        P = norm(P)
        b = __accu(P, X, Y)
        print(P)
        print(b)
        print()
        P[key] = value_x + stride
        P = norm(P)
        c = __accu(P, X, Y)
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
    print('One __MC process finish')
    return P


def __accu(P, X, Y):
    '''
    Compute accuracy
    :param P: Hyper-parameters, Dict
    :param X: Features of dataset
    :param Y: Labels of dataset
    :return: Accuracy on the valudation set
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