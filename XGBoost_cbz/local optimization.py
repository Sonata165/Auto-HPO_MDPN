'''
cbz
local optimization
对外提供接口为lopt函数，另外handle函数可以供测试用
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
    局部优化的全局调度函数
    :param P: 超参数字典
    :param X: 训练数据-属性
    :param Y: 训练数据-标签
    :return: 优化之后的超参数和优化的时间
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
    '''
    将超参数控制在合理的取值区间
    :param P:参数列表，是一个字典
    :return:返回一个字典，包含处理之后的参数
    '''

    def parameter_clean(parameter, range, include_min, include_max, type, delta=1e-5):
        '''
        对单个超参数进行范围控制
        :param parameter: 目标超参数，浮点数
        :param range: 超参数范围
        :param include_min: 控制区间左侧开闭，bool变量
        :param include_max: 控制区间右侧开闭，bool变量
        :param type: 参数类型，字符串
        :param delta: 如果是开区间，则从边界去掉一个无穷小后使用
        :return:
        '''
        inf = 1e10
        min = range[0]
        max = range[1]
        if min == 'inf':
            min = -inf
        if max == 'inf':
            max = inf
        p = parameter
        # 首先调整范围
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
        # 最后调整数据类型
        if type == 'int':
            p = int(p)
        return p

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
            # 范围存疑，api未介绍
            p = parameter_clean(p, (1, 300), True, True, 'int')
        P[key] = p
        P[key] = p
    return P


def func(P, X, Y, l, r):
    '''
    局部优化的核心处理函数，递归
    :param P: 同上
    :param X: 同上
    :param Y: 同上
    :param l: 此次调整的参数列表的最左参数下标，闭区间
    :param r: ~····················右··············~
    :return: 优化好的参数列表
    '''
    print('递归')
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


def DMC(P, key_st, key_nd, X, Y):
    P = MC(P, key_st, X, Y)
    P = MC(P, key_nd, X, Y)
    return P


def MC(P, key, X, Y):
    '''
    一元爬山法的实现
    :param P: 同上
    :param key: 超参数名字
    :param X:同上
    :param Y:同上
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
    print('开始一次MC训练')
    print('优化参数为' + str(key))
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
    print('一次MC训练结束')
    return P


def sMC(P, key, X, Y):
    '''
    与MC结构类似，但是是对某个参数进行的特殊处理
    :param P:
    :param key:
    :param X:
    :param Y:
    :return:
    '''
    stride = 2
    threshold = 0.2
    value_x = P[key]
    value_last_x = INF
    s_value_x = value_x
    print('开始一次MC训练')
    print('优化参数为' + str(key))
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
    print('一次MC训练结束')
    return P


def accu(P, X, Y):
    '''
    计算正确率
    :param P: 同上
    :param X: 同上
    :param Y: 同上
    :return: 验证集上的正确率
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
    '''
    计算一个数据集在一组超参数下的分类正确率
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param dic: 超参数
    :param n: 类别数
    :return:
    '''
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
