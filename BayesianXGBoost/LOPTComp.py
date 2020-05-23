import pandas as pd
import xgboost as xgb
import numpy as np
from EncoderTrainer import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import keras
import keras.backend as K
import os
import keras.models
import tensorflow as tf
from math import *
import time
from final import *
import multiprocessing as mp
import subprocess
from subprocess import PIPE
import sys


def parameter_clean(parameter, range, include_min, include_max, type, delta=1e-5, inf=1e10):
    '''
    Clean hyper-parameters
    '''
    min = range[0]
    max = range[1]
    if min == 'inf':
        min = -inf
    if max == 'inf':
        max = inf
    p = parameter
    # adjust range
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
    # adjust type
    if type == 'int':
        p = int(p)
    return p


def handle(x_train, y_train, x_test, y_test, dic, n):
    '''
    Train with given hyper-parameters and dataset, test with XGBoost
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


def main(argv):
    '''
    Read dataset and hyper-parameters,
    Use LOPT to optimize,
    Test with handle.py
    Output results.
    '''
    # generate list of file paths
    path_list = os.listdir('TrainData/')
    result_path = os.listdir('result_CN_ACCU/')
    # read coreNet
    param_list = []
    filename_list = []  # use to contain filenames
    # read hyper-parameters
    predict_params_data = pd.read_csv('predicted_parameter.csv')
    for file in path_list:
        res_file_name = "result" + file[0:-4] + '.txt'
        if res_file_name in result_path:
            continue
        if int(file[0:-4]) % 4 != int(argv[1]):
            continue
        print(file)
        # remove '.csv', save filenames
        filename_list.append(file[0:-4])
        # read dataset
        train_data = pd.read_csv('TrainData/' + file)
        test_data = pd.read_csv('TestData/' + file)
        # extract x,y of training sets
        y_train = train_data.pop('Label')
        y_train = np.array(y_train)
        x_train = np.array(train_data)
        # extract x,y of testing sets
        y_test = test_data.pop('Label')
        y_test = np.array(y_test)
        x_test = np.array(test_data)
        """
        # encoder
        encoder = AutoEncoder(input_shape=(x_for_encode.shape[1],), first_output_shape=(200,),
                              second_output_shape=(50,))
        encoder.train(x_for_encode, 500, 128)
        # get predicted hyper-parameters
        v = encoder.get_feature()
        v[0] = np.expand_dims(v[0], axis=0)
        v[1] = np.expand_dims(v[1], axis=0)
        # shape1 = v[0].shape
        # v[0] = v[0].reshape((shape1[0], -1))
        # v[0] = StandardScaler().fit_transform(v[0])
        # v[0] = v[0].reshape(shape1)
        # shape2 = v[1].shape
        # v[1] = v[1].reshape((shape2[0], -1))
        # v[1] = StandardScaler().fit_transform(v[1])
        # v[1] = v[1].reshape(shape2)
        # print(v)
        predict_params = core_net.predict(v)[0]
        K.clear_session()
        # redo normalization
        
        for ele in np.nditer(predict_params, op_flags=['readwrite']):
            if ele > 0:
                ele[...] = pow(10, ele)
            elif ele < 0:
                ele[...] = -pow(10, -ele)
        """
        """
        predict_params = np.arctanh(predict_params)
        predict_params[-1] = pow(10, predict_params[-1])
        predict_params[0] = pow(10, predict_params[0])
        predict_params[3] = pow(10, predict_params[3])
        param_list.append(predict_params.tolist())
        print(predict_params)
        """
        # generate xgboost hyper-parameter list
        keys = ['max_delta_step', 'gamma', 'min_child_weight', 'max_depth', 'reg_lambda', 'subsample',
                'colsample_bytree', 'colsample_bylevel', 'learning_rate', 'reg_alpha', 'n_estimators']
        predict_params = predict_params_data[file[0:-4]]
        predict_params = np.array(predict_params).tolist()
        dic = {}
        i = 0
        for key in keys:
            p = predict_params[i]
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
                # we are uncertain about the range here
                p = parameter_clean(p, (1, 300), True, True, 'int')
            dic[key] = p
            i += 1
        try:
            with open('CoreNetOld\\Full_Random_float_3log10_RangeControl\\result_CN_ACCU\\result' + file[0:-4] + '.txt', 'r') as f:
                line = f.readline()
                phrase_time = float(line.split(',')[1])
            dic, time = lopt(dic, x_train, y_train)
            temp = []
            for key in keys:
                temp.append(dic[key])
            param_list.append(temp)
            # read time
            period = time + phrase_time
            print(dic)
            # pd.DataFrame(dic, columns=keys).to_csv('parameterPredicted/' + file[0:-4] + '.csv')
            # test with xgboost
            num = set(y_train)
            n = len(num)
            ret = handle(x_train, y_train, x_test, y_test, dic, n)
        except:
            print(file + "Error, maybe y_val contains values not in y_train")
            with open('Error.log', 'a') as f:
                f.write('from CNComp.py --' + file + 'Error, maybe y_val contains values not in y_train\n')
            continue
        # acc_list.append(acc)
        file = "result_LOPT_ACCU/result" + file[0:-4] + '.txt'
        with open(file, 'w', encoding="utf8") as f:
            f.write(str(ret) + "," + str(period))
            f.write("\n")
    # save output of CoreNet
    print("param_list")
    print(param_list)
    pd.DataFrame(np.array(param_list).T, columns=filename_list).to_csv('predicted_parameter_LOPT.csv')


if __name__ == '__main__':
    main(sys.argv)
    # po = mp.Pool(8)
    # p_conn,c_conn = mp.Pipe()
    # predict_params_data = pd.read_csv('predicted_parameter.csv')
    # for i in range(0, 9):
    #     po.apply_async(encode_CN_parameter, (pi,c_conn,predict_params_data))
    # po.close()
    # child_return = p_conn.recv()
    # print(child_return)
    # po.join()
