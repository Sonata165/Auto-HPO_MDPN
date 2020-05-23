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


def buildCN():
    '''
    Cosntruct a random innitialized CoreNet
    '''
    input_dim_weights = (1001, 200, 1)
    input_dim_bias = (201, 50, 1)
    # input_dim = tuple(input('please input the dimension of the network, without samples'))
    # output_dim = input('please input the dimension of the output')
    input_weights = Input(input_dim_weights)
    input_bias = Input(input_dim_bias)
    w = Conv2D(3, (9, 9), strides=4, activation='relu')(input_weights)
    w = BatchNormalization()(w)
    w = Conv2D(5, (15, 15), strides=2, activation='relu')(w)
    w = BatchNormalization()(w)
    w = Conv2D(3, (15, 15), activation='relu')(w)
    w = BatchNormalization()(w)
    w = Flatten()(w)
    b = Conv2D(3, (7, 7), strides=4, activation='relu')(input_bias)
    b = BatchNormalization()(b)
    b = Conv2D(5, (7, 7), strides=2, activation='relu')(b)
    b = BatchNormalization()(b)
    b = Flatten()(b)
    layer = Concatenate()([w, b])
    layer = Dense(64, activation='tanh')(layer)
    # layer = Reshape((64, -1))(layer)
    # layer = LSTM(32, activation='tanh')(layer)
    # layer = Dense(16, activation='tanh')(layer)
    layer = Dense(11, activation='tanh')(layer)
    model = Model(inputs=[input_weights, input_bias], outputs=[layer])
    return model


def parameter_clean(parameter, range, include_min, include_max, type, delta=1e-5, inf=1e10):
    '''
    Clean hyper-parameters
    :param parameter:
    :param range:
    :param include_min:
    :param include_max:
    :param type:
    :param delta:
    :param inf:
    :return:
    '''
    min = range[0]
    max = range[1]
    if min == 'inf':
        min = -inf
    if max == 'inf':
        max = inf
    p = parameter
    # First adjust range
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
    # Then adjust data type
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


def encode_CN_parameter():
    '''
    Read dataset
    Random initialize CoreNet
    Predict the hyper-parameters
    Test and output
    :return:
    '''
    # Generate a list of file
    path_list = os.listdir('TrainData/')
    result_path = os.listdir('result_Random_ACCU/')
    # Read CoreNet
    path_core_network = 'ckpt.h5'

    # record the filenames
    filename_list = []
    for file in path_list:
        res_file_name = "result" + file[0:-4] + '.txt'
        if res_file_name in result_path:
            continue
        print(file)
        core_net = buildCN()
        # remove '.csv', save filename only
        filename_list.append(file[0:-4])
        # read dataset
        train_data = pd.read_csv('TrainData/' + file)
        test_data = pd.read_csv('TestData/' + file)
        # extract x,y of training set
        y_train = train_data.pop('Label')
        y_train = np.array(y_train)
        x_train = np.array(train_data)
        # extract x,y of testing dataset
        y_test = test_data.pop('Label')
        y_test = np.array(y_test)
        x_test = np.array(test_data)

        start_time = time.time()  # record start time

        # Only normalize training set (x, y) which will be encoded to get the optimized hyper-parameters
        train_data = pd.read_csv('TrainData/' + file)
        train_data = StandardScaler().fit_transform(train_data)
        x_for_encode = np.array(train_data)
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
        # undo normalization
        """
        for ele in np.nditer(predict_params, op_flags=['readwrite']):
            if ele > 0:
                ele[...] = pow(10, ele)
            elif ele < 0:
                ele[...] = -pow(10, -ele)
        """
        predict_params = np.arctanh(predict_params)
        print(predict_params)
        # Generate XGBoost parameter list
        keys = ['max_delta_step', 'gamma', 'min_child_weight', 'max_depth', 'reg_lambda', 'subsample',
                'colsample_bytree', 'colsample_bylevel', 'learning_rate', 'reg_alpha', 'n_estimators']
        dic = {}
        i = 0
        for key in keys:
            p = predict_params[i]
            if key == 'max_delta_step':
                p = parameter_clean(p, (0, 'inf'), True, True, 'int')
            elif key == 'gamma':
                p = parameter_clean(p, (0, 'inf'), True, True, 'float')
            elif key == 'min_child_weight':
                p = parameter_clean(p, (0, 'inf'), True, True, 'int')
            elif key == 'max_depth':
                p = parameter_clean(p, (0, 'inf'), True, True, 'int')
            elif key == 'reg_lambda':
                p = parameter_clean(p, (0, 'inf'), True, True, 'float')
            elif key == 'subsample':
                p = parameter_clean(p, (0, 1), False, True, 'float')
            elif key == 'colsample_bytree':
                p = parameter_clean(p, (0, 1), False, True, 'float')
            elif key == 'colsample_bylevel':
                p = parameter_clean(p, (0, 1), False, True, 'float')
            elif key == 'learning_rate':
                p = parameter_clean(p, (0, 1), True, True, 'float')
            elif key == 'reg_alpha':
                p = parameter_clean(p, (0, 'inf'), True, True, 'float')
            elif key == 'n_estimators':
                # We are uncertain about the range here because API didn't provide information about this
                p = parameter_clean(p, (0, 'inf'), True, True, 'int')
            dic[key] = p
            i += 1
        end_time = time.time()  # record end time
        period = end_time - start_time
        print(dic)
        # pd.DataFrame(dic, columns=keys).to_csv('parameterPredicted/' + file[0:-4] + '.csv')
        # Test with XGBoost
        num = set(y_train)
        n = len(num)
        try:
            ret = handle(x_train, y_train, x_test, y_test, dic, n)
        except:
            print(file + "Error, maybe y_val contains values not in y_train")
            with open('Error.log', 'a') as f:
                f.write('from RandomComp.py --' + file + 'Error, maybe y_val contains values not in y_train\n')
            continue
        # acc_list.append(acc)
        file = "result_Random_ACCU/result" + file[0:-4] + '.txt'
        with open(file, 'w', encoding="utf8") as f:
            f.write(str(ret) + "," + str(period))
            f.write("\n")
    # form = pd.DataFrame(acc_list, columns=path_list)
    # form.to_csv('result_CN_ACCU/CN_acc_result.csv')


if __name__ == '__main__':
    encode_CN_parameter()
