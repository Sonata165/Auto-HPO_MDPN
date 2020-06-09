import pandas as pd
import xgboost as xgb
import numpy as np
import keras
import keras.backend as K
import os
import keras.models
import tensorflow as tf
import time
import traceback
from math import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from XGBoost.Utils import *
from XGBoost.EncoderTrainer import *
from XGBoost.Constants import *


def main():
    predict_with_untrained_mdpn()


def build_mdpn():
    '''
    Cosntruct a random initialized mdpn
    '''
    input_dim_weights = (1001, 200, 1)
    input_dim_bias = (201, 50, 1)
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
    layer = Dense(256, activation='tanh')(layer)
    layer = Dense(64, activation='tanh')(layer)
    layer = Dense(11, activation='tanh')(layer)
    model = Model(inputs=[input_weights, input_bias], outputs=[layer])
    return model


def predict_with_untrained_mdpn():
    '''
    Read dataset
    Random initialize MDPN
    Predict the hyper-parameters
    Test and output
    '''
    input_folder = 'data/test/after_cutting/'
    result_folder = 'results/random/'
    path_list = os.listdir(input_folder)
    result_path = os.listdir(result_folder)

    mdpn = build_mdpn()
    filename_list = []
    param_list = []
    for file in path_list:
        if '.csv' not in file:
            continue

        filename = file.split('.')[0]
        res_file_name = "result" + filename + '.txt'
        print(file)
        filename_list.append(file[0:-4])

        # read dataset
        train_data = pd.read_csv('data/test/train_clf/' + file)
        test_data = pd.read_csv('data/test/test_clf/' + file)

        # extract x, y of train sets
        y_train = train_data.pop('Label')
        y_train = np.array(y_train)
        x_train = np.array(train_data)

        # extract x, y of test sets
        y_test = test_data.pop('Label')
        y_test = np.array(y_test)
        x_test = np.array(test_data)

        start_time = time.time()  # record start time

        # Normalize the training set of XGBoost
        train_data = pd.read_csv('data/test/train_clf/' + file)
        train_data = np.array(train_data)
        train_data = StandardScaler().fit_transform(train_data)
        x_for_encode = np.array(train_data)

        # encoder
        encoder = AutoEncoder(input_shape=(x_for_encode.shape[1],), first_output_shape=(200,),
                              second_output_shape=(50,))
        encoder.train(x_for_encode, 500, 128)

        # Get predicted hyper-parameters
        v = encoder.get_feature()
        v[0] = np.expand_dims(v[0], axis=0)
        v[1] = np.expand_dims(v[1], axis=0)
        predict_params = mdpn.predict(v)[0]
        K.clear_session()

        # Bound the raw output
        epsilon = 0.999999
        for i in predict_params:
            if i < -epsilon:
                i = -epsilon
            if i > epsilon:
                i = epsilon

        # Undo nomalization
        predict_params = np.arctanh(predict_params)
        predict_params[-1] = pow(10, predict_params[-1])
        predict_params[0] = pow(10, predict_params[0])
        predict_params[3] = pow(10, predict_params[3])

        # Bound the output
        dic = {}
        i = 0
        for key in KEYS:
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
                p = parameter_clean(p, (0, 'inf'), True, True, 'int')
            dic[key] = p
            i += 1

        # Final output
        param_list.append(predict_params.tolist())
        print(predict_params)

        end_time = time.time()  # record end time
        period = end_time - start_time
        print(dic)

        # Test with XGBoost
        num = set(y_train)
        n = len(num)
        try:
            ret = handle(x_train, y_train, x_test, y_test, dic, n)
        except:
            exstr = traceback.format_exc()
            print(exstr)
            print(file + "Error, maybe y_val contains values not in y_train")
            with open('Error.log', 'a') as f:
                f.write('from RandomComp.py --' + file + 'Error, maybe y_val contains values not in y_train\n')
            continue

        # Save the result
        file = result_folder + filename + '.txt'
        with open(file, 'w', encoding="utf8") as f:
            f.write(str(ret) + "," + str(period))
            f.write("\n")

    # Save output of random mdpn
    pd.DataFrame(np.array(param_list).T, columns=filename_list).to_csv('results/predicted_parameters_random_mdpn.csv')

if __name__ == '__main__':
    main()
