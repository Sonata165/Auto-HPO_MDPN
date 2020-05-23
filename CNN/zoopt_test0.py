import numpy as np
import keras.backend as bk
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from zoopt import Dimension, Objective, Parameter, Opt
from sklearn.model_selection import train_test_split
import time
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt

import os

global dataset
id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
round = 0
EPOCHS_MNIST = 100
EPOCHS_SVHN = 100
BATCH_SIZE = 1024


def eval(solution):
    '''
    The function to be optimized!
    :param solution:
    :return:
    '''
    global round
    x = solution.get_x()
    round += 1
    print("round =", round, x)
    global dataset
    value = evaluate_param_multi_gpu(dataset, x)
    return value[0]


def evaluate_param_multi_gpu(dataset, params):
    assert len(params) == 19

    x_train, x_test, y_train, y_test = dataset

    c1_channel = params[0]
    c1_kernel = params[1]
    c1_size2 = params[2]  # ？？？
    c1_size3 = params[3]  # ？？？
    c2_channel = params[4]
    c2_kernel = params[5]
    c2_size2 = params[6]  # ？？？
    c2_size3 = params[7]  # ？？？
    p1_type = params[8]  # Pooling Type (max / avg)
    p1_kernel = params[9]  # kernel size
    p1_stride = params[10]  # stride size
    p2_type = params[11]
    p2_kernel = params[12]
    p2_stride = params[13]
    n1 = params[14]  # hidden layer size
    n2 = params[15]
    n3 = params[16]
    n4 = params[17]
    learn_rate = params[18]

    # check range
    assert isinstance(c1_channel, int) and c1_channel >= 1
    assert isinstance(c1_kernel, int) and c1_kernel >= 1 and c1_kernel <= 28
    # assert isinstance(c1_size2, int) and c1_size2 >= 1
    # assert isinstance(c1_size3, int) and c1_size3 >= 1
    assert isinstance(c2_channel, int) and c2_channel >= 1
    assert isinstance(c2_kernel, int) and c2_kernel >= 1 and c2_kernel <= 28
    # assert isinstance(c2_size2, int) and c2_size2 >= 1
    # assert isinstance(c2_size3, int) and c2_size3 >= 1
    # NOTE：0: max，1: avg
    assert isinstance(p1_type, int) and (p1_type == 0 or p1_type == 1)
    assert isinstance(p1_kernel, int) and p1_kernel >= 1 and p1_kernel <= 28
    assert isinstance(p1_stride, int) and p1_stride >= 1
    assert isinstance(p2_type, int) and (p2_type == 0 or p2_type == 1)
    assert isinstance(p2_kernel, int) and p2_kernel >= 1 and p2_kernel <= 28
    assert isinstance(p2_stride, int) and p2_stride >= 1
    assert isinstance(n1, int) and n1 >= 1
    assert isinstance(n2, int) and n2 >= 1
    assert isinstance(n3, int) and n3 >= 1
    assert isinstance(n4, int) and n4 >= 1
    assert isinstance(learn_rate, float) and learn_rate > 0 and learn_rate < 1

    ''' Building Model '''

    model = Sequential()
    # input: 28x28 images with 1 channels -> (28, 28, 1) tensors.
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', input_shape=x_train[0].shape,
                     padding='same'))
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', padding='same'))
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', padding='same'))
    if p1_type == 0:
        model.add(MaxPooling2D(pool_size=p1_kernel, strides=p1_stride, padding='same'))
    elif p1_type == 1:
        model.add(AveragePooling2D(pool_size=p1_kernel, strides=p1_stride, padding='same'))

    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu', padding='same'))
    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu', padding='same'))
    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu', padding='same'))
    if p2_type == 0:
        model.add(MaxPooling2D(pool_size=p2_kernel, strides=p2_stride, padding='same'))
    elif p2_type == 1:
        model.add(AveragePooling2D(pool_size=p2_kernel, strides=p2_stride, padding='same'))

    model.add(Flatten())
    model.add(Dense(n1, activation='relu'))
    model.add(Dense(n2, activation='relu'))
    model.add(Dense(n3, activation='relu'))
    model.add(Dense(n4, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # model.summary()
    ''' Parallel '''

    n_GPUs = 1
    # model = multi_gpu_model(model, n_GPUs)

    ''' Compile '''
    adam = Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    if x_train[0].shape[0] == 28:
        epochs = EPOCHS_MNIST
    elif x_train[0].shape[0] == 32:
        epochs = EPOCHS_SVHN
    else:
        die

    ''' Training and Testing '''
    model.fit(x_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    bk.clear_session()
    return (loss, accuracy)


def evaluate_param(dataset, params):
    '''
    Evaluate the performance of a set of hyper-parameters (19 in all) running on a given dataset
    :param dataset: a tuple(x_train, y_train, x_test, y_test),其中x_train和y_train是来自train数据集的，x_test和y_test是来自test数据集的
    :return accu: the classification accuracy
    '''
    assert len(params) == 19

    x_train, y_train, x_test, y_test = dataset

    c1_channel = params[0]
    c1_kernel = params[1]
    c1_size2 = params[2]  # ？？？
    c1_size3 = params[3]  # ？？？
    c2_channel = params[4]
    c2_kernel = params[5]
    c2_size2 = params[6]  # ？？？
    c2_size3 = params[7]  # ？？？
    p1_type = params[8]  # Pooling Type (max / avg)
    p1_kernel = params[9]  # kernel size
    p1_stride = params[10]  # stride size
    p2_type = params[11]
    p2_kernel = params[12]
    p2_stride = params[13]
    n1 = params[14]  # hidden layer size
    n2 = params[15]
    n3 = params[16]
    n4 = params[17]
    learn_rate = params[18]

    # Check the range of hyper-parameters
    assert isinstance(c1_channel, int) and c1_channel >= 1
    assert isinstance(c1_kernel, int) and c1_kernel >= 1 and c1_kernel <= 28
    # assert isinstance(c1_size2, int) and c1_size2 >= 1
    # assert isinstance(c1_size3, int) and c1_size3 >= 1
    assert isinstance(c2_channel, int) and c2_channel >= 1
    assert isinstance(c2_kernel, int) and c2_kernel >= 1 and c2_kernel <= 28
    # assert isinstance(c2_size2, int) and c2_size2 >= 1
    # assert isinstance(c2_size3, int) and c2_size3 >= 1
    # NOTE: 0: max，1: avg
    assert isinstance(p1_type, int) and (p1_type == 0 or p1_type == 1)
    assert isinstance(p1_kernel, int) and p1_kernel >= 1 and p1_kernel <= 28
    assert isinstance(p1_stride, int) and p1_stride >= 1
    assert isinstance(p2_type, int) and (p2_type == 0 or p2_type == 1)
    assert isinstance(p2_kernel, int) and p2_kernel >= 1 and p2_kernel <= 28
    assert isinstance(p2_stride, int) and p2_stride >= 1
    assert isinstance(n1, int) and n1 >= 1
    assert isinstance(n2, int) and n2 >= 1
    assert isinstance(n3, int) and n3 >= 1
    assert isinstance(n4, int) and n4 >= 1
    assert isinstance(learn_rate, float) and learn_rate > 0 and learn_rate < 1

    ''' Building the Model '''

    model = Sequential()
    # input: 28x28 images with 1 channels -> (28, 28, 1) tensors.
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', input_shape=x_train[0].shape,
                     padding='same'))
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', padding='same'))
    model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', padding='same'))
    if p1_type == 0:
        model.add(MaxPooling2D(pool_size=p1_kernel, strides=p1_stride, padding='same'))
    elif p1_type == 1:
        model.add(AveragePooling2D(pool_size=p1_kernel, strides=p1_stride, padding='same'))

    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu', padding='same'))
    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu', padding='same'))
    model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu', padding='same'))
    if p2_type == 0:
        model.add(MaxPooling2D(pool_size=p2_kernel, strides=p2_stride, padding='same'))
    elif p2_type == 1:
        model.add(AveragePooling2D(pool_size=p2_kernel, strides=p2_stride, padding='same'))

    model.add(Flatten())
    model.add(Dense(n1, activation='relu'))
    model.add(Dense(n2, activation='relu'))
    model.add(Dense(n3, activation='relu'))
    model.add(Dense(n4, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    adam = Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    if x_train[0].shape[0] == 28:
        epochs = EPOCHS_MNIST
    elif x_train[0].shape[0] == 32:
        epochs = EPOCHS_SVHN
    else:
        die

    ''' Training and Testing '''
    model.fit(x_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    bk.clear_session()

    return accuracy


def search(X, Y):
    '''
    Search the best hyper-paramers for the given dataset Using ZOOpt
    :param _dataset: the given dataset
    :return: (best hyper-parameters，performance of the best hyper-parameters)
    '''
    global dataset
    dataset = train_test_split(X, Y, test_size=0.1, random_state=33)
    dim = Dimension(
        19,
        [[16, 32], [1, 8], [1, 1], [1, 1], [16, 32],
         [1, 8], [1, 1], [1, 1], [0, 1], [1, 8],
         [1, 10], [0, 1], [1, 8], [1, 10], [40, 50],
         [30, 40], [20, 30], [10, 20], [0.0001, 0.001]],
        [False, False, False, False, False,
         False, False, False, False, False,
         False, False, False, False, False,
         False, False, False, True]
    )
    obj = Objective(eval, dim)
    # perform optimization
    global round
    round = 0
    start = time.time()
    solution = Opt.min(obj, Parameter(budget=30))
    end = time.time()
    # print result
    solution.print_solution()
    return solution.get_x(), end - start


train_data_path = 'experiment_data/train_data/'
test_data_path = 'experiment_data/test_data/'
param_save_path = 'zoopt/best_params/'
analysis_save_path = 'zoopt/data_analysis/'
files = os.listdir(train_data_path)
i = 0
for file in files:
    if i % 8 != id:
        i += 1
        continue
    i += 1
    ssss = time.time()
    # Construct data
    f = open(train_data_path + file, 'rb')
    obj = pickle.load(f)
    f.close()
    print(file + ' read over!')
    x_train = obj['X']
    y_train = obj['y']
    y_train = np_utils.to_categorical(y_train)
    f = open(test_data_path + file, 'rb')
    obj = pickle.load(f)
    f.close()
    print(file + ' read over!')
    x_test = obj['X']
    y_test = obj['y']
    y_test = np_utils.to_categorical(y_test)

    # Compute the best hyper-parameter
    P, time_consuming = search(x_train, y_train)
    # Save the best hyper-parameter
    df = pd.DataFrame([P])
    df.to_csv(param_save_path + file.replace('subset', '').replace('pkl', 'csv'), index=False)
    # compute accuracy
    accu = evaluate_param((x_train, y_train, x_test, y_test), P)
    # save accu, time
    f = open(analysis_save_path + file.replace('subset', '').replace('pkl', 'txt'), 'w')
    f.write(str(accu) + ',' + str(time_consuming))
    f.close()
    print('Evaluation mectrics save successfully!')
    print(P, accu, time_consuming)
    print('The time of processing one file is ')
    print(time.time() - ssss)
    print()
