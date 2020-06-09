import os
import numpy as np
import scipy.io as sio
import pickle as pk
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

def read_mat(path, padding=False):
    '''
        Read a '.mat' dataset
        :param path: path of dataset
        :param padding: whether to do padding to the MNIST dataset
        :return: x, y
        '''
    dataset = sio.loadmat(path)
    x = dataset['X']
    y = dataset['y']
    y = y.reshape(y.shape[1])
    y = np_utils.to_categorical(y)
    if padding and x.shape[1] != 32:
        x = np.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
        x = np.concatenate((x, x, x), axis=-1)

    return (x, y)

def read_dataset(path, padding=False):
    '''
    Read a '.mat' dataset
    :param path: path of dataset
    :param padding: whether to do padding to the MNIST dataset
    :return: (x_train, y_train, x_test, y_test)
    '''
    dataset = sio.loadmat(path)
    x = dataset['X']
    y = dataset['y']
    y = y.reshape(y.shape[1])

    if padding and x.shape[1] != 32:
        x = np.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
        x = np.concatenate((x, x, x), axis=-1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return (x_train, y_train, x_test, y_test)


def read_feature(filename):
    '''
    Read pkl files that save feature
    :param filename:
    :return: feature  ndarray    mnist: 19806
    '''
    f = open(filename, 'rb')
    feature = pk.load(f)
    ret = []
    for i in range(0, len(feature)):
        if i == 1 or i == 3 or i == 5:
            continue
        t1 = feature[i][0].flatten()
        for j in t1:
            ret.append(j)
        t2 = feature[i][1].flatten()
        for j in t2:
            ret.append(j)
    ret = np.array(ret)
    return ret


def read_label(filename):
    '''
    Read pkl file containing filename
    :param filename:
    :return: label                  mnist: 19
    '''
    f = open(filename, 'rb')
    ret = pk.load(f)
    ret = np.array(ret)
    return ret


def read_feature_and_label(type, FEATHRE_PATH, LABEL_PATH):
    label_files = os.listdir(LABEL_PATH)

    x = []
    y = []
    for file in label_files:
        if type not in file:
            continue
        sample = read_feature(FEATHRE_PATH + file)
        x.append(sample)
        y.append(read_label(LABEL_PATH + file))

    x = np.array(x)
    y = np.array(y)
    return (x, y)
