import keras
import pickle as pk
from .ReadDataset import *
from .DataEncode import *
from .ZooptUtils import *
import os
import pandas as pd
import numpy as np
import time
import sys
import keras.backend as K
from .Utils import *

def main(argv):
    '''
    Read dataset
    Read Corenet
    Predict Hyper-parameters
    Test with evaluate_param_multi_gpu
    Output results
    '''
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    pid = int(argv[1])
    skip = argv[2]
    print("pid:" + str(pid))
    # Evaluate
    if pid < 0:
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        pass
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(pid)
    exp_path = '../12.27_dataset/experiment_data/'
    train_data_path = exp_path + 'train_data/'
    train_data_list = os.listdir(train_data_path)
    test_data_path = exp_path + 'test_data/'
    for file in train_data_list:
        result_list = os.listdir('../12.27_dataset/result_CN/')
        if skip == 't' and file + '.txt' in result_list:
            print(file + "already have result, skip")
            continue
        if 'mnist' in file:
            num = int(file[12:-4])
        else:
            num = int(file[11:-4])
        if pid >= 0 and num % 2 != pid:
            print(file + "skip ,num:" + str(num))
            continue
        print(file)
        # Read training data
        with open(train_data_path + file, 'rb') as f:
            train_data = pk.load(f)
        x_train = np.array(train_data['X'])
        y_train = np.array(train_data['y'])
        y_train = keras.utils.to_categorical(y_train)
        # Read testing data
        with open(test_data_path + file, 'rb') as f:
            test_data = pk.load(f)
        x_test = np.array(test_data['X'])
        y_test = np.array(test_data['y'])
        y_test = keras.utils.to_categorical(y_test)
        # Pad MNIST dataset
        if x_train.shape[1] != 32:
            x_train_for_encode = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
            x_train_for_encode = np.concatenate((x_train_for_encode, x_train_for_encode, x_train_for_encode), axis=-1)
        else:
            x_train_for_encode = x_train
        # Generate pred hyper-parameters by 'train', then train the CNN
        # Encoding
        start_time = time.time()
        feature = encode_xy((x_train_for_encode, y_train), epochs=50, batchsize=64)
        feature = analyze_feature(feature)
        feature = feature.reshape(1, feature.shape[0], 1)
        print(feature.shape)
        # Use MDPN to predict hyper-parameters
        model = keras.models.load_model('../12.27_dataset/CNNCoreNet_ckpt.h5')
        pred = model.predict(feature)
        # Undo normalization
        pred = np.arctanh(pred)
        pred[:-1] = np.power(0.0001, pred[:, -1])
        for index in range(0, 19):
            if index in [8, 11, 18]:
                continue
            pred[:, index] = np.power(10, pred[:, index])
        # Save hyper-parameters
        if not os.path.exists('../12.27_dataset/Predicted_parameters/'):
            os.makedirs('../12.27_dataset/Predicted_parameters/')
        with open('../12.27_dataset/Predicted_parameters/' + file, 'wb') as f:
            pk.dump(pred, f)
        # cut the range of hyper-parameters
        pred = np.squeeze(pred)
        para = []
        for index in range(0, 19):
            if index in [2, 3, 6, 7]:
                para.append(pred[index])
                continue
            elif index == 0:
                para.append(parameter_clean(pred[index], (16, 32), True, True, 'int'))
            elif index == 1:
                para.append(parameter_clean(pred[index], (1, 8), True, True, 'int'))
            elif index == 4:
                para.append(parameter_clean(pred[index], (16, 32), True, True, 'int'))
            elif index == 5:
                para.append(parameter_clean(pred[index], (1, 8), True, True, 'int'))
            elif index == 8:
                para.append(parameter_clean(pred[index], (0, 1), True, True, 'int'))
            elif index == 9:
                para.append(parameter_clean(pred[index], (1, 8), True, True, 'int'))
            elif index == 10:
                para.append(parameter_clean(pred[index], (1, 10), True, True, 'int'))
            elif index == 11:
                para.append(parameter_clean(pred[index], (0, 1), True, True, 'int'))
            elif index == 12:
                para.append(parameter_clean(pred[index], (1, 8), True, True, 'int'))
            elif index == 13:
                para.append(parameter_clean(pred[index], (1, 10), True, True, 'int'))
            elif index == 14:
                para.append(parameter_clean(pred[index], (40, 50), True, True, 'int'))
            elif index == 15:
                para.append(parameter_clean(pred[index], (30, 40), True, True, 'int'))
            elif index == 16:
                para.append(parameter_clean(pred[index], (20, 30), True, True, 'int'))
            elif index == 17:
                para.append(parameter_clean(pred[index], (10, 20), True, True, 'int'))
            elif index == 18:
                para.append(parameter_clean(pred[index], (0.0001, 0.001), True, True, 'float'))
        end_time = time.time() - start_time
        print(para)
        # Evaluate
        loss, acc = evaluate_param_multi_gpu((x_train, y_train, x_test, y_test), para, pid)
        K.clear_session()
        # Save results
        if not os.path.exists('../12.27_dataset/result_CN/'):
            os.makedirs('../12.27_dataset/result_CN/')
        with open('../12.27_dataset/result_CN/' + file + '.txt', 'w') as f:
            f.write(str(acc) + ',' + str(end_time) + ',' + str(loss))


def analyze_feature(feature):
    '''
    flatten weight and bias which is to be the meta-feature
    :param feature: the output of encoder
    :return:
    '''
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


if __name__ == '__main__':
    main(sys.argv)
