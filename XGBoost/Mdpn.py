import numpy as np
import sys
import shutil
from math import *
from sklearn.preprocessing import *

import os
import pandas as pd
import keras
import sys
import pickle
from keras.layers import *
from keras.models import load_model
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Embedding, LSTM
from keras.callbacks import *
from keras.utils import multi_gpu_model
from keras.layers import CuDNNLSTM

from XGBoost.Constants import *

def main():
    '''
    Train the MDPN
    '''

    inputpath = 'meta_features/'
    outputpath = 'labels/'

    x_temp = []
    x_train1 = None
    x_train2 = None
    y = []
    folder = 'data/train/meta_features/'
    files = os.listdir(folder)
    # 2874
    for file in files:
        print(file)
        with open(folder + file, 'rb') as f:
            # Read meta-features
            x_temp = pickle.load(f)
            x_temp[0] = x_temp[0].tolist()
            x_temp[1] = x_temp[1].tolist()
            if x_train1 is None:
                x_train1 = [x_temp[0]]
                x_train2 = [x_temp[1]]
            else:
                x_train1.append(x_temp[0])
                x_train2.append(x_temp[1])

            # Read labels
            label = pd.read_csv('data/train/labels/' + file.split('.')[0] + '.csv', index_col=0, header=None).values
            label = label.reshape(-1)
            y.append(label)
    x_train1 = np.array(x_train1)
    x_train2 = np.array(x_train2)
    y = np.array(y)

    y[:, -1] = np.log10(y[:, -1])
    y[:, 0] = np.log10(y[:, 0])
    y[:, 3] = np.log10(y[:, 3])
    y = np.tanh(y)
    print(y.shape)
    print('begin to train!')
    train(x_train=[x_train1, x_train2], y_train=y, epoch=MDPN_EPOCHS, batch_size=MDPN_BATCH_SIZE)


def normalize():
    '''
    Normalize dataset
    '''
    df = pd.read_csv('data.csv')
    input_name = []
    for name in df.columns:
        if name.__contains__('input'):
            input_name.append(name)
    stat = df[input_name].describe().transpose()
    mean = stat['mean']
    std = stat['std']
    for i in df[input_name]:
        if std[i] != 0:
            df[i] = (df[i] - mean[i]) / std[i]
        else:
            df[i] = df[i] - mean[i]
    df.to_csv('DDT.csv', index=False)
    return


def train(x_train, y_train, epoch, batch_size):
    '''
    If model.h5 is empty, train from beginning,
    else train from nn.h5's model,
    use the data from labels's data.csv,
    save the model in nn.h5.
    :param x_train: train data
    :param y_train: labels of train data
    :param epoch: epoch of MDPN
    :param batch_size: training batch size of MDPN
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
    layer = Dropout(0.3)(layer)
    layer = BatchNormalization()(layer)
    layer = Dense(64, activation='tanh')(layer)
    layer = Dropout(0.3)(layer)
    layer = BatchNormalization()(layer)
    layer = ELU()(layer)
    layer = Dense(11, activation='tanh')(layer)
    model = Model(inputs=[input_weights, input_bias], outputs=[layer])

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, mode='min', factor=0.8)
    check_point = ModelCheckpoint(filepath='TrainedMdpn.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                  save_weights_only=False, period=5)
    # tensor_board = keras.callbacks.TensorBoard(log_dir='./tensor_board_logs', write_grads=True, write_graph=True,
    #                                            write_images=True)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(0.01, clipnorm=1.0))
    model.summary()
    model.fit(x=x_train, y=y_train, epochs=epoch, validation_split=0.1, verbose=1, shuffle=True, batch_size=batch_size,
              callbacks=[reduce_lr, check_point])
    model.save('model.h5')


def predict(vector):
    '''
    use the model in nn.h5 to predict.
    :param vector: vector is the feature vector of some dataset
    :return: params are the predicted best params
    '''
    model = load_model('nn.h5')
    return model.predict(vector)


if __name__ == '__main__':
    main()
