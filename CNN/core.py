'''
author:
chargehand: Bozhou Chen
important members: Kaixin Zhang, Longshen Ou, Chenmin Ba
'''
import numpy as np
import sys
import shutil
from math import *
from sklearn.preprocessing import *
# import preprocess  # 这个不用管，能运行
import os
import pandas as pd
import keras
import sys
from keras.layers import *
from keras.models import load_model
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Embedding, LSTM
from keras.callbacks import *
from keras.utils import multi_gpu_model
from keras.layers import CuDNNLSTM

sys.path.append('..')


def normalize():
    '''
    Normalize dataset
    :return:
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


def train(x_train, y, epoch, batch_size):
    '''
    train method
    input:none
    output:none
    describe:
    if nn.h5 is empty, train from beginning,
    else train from nn.h5's model,
    use the data from data_ok's data.csv,
    save the model in nn.h5.
    '''
    # if os.path.exists('ckpt.h5') and os.path.getsize('ckpt.h5') != 0:
    #     model = load_model('ckpt.h5')
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

    # os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
    # model = multi_gpu_model(model, gpus=2)
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=10, mode='min', factor=0.8)
    check_point = ModelCheckpoint(filepath='ckpt.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                  save_weights_only=False, period=5)
    tensor_board = keras.callbacks.TensorBoard(log_dir='./tensor_board_logs', write_grads=True, write_graph=True,
                                               write_images=True)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.RMSprop(0.01, clipnorm=1.0))
    model.summary()
    model.fit(x=x_train, y=y, epochs=epoch, validation_split=0.1, verbose=1, shuffle=True, batch_size=batch_size,
              callbacks=[reduce_lr, check_point, tensor_board])
    return


def predict(vector):
    '''
    predict method
    input:vector is the feature vector of some dataset
    output:params are the predicted best params
    describe:
    use the model in nn.h5 to predict.
    '''
    model = load_model('nn.h5')
    return model.predict(vector)


inputpath = 'data_encoded/'
outputpath = 'data_ok/'
import pickle

x_temp = []
x_train1 = None
x_train2 = None
y = []
exislis = ['100.csv', '101.csv', '102.csv', '103.csv', '104.csv', '105.csv', '106.csv', '107.csv', '108.csv', '109.csv',
           '11.csv', '110.csv', '111.csv', '112.csv', '113.csv', '114.csv', '115.csv', '116.csv', '117.csv', '118.csv',
           '119.csv', '12.csv', '120.csv', '123.csv', '124.csv', '125.csv', '126.csv', '127.csv', '128.csv', '129.csv',
           '13.csv', '136.csv', '137.csv', '138.csv', '139.csv', '14.csv', '140.csv', '141.csv', '142.csv', '143.csv',
           '144.csv', '145.csv', '146.csv', '147.csv', '148.csv', '149.csv', '15.csv', '150.csv', '151.csv', '152.csv',
           '153.csv', '154.csv', '155.csv', '156.csv', '157.csv', '158.csv', '159.csv', '16.csv', '160.csv', '161.csv',
           '162.csv', '163.csv', '164.csv', '165.csv', '166.csv', '167.csv', '168.csv', '169.csv', '17.csv', '170.csv',
           '171.csv', '172.csv', '173.csv', '174.csv', '175.csv', '176.csv', '177.csv', '178.csv', '179.csv', '18.csv',
           '180.csv', '181.csv', '182.csv', '183.csv', '184.csv', '185.csv', '186.csv', '187.csv', '188.csv', '189.csv',
           '19.csv', '190.csv', '191.csv', '192.csv', '193.csv', '194.csv', '195.csv', '196.csv', '197.csv', '198.csv',
           '199.csv', '20.csv', '200.csv', '201.csv', '202.csv', '203.csv', '204.csv', '205.csv', '206.csv', '207.csv',
           '208.csv', '209.csv', '21.csv', '210.csv', '211.csv', '212.csv', '213.csv', '214.csv', '215.csv', '216.csv',
           '217.csv', '218.csv', '219.csv', '22.csv', '220.csv', '221.csv', '222.csv', '223.csv', '224.csv', '225.csv',
           '226.csv', '227.csv', '228.csv', '229.csv', '23.csv', '230.csv', '231.csv', '232.csv', '233.csv', '234.csv',
           '235.csv', '236.csv', '237.csv', '24.csv', '25.csv', '251.csv', '252.csv', '253.csv', '255.csv', '256.csv',
           '257.csv', '258.csv', '259.csv', '26.csv', '260.csv', '261.csv', '263.csv', '264.csv', '265.csv', '266.csv',
           '267.csv', '27.csv', '270.csv', '271.csv', '272.csv', '273.csv', '274.csv', '275.csv', '276.csv', '277.csv',
           '278.csv', '279.csv', '280.csv', '281.csv', '282.csv', '283.csv', '284.csv', '285.csv', '286.csv', '287.csv',
           '288.csv', '289.csv', '290.csv', '291.csv', '292.csv', '293.csv', '294.csv', '295.csv', '296.csv', '297.csv',
           '298.csv', '299.csv', '300.csv', '301.csv', '302.csv', '303.csv', '304.csv', '305.csv', '306.csv', '307.csv',
           '308.csv', '309.csv', '310.csv', '311.csv', '312.csv', '313.csv', '314.csv', '315.csv', '316.csv', '317.csv',
           '318.csv', '319.csv', '320.csv', '321.csv', '322.csv', '323.csv', '324.csv', '325.csv', '326.csv', '327.csv',
           '328.csv', '329.csv', '330.csv', '331.csv', '332.csv', '333.csv', '34.csv', '38.csv', '40.csv', '44.csv',
           '45.csv', '46.csv', '47.csv', '48.csv', '49.csv', '50.csv', '51.csv', '52.csv', '53.csv', '54.csv', '55.csv',
           '56.csv', '57.csv', '58.csv', '59.csv', '60.csv', '61.csv', '62.csv', '63.csv', '64.csv', '65.csv', '66.csv',
           '67.csv', '68.csv', '69.csv', '70.csv', '71.csv', '72.csv', '73.csv', '74.csv', '75.csv', '76.csv', '77.csv',
           '78.csv', '79.csv', '8.csv', '80.csv', '81.csv', '82.csv', '83.csv', '84.csv', '85.csv', '86.csv', '87.csv',
           '88.csv', '89.csv', '9.csv', '90.csv', '91.csv', '92.csv', '93.csv', '94.csv', '95.csv', '96.csv', '97.csv',
           '98.csv', '99.csv']
# 2874
for i in range(1, 2874):
    print(i)
    try:
        if i == 2245:
            continue
        if exislis.__contains__(str(i) + '.csv'):
            continue
        file = open(inputpath + str(i) + '.json', 'rb')
        x_temp = pickle.load(file)
        x_temp[0] = x_temp[0].tolist()
        # np.expand_dims(x_temp[0],axis=0)
        x_temp[1] = x_temp[1].tolist()
        # np.expand_dims(x_temp[1], axis=0)
        file.close()
        if x_train1 is None:
            x_train1 = [x_temp[0]]
            x_train2 = [x_temp[1]]
        else:
            # x_train1 = np.concatenate((x_train1,x_temp[0]),axis=0)
            # x_train2 = np.concatenate((x_train2,x_temp[1]),axis=0)
            x_train1.append(x_temp[0])
            x_train2.append(x_temp[1])
        df = pd.read_csv(outputpath + str(i) + 'label.csv').values
        y.append(df[0])
    except:
        file = open('log.txt', 'a')
        file.write(str(i) + ' chuxianyichang')
        file.close()
        # print(str(i) + 'extraction happens!')
        continue
x_train1 = np.array(x_train1)
x_train2 = np.array(x_train2)
# shape1 = x_train1.shape
# x_train1 = x_train1.reshape((shape1[0], -1))
# x_train1 = StandardScaler().fit_transform(x_train1)
# x_train1 = x_train1.reshape(shape1)
# shape2 = x_train2.shape
# x_train2 = x_train2.reshape((shape2[0], -1))
# x_train2 = StandardScaler().fit_transform(x_train2)
# x_train2 = x_train2.reshape(shape2)
y = np.array(y)
# preprocess y
y = np.tanh(y)
"""
for ele in np.nditer(y, op_flags=['readwrite']):
    if ele > 0:
        ele[...] = log10(ele)
    elif ele < 0:
        ele[...] = - log10(-ele)
"""
print(y.shape)
print('begin to train!')
train([x_train1, x_train2], y, int(1e3), 128)
