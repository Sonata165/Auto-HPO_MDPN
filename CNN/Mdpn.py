import keras
from keras.layers import *
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense
from keras.callbacks import *
from sklearn.utils import shuffle
from math import *

from CNN.ReadDataset import *
from CNN.Constants import *


def main():
    '''
    run build_mdpn
    '''
    GPU = False

    if GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # n_GPUs = 8

    build_mdpn()


def build_mdpn():
    '''
    Read training set and shuffle
    Normalize dataset
    Build NN and train it
    Output CoreNet
    '''
    feature_path = 'data/feature/'
    label_path = 'data/result/'
    # Read two types of datasets
    (x, y) = read_feature_and_label('mnist', feature_path, label_path)
    (x_svhn, y_svhn) = read_feature_and_label('svhn', feature_path, label_path)
    # Join two types of datasets

    x = np.concatenate((x, x_svhn), axis=0)
    y = np.concatenate((y, y_svhn), axis=0)
    #
    # x = x_svhn
    # y = y_svhn

    # shuffle
    x, y = shuffle(x, y, random_state=33)
    x = np.expand_dims(x, axis=-1)

    y[:, -1] = np.log10(y[:, -1]) / log10(0.0001)
    for index in range(0, 19):
        if index in [8, 11, 18]:
            continue
        y[:, index] = np.log10(y[:, index])
    y = np.tanh(y)

    train(x, y)


def train(x_train, y):
    '''
    train method
    input:none
    output:none
    describe:
    if nn.h5 is empty, train from beginning,
    else train from nn.h5's model,
    use the data from labels's data.csv,
    save the model in nn.h5.
    '''
    input_dim = (20094, 1)
    input_ = Input(input_dim)
    w = Conv1D(4, 16, strides=4, bias_regularizer=keras.regularizers.l1_l2())(input_)
    w = BatchNormalization()(w)
    w = Activation('tanh')(w)
    w = Conv1D(8, 16, strides=4, bias_regularizer=keras.regularizers.l1_l2())(w)
    w = BatchNormalization()(w)
    w = Activation('tanh')(w)
    w = Conv1D(8, 8, strides=2, bias_regularizer=keras.regularizers.l1_l2())(w)
    w = BatchNormalization()(w)
    w = Activation('tanh')(w)
    w = Conv1D(2, 3, strides=1, bias_regularizer=keras.regularizers.l1_l2())(w)
    w = BatchNormalization()(w)
    w = Activation('tanh')(w)
    w = Flatten()(w)
    layer = Dense(64, activation='tanh')(w)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.3)(layer)
    layer = Dense(32, activation='tanh')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.3)(layer)
    layer = Dense(19, activation='tanh')(layer)
    model = Model(inputs=[input_], outputs=[layer])

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, mode='min', factor=0.5)
    check_point = ModelCheckpoint(filepath='model/CNNCoreNet_ckpt.h5',
                                  monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

    # if GPU:
    #     ''' 并行 '''
    #     model = multi_gpu_model(model, n_GPUs)

    model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(0.001, clipnorm=1.0))
    model.summary()
    model.fit(x=x_train, y=y, epochs=MDPN_EPOCHS, validation_split=0.1, verbose=1, shuffle=True,
              batch_size=MDPN_BATCH_SIZE,
              callbacks=[reduce_lr, check_point, early_stop])
    model.save('model/CNNCoreNet1.h5')
    return


if __name__ == '__main__':
    main()
