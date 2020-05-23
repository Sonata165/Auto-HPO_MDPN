import numpy as np
import keras.backend as bk
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from zoopt import Dimension, Objective, Parameter, Opt
from read_dataset import *
import matplotlib.pyplot as plt
import os

dataset = None
round = 0
EPOCHS_MNIST = 100
EPOCHS_SVHN = 100
BATCH_SIZE = 1024


def main():
    global dataset
    # dataset = read_mnist_data()
    dataset = read_svhn_subset()
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
    # Set optimized target
    obj = Objective(eval, dim)
    # perform optimization
    solution = Opt.min(obj, Parameter(budget=10))
    # print result
    solution.print_solution()

    plt.plot(obj.get_history_bestsofar())
    plt.savefig('figure.png')


def eval(solution):
    '''
    function to be optimized!
    :param solution:
    :return:
    '''
    global round
    x = solution.get_x()
    round += 1
    print("round =", round, x)
    value = evaluate_param_multi_gpu(dataset, x)
    return value[0]


def evaluate_param_multi_gpu(dataset, params, pid):
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

    # check range
    assert isinstance(c1_channel, int) and c1_channel >= 1
    assert isinstance(c1_kernel, int) and c1_kernel >= 1 and c1_kernel <= 28
    # assert isinstance(c1_size2, int) and c1_size2 >= 1
    # assert isinstance(c1_size3, int) and c1_size3 >= 1
    assert isinstance(c2_channel, int) and c2_channel >= 1
    assert isinstance(c2_kernel, int) and c2_kernel >= 1 and c2_kernel <= 28
    # assert isinstance(c2_size2, int) and c2_size2 >= 1
    # assert isinstance(c2_size3, int) and c2_size3 >= 1
    # NOTE：0: max pooling，1: avg pooling
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

    ''' Build Model '''

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

    ''' parallel '''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(pid)
    # n_GPUs = 8
    # model = multi_gpu_model(model, n_GPUs)

    ''' Compile '''
    adam = Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    if x_train[0].shape[0] == 28:
        epochs = EPOCHS_MNIST
    elif x_train[0].shape[0] == 32:
        epochs = EPOCHS_SVHN
    else:
        raise Exception("Wrong IMG size " + str(x_train[0].shape))

    ''' Training and Testing '''

    print('Training ------------')
    model.fit(x_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=0)

    print('Testing ------------')
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print('test loss: ', loss)
    print('test accuracy: ', accuracy)

    bk.clear_session()

    return (loss, accuracy)


def evaluate_param(dataset, params):
    '''
    Evaluate a set of hyper-parameters (19 in all) to see the performance of running CNN on the given dataset
    :param params: hyper-parameter list, len should be 19
    :param dataset: the given dataset
    :return: evaluation value，(loss, accuracy)
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

    # Check hyper-parameter range
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

    ''' Build Model '''

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

    ''' Train and Test '''

    print('Training ------------')
    model.fit(x_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=0)

    print('Testing ------------')
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print('test loss: ', loss)
    print('test accuracy: ', accuracy)

    bk.clear_session()

    return (loss, accuracy)


def search(_dataset):
    '''
    Search the best hyper-paramers for the given dataset Using ZOOpt
    :param _dataset: the given dataset
    :return: (best hyper-parameters，performance of the best hyper-parameters)
    '''
    global dataset
    dataset = _dataset
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
    solution = Opt.min(obj, Parameter(budget=30))
    # print result
    solution.print_solution()

    plt.plot(obj.get_history_bestsofar())
    plt.savefig('figure.png')
    return (solution.get_x(), solution.get_value())


def search1(_dataset):
    return ([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4], 0.2)

# if __name__ == '__main__':
#     main()
