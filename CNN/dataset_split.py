import random
import scipy.io as sio
import numpy as np
from keras.datasets import mnist
bias = 400
def main():
    split_dataset(name='mnist', subset_num=500, subset_size=80)
    split_dataset(name='svhn', subset_num=500, subset_size=400)

def split_dataset(name, subset_num, subset_size):
    '''
    Do stratification sampling to MNIST dataset
    '''
    if name == 'mnist':
        x_train, y_train, x_test, y_test = read_mnist_data()
    elif name == 'svhn':
        x_train, y_train, x_test, y_test = read_svhn_data()
    else:
        print('invalid name!')
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    # Created inverted index
    dic = {}
    for i in range(len(y)):
        if y[i] in dic:
            dic[y[i]].append(x[i])
        else:
            dic[y[i]] = [x[i]]
    print(dic.keys())
    print(len(dic[0]))
    print(dic[0][0].shape)

    # Find minimum number of instances among all the classes
    min = len(y)
    for i in dic.keys():
        if (len(dic[i]) < min):
            min = len(dic[i])
    print(min)
    # Sample 20 times for MNIST and SVNH (maybe because the result is 6313, 6254)

    for j in range(subset_num): # 20 groups
        print(j)
        temp_x = []
        temp_y = []
        for i in range(subset_size): # 500 times each group
            print(i)
            d = random.randint(0, min-1)
            for k in dic.keys():
                temp_x.append(dic[k][i])
                temp_y.append(k)
        temp_x = np.asarray(temp_x)
        temp_y = np.asarray(temp_y)
        print(temp_x.shape)
        print(temp_y.shape)
        subset = {'X':temp_x, 'y':temp_y}
        sio.savemat('../12.27_dataset/subset/' + name + '_subset' + str(j+bias) + '.mat', subset)

def read_mnist_data():
    '''
    Read and preprocess MNIST dataset
    :return: (x_train, y_train, x_test, y_test)
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = (x_train / 255.).reshape([60000, 28, 28, 1])  # normalize
    x_test = (x_test / 255.).reshape([10000, 28, 28, 1])  # normalize

    return (x_train, y_train, x_test, y_test)

def read_svhn_data():
    '''
    Read and preprocess SVHN dataset
    :return: (x_train, y_train, x_test, y_test)
    '''
    mat1 = sio.loadmat('../12.27_dataset/train_32x32.mat')
    X1 = mat1['X']
    x_train = []
    for i in range(X1.shape[3]):
        x_train.append(X1[:,:,:,i])
    x_train = np.array(x_train)
    Y1 = mat1['y']
    for i in range(len(Y1)):
        if Y1[i] == 10:
            Y1[i] = 0
    y_train = Y1.reshape(Y1.shape[0])

    mat2 = sio.loadmat('../12.27_dataset/test_32x32.mat')
    X2 = mat2['X']
    x_test = []
    for i in range(X2.shape[3]):
        x_test.append(X2[:,:,:,i])
    x_test = np.array(x_test)
    Y2 = mat2['y']
    for i in range(len(Y2)):
        if Y2[i] == 10:
            Y2[i] = 0
    y_test = Y2.reshape(Y2.shape[0])

    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()