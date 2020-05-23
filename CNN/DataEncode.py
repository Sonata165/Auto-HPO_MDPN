import CNNEncoderTrainer
import pickle as pk
import numpy as np
import keras.backend as bk
from read_dataset import *
import Predict
import time
"""
dataset = pd.read_csv('C:\\Users\\zkx74\\PycharmProjects\\data.csv')
dataset = dataset.drop(['Quality_label','Unnamed: 0'],axis=1)
data = np.array(dataset)
data = np.tanh(data)
data = StandardScaler().fit_transform(data)
encoder = EncoderTrainer.AutoEncoder(input_shape=(data.shape[1],), first_output_shape=(10,), second_output_shape=(2,))
encoder.train(data, epoch=1000, batch_size=256)
weights1, weights2 = encoder.get_feature()
"""


# x_train, y_train, x_test, y_test = read_mnist_subset()
# print(x_train.shape)
# # x_train = np.random.rand(4000,28,28,1)
# encoder = CNNEncoderTrainer.AutoEncoder(input_shape=(28, 28, 1), label_shape=(10,))
# # x = np.ones((1, 28, 28, 1))
# # y = encoder.auto_encoder.predict(x)
# encoder.train([x_train,y_train], epoch=10, batch_size=32)
# # print(y)
# a = 1
flag = 0

def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(flag)
    compute_feature()


def compute_feature():
    DATASET_PATH = '../12.27_dataset/subset/'
    files = os.listdir(DATASET_PATH)
    RESULT_PATH = '../12.27_dataset/feature/'
    reslts = os.listdir(RESULT_PATH)
    for file in files:
        if 'mnist' in file:
            num = int(file[12:-4])
        else:
            num = int(file[11:-4])
        if num % 8 != flag:
            continue
        this_name = file + '.pkl'

        if this_name in reslts:
            continue
        dataset = read_dataset(DATASET_PATH + file, padding=True)


        # Else, do Encoding
        feature = encode(dataset)

        # save as pkl file
        with open('../12.27_dataset/feature/' + file + '.pkl', 'wb') as f:
            pk.dump(feature, f)


def encode(dataset):
    '''
    Encode dataset
    :param dataset: format is [x_train, y_train, x_test, y_test]
    '''
    x_train, y_train, x_test, y_test = dataset
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    print('x shape:', x[0].shape)
    encoder = CNNEncoderTrainer.AutoEncoder(input_shape=x[0].shape, label_shape=(10,))
    encoder.train([x, y], epoch=100, batch_size=32)
    ret = encoder.get_feature()
    bk.clear_session()
    return ret


def encode_xy(dataset,epochs=50,batchsize=32):
    x, y = dataset
    encoder = CNNEncoderTrainer.AutoEncoder(input_shape=x[0].shape, label_shape=(10,))
    encoder.train([x, y], epoch=epochs, batch_size=batchsize)
    ret = encoder.get_feature()
    bk.clear_session()
    return ret
