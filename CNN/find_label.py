import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import pickle as pk

from read_dataset import *
from multiprocessing import Pool
from zoopt_test import search
flag = 0
def main():
    '''
    Read in all the datasets in './datasets/subset/'
    Compute optimized hyper-parameters with ZOOpt
    Save results to './cnn_label.csv'
    Format is as follow:
                param1  param2  ...     param19
    dataset1    a1      b1              s1
    dataset2    a2      b2              s2
        ...
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(flag)
    find_label_single()
    # find_label_local()

def prt(i):
    for j in range(0, 100):
        print(i)

def find_label_local():
    DATASET_PATH = '../12.27_dataset/subset/'
    files = os.listdir(DATASET_PATH)
    RESULT_PATH = '../12.27_dataset/result/'
    reslts = os.listdir(RESULT_PATH)
    for i in range(1, 11):
        index = -1 * i
        file = files[index]
        print('********************************************')
        print(file)
        # If already computed, then skip
        dataset = read_dataset(DATASET_PATH + file)
        this_name = file + '.pkl'
        if this_name in reslts:
            continue

        # Else, find the optimized hyper-parameters
        param, result = search(dataset)

        # save to pkl file
        f = open('../12.27_dataset/result/' + file + '.pkl', 'wb')
        pk.dump(param, f)
        pk.dump(result, f)
        f.close()
        print('*********************************************\n')

def find_label_single():
    '''
    :param pr_id: range: [0, 19]
    :return:
    '''
    DATASET_PATH = '../12.27_dataset/subset/'
    files = os.listdir(DATASET_PATH)
    RESULT_PATH = '../12.27_dataset/result/'
    reslts = os.listdir(RESULT_PATH)
    for file in files:
        print('********************************************')
        print(file)
        if 'mnist' in file:
            num = int(file[12:-4])
        else:
            num = int(file[11:-4])
        if num % 8 != flag:
            continue

        # If already computed, then skip
        this_name = file + '.pkl'
        if this_name in reslts:
            continue
        dataset = read_dataset(DATASET_PATH + file)
        # Else, find the optimized hyper-parameters
        param, result = search(dataset)

        # save to pkl file
        f = open('../12.27_dataset/result/' + file + '.pkl', 'wb')
        pk.dump(param, f)
        pk.dump(result, f)
        f.close()
        print('*********************************************\n')

if __name__ == '__main__':
    main()