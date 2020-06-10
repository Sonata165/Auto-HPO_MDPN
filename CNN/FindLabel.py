from CNN.ReadDataset import *
from CNN.ZooptUtils import search

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
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(flag)
    find_label()


def prt(i):
    for j in range(0, 100):
        print(i)


def find_label():
    '''
    Find optimized hyper-parameters for all datasets using ZOOpt
    '''
    DATASET_PATH = 'data/subset/'
    files = os.listdir(DATASET_PATH)
    RESULT_PATH = 'data/result/'
    results = os.listdir(RESULT_PATH)
    for file in files:
        filename = file.split('.')[0]
        print('********************************************')
        print(file)
        # If already computed, then skip
        dataset = read_dataset(DATASET_PATH + file)
        this_name = filename + '.pkl'
        if this_name in results:
            continue

        # Else, find the optimized hyper-parameters
        param, result = search(dataset)

        # save to pkl file
        f = open('data/result/' + file[:-4] + '.pkl', 'wb')
        pk.dump(param, f)
        pk.dump(result, f)
        f.close()
        print('*********************************************\n')


if __name__ == '__main__':
    main()
