import keras
import pickle as pk

from read_dataset import *
from DataEncode import *
from zoopt_test import *
import time


def parameter_clean(parameter, range, include_min, include_max, type, delta=1e-5, inf=1e10):
    min = range[0]
    max = range[1]
    if min == 'inf':
        min = -inf
    if max == 'inf':
        max = inf
    p = parameter
    # First adjust range
    if include_min and p < min:
        p = min
    if not include_min and p <= min:
        if type == 'int':
            p = min + 1
        else:
            p = min + delta
    if include_max and p > max:
        p = max
    if not include_max and p >= max:
        if type == 'int':
            p = max - 1
        else:
            p = max - delta
    # Then adjust data type
    if type == 'int':
        p = int(p)
    return p


def main():
    for i in range(1, 2):
        path = '../12.27_dataset/subset/'
        file = 'mnist_subset' + str(i) + '.mat'
        x, y = read_dataset_xy(path + file, padding=True)
        print(x.shape)
        print(y.shape)
        feature = encode_xy((x, y))
        feature = analyze_feature(feature)
        feature = feature.reshape(1, feature.shape[0], 1)
        if os.path.exists('../12.27_dataset/result/' + file + '.pkl'):
            with open('../12.27_dataset/result/' + file + '.pkl', 'rb') as f:
                label = pk.load(f)
        else:
            dataset = read_dataset(path + file)
            label, result = search(dataset)
        label = np.array(label)
        print(feature.shape)
        print(label.shape)
        model = keras.models.load_model('CNNCoreNet_ckpt.h5')
        pred = model.predict(feature)
        # undo normalization
        pred = np.arctanh(pred)
        pred[:-1] = np.power(0.0001, pred[:, -1])
        for index in range(0, 19):
            if index in [8, 11, 18]:
                continue
            pred[:, index] = np.power(10, pred[:, index])
        print(pred)
        print(label)
        # cut the range
        for index in range(0, 19):
            if index in [2, 3, 6, 7]:
                continue
            elif index == 0:
                pred[index] = parameter_clean(pred[index], (16, 32), True, True, 'int')
            elif index == 1:
                pred[index] = parameter_clean(pred[index], (1, 8), True, True, 'int')
            elif index == 4:
                pred[index] = parameter_clean(pred[index], (16, 32), True, True, 'int')
            elif index == 5:
                pred[index] = parameter_clean(pred[index], (1, 8), True, True, 'int')
            elif index == 8:
                pred[index] = parameter_clean(pred[index], (0, 1), True, True, 'int')
            elif index == 9:
                pred[index] = parameter_clean(pred[index], (1, 8), True, True, 'int')
            elif index == 10:
                pred[index] = parameter_clean(pred[index], (1, 10), True, True, 'int')
            elif index == 11:
                pred[index] = parameter_clean(pred[index], (0, 1), True, True, 'int')
            elif index == 12:
                pred[index] = parameter_clean(pred[index], (1, 8), True, True, 'int')
            elif index == 13:
                pred[index] = parameter_clean(pred[index], (1, 10), True, True, 'int')
            elif index == 14:
                pred[index] = parameter_clean(pred[index], (40, 50), True, True, 'int')
            elif index == 15:
                pred[index] = parameter_clean(pred[index], (30, 40), True, True, 'int')
            elif index == 16:
                pred[index] = parameter_clean(pred[index], (20, 30), True, True, 'int')
            elif index == 17:
                pred[index] = parameter_clean(pred[index], (10, 20), True, True, 'int')
            elif index == 18:
                pred[index] = parameter_clean(pred[index], (0.0001, 0.001), True, True, 'float')
        # Evaluate
        evaluate_param_multi_gpu((), pred)
        if not os.path.exists('Predict'):
            os.makedirs('Predict')
        with open('Predict/pred' + str(i) + '.pkl', 'wb') as f:
            pk.dump(pred, f)
            pk.dump(label, f)


def analyze_feature(feature):
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
    main()
