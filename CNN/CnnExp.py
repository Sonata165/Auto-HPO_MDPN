import pickle as pk
import os
import numpy as np
import time
import keras.backend as K

from CNN.Utils import *
from CNN.DataEncode import *
from CNN.ZooptUtils import *
from CNN.Constants import *


def main():
    '''
    Read dataset
    Read Corenet
    Predict Hyper-parameters
    Test with evaluate_param_multi_gpu
    Output results
    '''

    exp_path = 'data/experiment_data/'
    train_data_path = exp_path + 'train_data/'
    train_data_list = os.listdir(train_data_path)
    test_data_path = exp_path + 'test_data/'

    for file in train_data_list:
        print(file)

        # Read training data
        p = train_data_path + file
        x_train, y_train = read_mat(p, padding=True)

        # Read testing data
        p = test_data_path + file
        x_test, y_test = read_mat(p, padding=True)

        x_train_for_encode = x_train

        # Generate pred hyper-parameters by 'train', then train the CNN
        # Encoding
        start_time = time.time()

        feature = encode_xy((x_train_for_encode, y_train), epochs=ENCODER_EPOCHS, batchsize=ENCODER_BATCH_SIZE)
        feature = analyze_feature(feature)
        feature = feature.reshape((1, feature.shape[0], 1))

        # Use MDPN to predict hyper-parameters
        time_t1 = time.time()
        model = keras.models.load_model('model/trained_mdpn.h5')
        model.summary()
        time_t2 = time.time()
        pred = model.predict(feature)
        K.clear_session()

        # Undo normalization
        pred = np.arctanh(pred)
        pred[:-1] = np.power(0.0001, pred[:, -1])
        for index in range(0, 19):
            if index in [8, 11, 18]:
                continue
            pred[:, index] = np.power(10, pred[:, index])

        # Save hyper-parameters
        pred_param_folder = 'result/predicted_parameters/'
        if not os.path.exists(pred_param_folder):
            os.makedirs(pred_param_folder)
        with open(pred_param_folder + file, 'wb') as f:
            pk.dump(pred, f)

        # Bound the range of hyper-parameters
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
        end_time = time.time()
        total_time = end_time - start_time - (time_t2 - time_t1)

        # Evaluate
        loss, acc = evaluate_param_multi_gpu((x_train, y_train, x_test, y_test), para)
        K.clear_session()

        # Save results
        result_folder = 'result/results_mdpn/'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        with open(result_folder + file + '.txt', 'w') as f:
            f.write(str(acc) + ',' + str(total_time) + ', loss: ' + str(loss))


if __name__ == '__main__':
    main()
