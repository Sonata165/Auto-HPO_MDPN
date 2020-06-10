import os
import sys
import numpy as np
import pandas as pd

from XGBoost.Utils import *
from XGBoost.EncoderTrainer import *
from XGBoost.Constants import *
from XGBoost.Lopt import *


def main(argv):
    mdpn_lopt(argv)


def mdpn_lopt(argv):
    '''
    Read dataset and hyper-parameters,
    Use LOPT to optimize,
    Test with handle()
    Output experiment results.
    '''
    folder = 'data/test/after_cutting/'
    result_folder = 'results/lopt/'
    path_list = os.listdir(folder)
    result_path = os.listdir(result_folder)

    param_list = []
    filename_list = []  # use to contain filenames
    # Read all output of MDPN
    predict_params_data = pd.read_csv('results/predicted_parameters_mdpn.csv')
    for file in path_list:
        if '.csv' not in file:
            continue

        # Remove '.csv', save filenames
        filename = file.split('.')[0]
        filename_list.append(file[0:-4])
        res_file_name = "result" + filename + '.txt'
        print(file)

        # read dataset
        train_data = pd.read_csv('data/test/train_clf/' + file)
        test_data = pd.read_csv('data/test/test_clf/' + file)

        # extract x,y of training sets
        y_train = train_data.pop('Label')
        y_train = np.array(y_train)
        x_train = np.array(train_data)

        # extract x,y of testing sets
        y_test = test_data.pop('Label')
        y_test = np.array(y_test)
        x_test = np.array(test_data)

        # Read MDPN's output, save to 'dic'
        predict_params = predict_params_data[filename]
        predict_params = np.array(predict_params).tolist()
        dic = {}
        i = 0
        for key in KEYS:
            p = predict_params[i]
            dic[key] = p
            i += 1

        try:
            # Read time overhead of MDPN
            with open('results/mdpn/' + filename + '.txt',
                      'r') as f:
                line = f.readline()
                phrase_time = float(line.split(',')[1])

            # Optimize with LOPT, save results to 'param_list'
            dic, time = lopt(dic, x_train, y_train)
            temp = []
            for key in KEYS:
                temp.append(dic[key])
            param_list.append(temp)
            print(dic)

            # Compute total time
            total_time = time + phrase_time

            # test with xgboost
            num = set(y_train)
            n = len(num)
            ret = handle(x_train, y_train, x_test, y_test, dic, n)

            # Save result
            file = result_folder + filename + '.txt'
            with open(file, 'w', encoding="utf8") as f:
                f.write(str(ret) + "," + str(total_time))
                f.write("\n")

        except:
            print(file + "Error, maybe y_val contains values not in y_train")
            with open('Error.log', 'a') as f:
                f.write('from MdpnExp.py --' + file + 'Error, maybe y_val contains values not in y_train\n')
            continue

    # Save output of LOPT
    pd.DataFrame(np.array(param_list).T, columns=filename_list).to_csv('results/predicted_parameters_lopt.csv')


if __name__ == '__main__':
    main(sys.argv)
