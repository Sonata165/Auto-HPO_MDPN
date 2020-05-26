from .EncoderTrainer import *
import os
from .Lopt import *
import sys
from .Utils import *

def main(argv):
    '''
    Read dataset and hyper-parameters,
    Use LOPT to optimize,
    Test with handle()
    Output experiment results.
    '''
    # generate list of file paths
    path_list = os.listdir('TrainData/')
    result_path = os.listdir('result_CN_ACCU/')
    # read coreNet
    param_list = []
    filename_list = []  # use to contain filenames
    # read hyper-parameters
    predict_params_data = pd.read_csv('predicted_parameter.csv')
    for file in path_list:
        res_file_name = "result" + file[0:-4] + '.txt'
        if res_file_name in result_path:
            continue
        if int(file[0:-4]) % 4 != int(argv[1]):
            continue
        print(file)
        # remove '.csv', save filenames
        filename_list.append(file[0:-4])
        # read dataset
        train_data = pd.read_csv('TrainData/' + file)
        test_data = pd.read_csv('TestData/' + file)
        # extract x,y of training sets
        y_train = train_data.pop('Label')
        y_train = np.array(y_train)
        x_train = np.array(train_data)
        # extract x,y of testing sets
        y_test = test_data.pop('Label')
        y_test = np.array(y_test)
        x_test = np.array(test_data)
        # generate xgboost hyper-parameter list
        keys = ['max_delta_step', 'gamma', 'min_child_weight', 'max_depth', 'reg_lambda', 'subsample',
                'colsample_bytree', 'colsample_bylevel', 'learning_rate', 'reg_alpha', 'n_estimators']
        predict_params = predict_params_data[file[0:-4]]
        predict_params = np.array(predict_params).tolist()
        dic = {}
        i = 0
        for key in keys:
            p = predict_params[i]
            if key == 'max_delta_step':
                p = parameter_clean(p, (1, 10), True, True, 'float')
            elif key == 'gamma':
                p = parameter_clean(p, (0, 30), True, True, 'float')
            elif key == 'min_child_weight':
                p = parameter_clean(p, (0, 30), True, True, 'float')
            elif key == 'max_depth':
                p = parameter_clean(p, (3, 30), True, True, 'float')
            elif key == 'reg_lambda':
                p = parameter_clean(p, (0, 2), True, True, 'float')
            elif key == 'subsample':
                p = parameter_clean(p, (0.5, 1), False, True, 'float')
            elif key == 'colsample_bytree':
                p = parameter_clean(p, (0.5, 1), False, True, 'float')
            elif key == 'colsample_bylevel':
                p = parameter_clean(p, (0.5, 1), False, True, 'float')
            elif key == 'learning_rate':
                p = parameter_clean(p, (0.01, 0.2), True, True, 'float')
            elif key == 'reg_alpha':
                p = parameter_clean(p, (0, 2), True, True, 'float')
            elif key == 'n_estimators':
                # we are uncertain about the range here
                p = parameter_clean(p, (1, 300), True, True, 'int')
            dic[key] = p
            i += 1
        try:
            with open('CoreNetOld\\Full_Random_float_3log10_RangeControl\\result_CN_ACCU\\result' + file[0:-4] + '.txt', 'r') as f:
                line = f.readline()
                phrase_time = float(line.split(',')[1])
            dic, time = lopt(dic, x_train, y_train)
            temp = []
            for key in keys:
                temp.append(dic[key])
            param_list.append(temp)
            # read time
            period = time + phrase_time
            print(dic)
            # test with xgboost
            num = set(y_train)
            n = len(num)
            ret = handle(x_train, y_train, x_test, y_test, dic, n)
        except:
            print(file + "Error, maybe y_val contains values not in y_train")
            with open('Error.log', 'a') as f:
                f.write('from MdpnExp.py --' + file + 'Error, maybe y_val contains values not in y_train\n')
            continue
        # acc_list.append(acc)
        file = "result_LOPT_ACCU/result" + file[0:-4] + '.txt'
        with open(file, 'w', encoding="utf8") as f:
            f.write(str(ret) + "," + str(period))
            f.write("\n")
    # save output of CoreNet
    print("param_list")
    print(param_list)
    pd.DataFrame(np.array(param_list).T, columns=filename_list).to_csv('predicted_parameter_LOPT.csv')


if __name__ == '__main__':
    main(sys.argv)