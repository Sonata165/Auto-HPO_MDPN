from .EncoderTrainer import *
import keras
import keras.backend as K
import os
import keras.models
from math import *
import time
from .Utils import *


def main():
    mdpn_predict()



def mdpn_predict():
    '''
    Read data sets, encode with the trained Encoder, predict optimized hyper-parameters using trained mdpn,
    use handle() to test, output the results.
    '''
    # generate the list of paths
    path_list = os.listdir('TrainData/')
    result_path = os.listdir('result_CN_ACCU/')
    # read coreNet
    path_core_network = 'TrainedMdpn.h5'
    param_list = []
    filename_list = []  # used to save filenames
    for file in path_list:
        res_file_name = "result" + file[0:-4] + '.txt'
        if res_file_name in result_path:
            continue
        print(file)
        core_net = keras.models.load_model(path_core_network)
        filename_list.append(file[0:-4])
        # read dataset
        train_data = pd.read_csv('TrainData/' + file)
        test_data = pd.read_csv('TestData/' + file)
        # extract x, y of train sets
        y_train = train_data.pop('Label')
        y_train = np.array(y_train)
        x_train = np.array(train_data)
        # extract x, y of test sets
        y_test = test_data.pop('Label')
        y_test = np.array(y_test)
        x_test = np.array(test_data)
        start_time = time.time()  # record the start time
        # Normalize the training set of the encoder
        train_data = pd.read_csv('TrainData/' + file)
        train_data = StandardScaler().fit_transform(train_data)
        x_for_encode = np.array(train_data)
        # encoder
        encoder = AutoEncoder(input_shape=(x_for_encode.shape[1],), first_output_shape=(200,),
                              second_output_shape=(50,))
        encoder.train(x_for_encode, 500, 128)
        # get the predicted hyper-parameters
        v = encoder.get_feature()
        v[0] = np.expand_dims(v[0], axis=0)
        v[1] = np.expand_dims(v[1], axis=0)
        predict_params = core_net.predict(v)[0]
        K.clear_session()
        # redo nomalization
        predict_params = np.arctanh(predict_params)
        predict_params[-1] = pow(10, predict_params[-1])
        predict_params[0] = pow(10, predict_params[0])
        predict_params[3] = pow(10, predict_params[3])
        param_list.append(predict_params.tolist())
        print(predict_params)
        # generate xgboost hyper-parameter list
        keys = ['max_delta_step', 'gamma', 'min_child_weight', 'max_depth', 'reg_lambda', 'subsample',
                'colsample_bytree', 'colsample_bylevel', 'learning_rate', 'reg_alpha', 'n_estimators']
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
        end_time = time.time()  # Record running time
        period = end_time - start_time
        print(dic)

        # Test xgboost
        num = set(y_train)
        n = len(num)
        try:
            ret = handle(x_train, y_train, x_test, y_test, dic, n)
        except:
            print(file + 'Error, maybe y_val contains values not in y_train')
            with open('Error.log', 'a') as f:
                f.write('from MdpnExp.py --' + file + 'Error, maybe y_val contains values not in y_train\n')
            continue
        file = "result_CN_ACCU/result" + file[0:-4] + '.txt'
        with open(file, 'w', encoding="utf8") as f:
            f.write(str(ret) + "," + str(period))
            f.write("\n")
    # save output of mdpn
    pd.DataFrame(np.array(param_list).T, columns=filename_list).to_csv('predicted_parameter.csv')


if __name__ == '__main__':
    main()
