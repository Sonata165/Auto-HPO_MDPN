'''
Use MDPN to predict optimized hyper-parameters
'''
import os
import keras.models
from math import *
import time

from XGBoost.EncoderTrainer import *
from XGBoost.Utils import *
from XGBoost.Constants import *


def main():
    mdpn_predict()


def mdpn_predict():
    '''
    Read data sets,
    Encode with the trained Encoder,
    Predict optimized hyper-parameters using trained mdpn,
    Use handle() to test,
    Output the results.
    '''
    input_folder = 'data/test/after_cutting/'
    result_folder = 'results/mdpn/'
    path_list = os.listdir(input_folder)
    result_path = os.listdir(result_folder)

    # Read pre-trained MDPN
    MDPN_path = 'TrainedMdpn.h5'
    # MDPN_path = 'model.h5'
    param_list = []
    filename_list = []  # used to save filenames
    mdpn = keras.models.load_model(MDPN_path)

    for file in path_list:
        if '.csv' not in file:
            continue

        res_file_name = "result" + file[0:-4] + '.txt'
        if res_file_name in result_path:
            continue
        print(file)
        filename_list.append(file[0:-4])

        # read dataset
        train_data = pd.read_csv('data/test/train_clf/' + file)
        test_data = pd.read_csv('data/test/test_clf/' + file)

        # extract x, y of train sets
        y_train = train_data.pop('Label')
        y_train = np.array(y_train)
        x_train = np.array(train_data)

        # extract x, y of test sets
        y_test = test_data.pop('Label')
        y_test = np.array(y_test)
        x_test = np.array(test_data)

        start_time = time.time()  # record the start time

        # Normalize the training set of XGBoost
        train_data = pd.read_csv('data/test/train_clf/' + file)
        train_data = np.array(train_data)
        train_data = StandardScaler().fit_transform(train_data)
        x_for_encode = np.array(train_data)

        # Encode the dataset
        encoder = AutoEncoder(input_shape=(x_for_encode.shape[1],), first_output_shape=(200,),
                              second_output_shape=(50,))
        encoder.train(x_for_encode, MDPN_EPOCHS, MDPN_BATCH_SIZE)

        # Get the predicted hyper-parameters
        v = encoder.get_feature()
        v[0] = np.expand_dims(v[0], axis=0)
        v[1] = np.expand_dims(v[1], axis=0)
        predict_params = mdpn.predict(v)[0]
        K.clear_session()

        # Bound the raw output
        epsilon = 0.999999
        for i in predict_params:
            if i < -epsilon:
                i = -epsilon
            if i > epsilon:
                i = epsilon

        # Undo nomalization
        predict_params = np.arctanh(predict_params)
        predict_params[-1] = pow(10, predict_params[-1])
        predict_params[0] = pow(10, predict_params[0])
        predict_params[3] = pow(10, predict_params[3])

        # Bound the output
        dic = {}
        i = 0
        for key in KEYS:
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
                p = parameter_clean(p, (1, 300), True, True, 'int')
            dic[key] = p
            predict_params[i] = p
            i += 1

        # Final output
        param_list.append(predict_params.tolist())
        print(predict_params)

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

        # Save the result
        file = result_folder + file[0:-4] + '.txt'
        with open(file, 'w', encoding="utf8") as f:
            f.write(str(ret) + "," + str(period))
            f.write("\n")

    # Save output of mdpn
    pd.DataFrame(np.array(param_list).T, columns=filename_list).to_csv('results/predicted_parameters_mdpn.csv')


if __name__ == '__main__':
    main()
