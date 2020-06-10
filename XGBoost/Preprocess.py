'''
Data Preprocessing.
Constain the dataset dimention to 1000 at most,
then use Bayesian Optimization to do labeling.
'''
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle as pk
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from bayes_opt import BayesianOptimization

from XGBoost.EncoderTrainer import AutoEncoder
from XGBoost.Constants import *


def main():
    # cut_training_sets()
    # cut_testing_sets()
    # label_all()
    # encode_all()
    split_datasets()


def __sample(data, targetfile):
    def __change(p):
        if np.random.rand(1) <= p:
            return True
        return False

    ans = data[0:1000]
    row_num = len(data)
    for index in range(1000, row_num):
        if __change(1000 / (index + 2)):
            rand_index = np.random.randint(0, 1000)
            ans[rand_index] = data[index]
    data = pd.DataFrame(ans)
    string = 'feature'
    columns = [string + str(i + 1) for i in range(999)]
    columns.append('Label')
    data.columns = columns
    data.to_csv(targetfile, index=False)


def __pad(data):
    '''
    Constrain the dimension of the datasets to 1000
    If dim > 1000, select 1000 columns;
    if dim < 1000, do 0-padding
    :param data: the dataset to be constrained, DataFrame
    :return: the constrained dataset
    '''
    column_name = []
    for i in range(1, 1000):
        column_name.append('feature' + str(i))
    column_name.append('Label')

    if data.shape[1] < 1000:
        remain = 1000 - data.shape[1]
        for i in range(remain):
            data.insert(0, 'temp_' + str(i), 0)
        data.columns = column_name
    elif data.shape[1] > 1000:
        remain = data.shape[1] - 1000
        arr = data.values
        arr = arr[:, -remain:]
        data = pd.DataFrame(arr)
        data.columns = column_name
    return data


def cut_training_sets():
    '''
    Constrain the dimension of training datasets in data/train/raw to 1000 dims at most.
    Save results to data/train/after_cutting.
    '''

    # Preprocessing for training sets
    sourcePath = 'data/train/raw'
    targetPath = 'data/train/after_cutting'
    files = os.listdir(sourcePath)

    id = 1
    for file in files:
        if '.csv' not in file:
            continue

        data = pd.read_csv(sourcePath + os.sep + file)

        # 0-padding
        data = __pad(data)

        # Sampling
        row_num = len(data.values)  # Get the number of instances in the dataset
        if row_num <= 2000:  # If <= 2000, don't do sampling
            data.to_csv(targetPath + os.sep + str(id) + '.csv', index=False)
            id = id + 1
        else:  # If bigger than 2000, sampling to subsets containing 1000 instances at most
            file_num = int(row_num / 1000)
            print(file)
            print(file_num)
            print()
            for i in range(file_num):
                __sample(data.values, targetPath + os.sep + str(id) + '.csv')
                id = id + 1


def cut_testing_sets():
    '''
    Constrain the dimension of testing datasets in data/test/raw to 1000 dims at most.
    Save results to data/test/after_cutting.
    '''
    sourcePath = 'data/test/raw'
    targetPath = 'data/test/after_cutting'
    files = os.listdir(sourcePath)

    id = 1
    for file in files:
        if '.csv' not in file:
            continue

        data = pd.read_csv(sourcePath + os.sep + file)

        data = __pad(data)

        row_num = len(data.values)  # Get the number of instances in the dataset
        if row_num <= 2000:  # If <= 2000, don't do sampling
            data.to_csv(targetPath + os.sep + str(id) + '.csv', index=False)
            id = id + 1
        else:  # If bigger than 2000, sampling to subsets containing 1000 instances at most
            file_num = int(row_num / 1000)
            print(file)
            print(file_num)
            print()
            for i in range(file_num):
                __sample(data.values, targetPath + os.sep + str(id) + '.csv')
                id = id + 1


def label_all():
    '''
    Label all training datasets in data/train/after_cutting, save results to dataset/labels
    '''

    def label(df_train, id):
        '''
        Label a dataset
        :param df_train: dataset to be labeled, DataFrame
        :param id: the id of the dataset to be labeled
        '''
        feature_name = []
        for x in df_train.columns:
            feature_name.append(x)
        print(feature_name)
        feature_name.remove("Label")

        def rf_cv(max_delta_step, gamma, min_child_weight, max_depth, reg_lambda,
                  subsample, colsample_bytree, colsample_bylevel, learning_rate,
                  reg_alpha, n_estimators):
            # Label
            # RandomForestClassifier
            xgbModel = xgb.XGBClassifier(
                booster='gbtree',
                objective='binary:logistic',
                eval_metric='auc',
                tree_method='exact',
                silent=False,
                n_jobs=4,
                seed=7,
                nthread=4,
                max_delta_step=int(max_delta_step),
                gamma=min(gamma, 1e18),
                min_child_weight=int(min_child_weight),
                max_depth=int(max_depth),
                reg_lambda=min(reg_lambda, 1e5),
                reg_alpha=min(reg_alpha, 1e5),
                subsample=min(subsample, 1),
                colsample_bytree=min(colsample_bytree, 1),
                colsample_bylevel=min(colsample_bylevel, 1),
                learning_rate=min(learning_rate, 0.2),
                n_estimators=int(n_estimators)
            )
            print(n_estimators)
            x = df_train[feature_name]
            row_number = x.shape[0]
            col_number = x.shape[1]
            X = []
            for index in feature_name:
                X.append(list(x[index]))
            x = X
            x = np.array(x).reshape(col_number, row_number).T
            y = np.array(list(df_train['Label'])).reshape(row_number)
            kfold = KFold(n_splits=2, random_state=7)
            val = cross_val_score(xgbModel, x, y, cv=kfold)
            print(val.mean())
            return val.mean()

        Params = {  # define the search space
            'max_delta_step': (1, 10),
            'gamma': (0, 1e5),
            'min_child_weight': (0, 1e5),
            'max_depth': (3, 10),
            'reg_lambda': (1, 1e4),
            'subsample': (0.5, 1),
            'colsample_bytree': (0.5, 1),
            'colsample_bylevel': (0.5, 1),
            'learning_rate': (0.01, 0.2),
            'reg_alpha': (0, 1e4),
            'n_estimators': (0, 1e4)
        }
        rf_bo = BayesianOptimization(
            rf_cv,
            Params
        )
        rf_bo.maximize()

        with open('temp.pkl', 'wb') as f:
            pk.dump(rf_bo.max, f)

        out = pd.Series(rf_bo.max['params'])
        res = pd.Series()
        for key in KEYS:
            res[key] = out[key]
        res.to_csv('data/train/labels/' + str(id) + '.csv')

        print(rf_bo.max['params'])

    sourcePath = 'data/train/raw'

    import os
    files = os.listdir(sourcePath)

    id = 1
    for file in files:
        if '.csv' not in file:
            continue
        data = pd.read_csv(sourcePath + os.sep + file)
        label(data, id)
        id += 1


def encode_all():
    '''
    Encode all datasets in data/train/after_cutting, save results to data/train/meta_features
    '''
    import os
    sourcePath = 'data/train/after_cutting'
    files = os.listdir(sourcePath)
    for file in files:
        if '.csv' not in file:
            continue
        data = pd.read_csv(sourcePath + os.sep + file)
        x_for_encode = np.array(data)
        encoder = AutoEncoder(input_shape=(x_for_encode.shape[1],), first_output_shape=(FIRST_OUTPUT_SIZE,),
                              second_output_shape=(SECOND_OUTPUT_SIZE,))
        encoder.train(x=x_for_encode, epoch=ENCODER_EPOCHS, batch_size=ENCODER_BATCH_SIZE)
        v = encoder.get_feature()  # shape: [(1001, 200, 1), (201, 50, 1)]

        # Save results
        with open('data/train/meta_features/' + file.split('.')[0] + '.pkl', 'wb') as f:
            pk.dump(v, f)


def split_datasets():
    '''
    Split Datasets for testing XGBoost's MDPN into training sets and testing sets
    Raw datasets for XGBoost is in data/test/after_cutting
    Results are saved to data/test/train_clf, data/test/test_clf
    '''
    input_folder = 'data/test/after_cutting/'
    out_folder_train = 'data/test/train_clf/'
    out_folder_test = 'data/test/test_clf/'
    files = os.listdir(input_folder)
    for file in files:
        if '.csv' not in file:
            continue
        data = pd.read_csv(input_folder + file)
        y = data.pop('Label')
        X = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train['Label'] = y_train
        X_test['Label'] = y_test
        X_train.to_csv(out_folder_train + file, index=False)
        X_test.to_csv(out_folder_test + file, index=False)


if __name__ == '__main__':
    main()
