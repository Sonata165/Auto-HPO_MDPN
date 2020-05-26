'''
Data Preprocessing.
Constain the dataset dimention to 1000 at most,
then use Bayesian Optimization to do labeling.
'''
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from bayes_opt import BayesianOptimization

def main():
    # cut()
    label_all()

def cut():
    '''
    Constrain the dimension of datasets in data/data_init to 1000 dims at most.
    Save results to data/data_init/data.
    '''
    def change(p):
        if np.random.rand(1) <= p:
            return True
        return False

    def sample(data, targetfile):
        ans = data[0:1000]
        row_num = len(data)
        for index in range(1000, row_num):
            if change(1000 / (index + 2)):
                rand_index = np.random.randint(0, 1000)
                ans[rand_index] = data[index]
        data = pd.DataFrame(ans)
        string = 'feature'
        columns = [string + str(i + 1) for i in range(999)]
        columns.append('Label')
        data.columns = columns
        data.to_csv(targetfile, index=False)
        return

    sourcePath = 'data/data_init'
    targetPath = 'data/data_init/data'
    id = 1
    import os
    files = os.listdir(sourcePath)
    files.remove('data')
    # files.remove('test.py')

    for file in files:
        if file == 'covtype.csv':
            continue
        data = pd.read_csv(sourcePath + os.sep + file)
        row_num = len(data.values)
        if row_num <= 2000:
            data.to_csv(targetPath + os.sep + str(id) + '.csv', index=False)
            id = id + 1
            continue
        file_num = int(row_num / 1000)
        print(file)
        print(file_num)
        print()
        for i in range(file_num):
            sample(data.values, targetPath + os.sep + str(id) + '.csv')
            id = id + 1
    return

def label_all():
    sourcePath = 'data/data_init/data'
    import os
    files = os.listdir(sourcePath)

    def label(df_train):
        '''
        Label a dataset
        :param df_train: dataset to be labeled, DataFrame
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
            # print(x.shape)
            # print(y.shape)
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
        print(rf_bo.res)

    for file in files:
        if file == 'covtype.csv':
            continue
        data = pd.read_csv(sourcePath + os.sep + file)
        label(data)


if __name__ == '__main__':
    main()