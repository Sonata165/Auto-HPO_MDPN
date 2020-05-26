import multiprocessing
import time
import pandas as pd
import xgboost as xgb
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def Bayes_parameter(x_train, y_train, n):
    '''
    Optimize hyperparameter with Bayesian Optimization.
    '''
    func = "multi:softmax"
    # func1 = "mlogloss"
    if n == 2:
        func = "binary:logistic"
        # func1 = "logloss"

    def rf_cv(max_delta_step, gamma, min_child_weight, max_depth, reg_lambda,
              subsample, colsample_bytree, colsample_bylevel, learning_rate,
              reg_alpha, n_estimators):
        xgbModel = xgb.XGBClassifier(
            booster='gbtree',
            objective=func,
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
        val = cross_val_score(xgbModel, x_train, y_train, cv=2)
        return val.mean()

    # Hyper-parameter self define
    Params = {
        'max_delta_step': (1, 10),
        'gamma': (0, 30),
        'min_child_weight': (0, 30),
        'max_depth': (3, 30),
        'reg_lambda': (0, 2),
        'subsample': (0.5, 1),
        'colsample_bytree': (0.5, 1),
        'colsample_bylevel': (0.5, 1),
        'learning_rate': (0.01, 0.2),
        'reg_alpha': (0, 2),
        'n_estimators': (1, 300)
    }
    rf_bo = BayesianOptimization(
        rf_cv,
        Params
    )
    rf_bo.maximize(n_iter=45)
    ret = rf_bo._space.max()
    return ret

def getXY(df_train):
    '''
    Unknown function, maybe witten by Longshen or Bozhou
    '''
    feature_name = []
    for x in df_train.columns:
        feature_name.append(x)
    # print(feature_name)
    feature_name.remove("Label")
    x = df_train[feature_name]
    row_number = x.shape[0]
    col_number = x.shape[1]
    X = []
    for index in feature_name:
        X.append(list(x[index]))
    x = X
    x = np.array(x).reshape(col_number, row_number).T
    y = np.array(list(df_train['Label'])).reshape(row_number)
    return x, y

def test_bayesian_accuracy(x_train, y_train, x_test, y_test, n, file):
    start_time = time.time()  # record the start time
    params = Bayes_parameter(x_train, y_train, n)
    end_time = time.time()  # record the ending time
    period = end_time - start_time
    print('Took %f second' % (period))
    params = params["params"]

    columns = ['max_delta_step', 'gamma', 'min_child_weight', 'max_depth', 'reg_lambda', 'subsample',
               'colsample_bytree', 'colsample_bylevel', 'learning_rate', 'reg_alpha', 'n_estimators']
    lis = []
    for string in columns:
        lis.append(params[string])
    df = pd.DataFrame([params], columns=columns)
    outFile = 'BayesParameters/' + file
    df.to_csv(outFile, index=False)
    print("file:" + file + " is saved,start handle function")
    from .Utils import handle
    ret = handle(x_train, y_train, x_test, y_test, params, n)
    path = "result_XGB_ACCU/result" + file[0:-4] + '.txt'
    with open(path, 'w', encoding="utf8") as f:
        f.write(str(ret) + "," + str(period))
        f.write("\n")


if __name__ == "__main__":
    lis = os.listdir("TrainData")
    result_path = os.listdir('result_XGB_ACCU/')
    cnt = 0
    po = multiprocessing.Pool(6)
    for name in lis:
        if "result" + name[0:-4] + '.txt' in result_path:
            continue
        print(name + ": have done " + str(cnt) + " file.")
        cnt += 1
        dataPath = "TrainData/" + name
        df_train = pd.read_csv(dataPath, encoding="utf8")
        x_train, y_train = getXY(df_train)
        dataPath = "TestData/" + name
        df_train = pd.read_csv(dataPath, encoding="utf8")
        x_test, y_test = getXY(df_train)
        num = set(y_train)
        n = len(num)
        print("The category number is " + str(n))
        po.apply_async(test_bayesian_accuracy, (x_train, y_train, x_test, y_test, n, name,))
    po.close()
    po.join()
