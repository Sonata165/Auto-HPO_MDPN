import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



def parameter_clean(parameter, range, include_min, include_max, type, delta=1e-5, inf=1e10):
    '''
    Control the range of a certain hyper-parameter
    :param parameter: the hyper-parameter whose range is to be controlled
    :param range: the range of that hyper-parameter
    :param include_min: if True, include minimum range
    :param include_max: if True, include maximum range
    :param type: the type of the hyper-parameter, String
    :param delta: if the boundary value of the range is exclusive, eg. (a, b), then
                    we treat it as [a + delta, b - delta]
    :return: the hyper-parameter whose range is controlled
    '''
    min = range[0]
    max = range[1]
    if min == 'inf':
        min = -inf
    if max == 'inf':
        max = inf
    p = parameter
    # First, adjust the range of parameters
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
    # Then adjust the type
    if type == 'int':
        p = int(p)
    return p


def handle(x_train, y_train, x_test, y_test, dic, n):
    '''
    Compute the classification accuracy of XGBoost running with a given set of hyper-parameters
    :param x_train: features of training set
    :param y_train: labels of training set
    :param x_test: features of testing set
    :param y_test: labels of testing set
    :param dic: a given set of hyper-parameters
    :param n: number of classes
    :return: classification accuracy
    '''
    func = "multi:softmax"
    func1 = "mlogloss"
    if n == 2:
        func = "binary:logitraw"
        func1 = "logloss"

    model = xgb.XGBClassifier(
        booster='gbtree',
        objective=func,
        eval_metric='auc',
        tree_method='exact',
        silent=False,
        n_jobs=4,
        seed=7,
        nthread=4,
        max_delta_step=int(dic["max_delta_step"]),
        gamma=dic["gamma"],
        min_child_weight=int(dic["min_child_weight"]),
        max_depth=int(dic["max_depth"]),
        reg_lambda=dic["reg_lambda"],
        reg_alpha=dic["reg_alpha"],
        subsample=dic["subsample"],
        colsample_bytree=dic["colsample_bytree"],
        colsample_bylevel=dic["colsample_bylevel"],
        learning_rate=dic["learning_rate"],
        n_estimators=int(dic["n_estimators"]),
    )

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=33)

    model.fit(x_train,
              y_train,
              eval_set=[(x_val, y_val)],
              eval_metric=func1,
              verbose=True)

    ### make prediction for test data
    y_pred = model.predict(x_test)

    ### model evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print("accuarcy: %.2f%%" % (accuracy * 100.0))
    return accuracy
