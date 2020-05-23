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

def handle(x_train, y_train, x_test, y_test, dic, n):
    '''
    Train with given hyper-parameters and test data, evaluate with XGBoost accuracy
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
              early_stopping_rounds=20,
              verbose=True)

    ### make prediction for test data
    y_pred = model.predict(x_test)

    ### model evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print("accuarcy: %.2f%%" % (accuracy * 100.0))
    return accuracy

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
        # This hyper-parameter confine the maximum weight change step of each tree.
        # If it's set  to 0, it means there's no restriction on it.
        # If it's positive, the algorithm will be more "conservative"
        # Usually, this hyper-parameter doesn't need to to be set, but when the number of instances of each class
        # is not balanced, it's helpful for logistic regression
        'max_delta_step': (1, 10),

        # 在树的叶节点进行进一步划分时所需的最小损失的减少值，
        # 在划分时最小损失值大于Gamma时，才会划分这个点，Gamma指定了节点分裂所需的最小损失函数下降值
        # The decrease value of minimum loss when doing further partition in the leaf nodes of the tree.
        # When the minimum loss is larger than Gamma, the leaf node will be partitioned
        'gamma': (0, 30),

        # 子节点的所有样本hessianh的和的最小值，如果在树的划分^H过程中样本的 h之和的最小值小于min_child_weight，
        # 那么不对该结点进行进一步的划分
        # The minimum value of the sum of hessianh of child nodes of all samples
        # If the minimum value of the sum of h of samples is less than min_child_weight,
        # then further partition of this node won't be done
        'min_child_weight': (0, 30),

        # 树的最大深度,增加这个值将使模型更为复杂、并容易过拟合
        # Maximum depth of the tree. The model will be more complex if it increases, which become less likely to
        # become overfitting.
        'max_depth': (3, 30),

        #  权重的L2正则化项
        # L2 regularization of weights
        'reg_lambda': (0, 2),

        # 假设设置为0.5，意味着xgboost将在树的增长之前随机抽取一半的数据用于训练
        # 可以^H防止过拟合
        # 在每次提升^H迭代时都会采样一次
        # If set to 0.5, XGBoost will random sample half of the data for training before the tree's scale increases
        # can avoid overfitting
        # sample once when each increasing ^H epoch
        'subsample': (0.5, 1),

        # 建立每一棵树的时候对样本的列（特征）进行采样,用于建立下一颗树
        # 可以防止过拟合
        # 在每次提升迭代时都会采样一次
        # Sample instance's columns when constructing a tree, which is used to build the next tree
        # can avoid ovefitting
        # sample once when each increasing ^H epoch
        'colsample_bytree': (0.5, 1),

        # 建立每一棵树的时候，对每一层的样本的列（特征）进行采样
        # 可以防止过拟合
        # 在每次提升迭代时都会采样一次
        # Sample instance's columns when constructing a tree, which is used to build the next tree
        # can avoid overfitting
        # sample once when each increasing ^H epoch
        'colsample_bylevel': (0.5, 1),

        # 学习率： 在每次迭代之后，可以直接得到每个叶子结点的权重，而使权值显小来提升模型的鲁棒性，可以防止过拟合
        # Learning rate: after each iteration, can get the weight of each leaf tree directly,
        # make the weight value looks small, to increase the robustness of the model
        # can prevent overfitting
        'learning_rate': (0.01, 0.2),

        # 权重的L1正则化项(与Lasso regression类似)
        # 可以产生稀疏矩阵，加快算法收敛速度
        # L1 regularization term of weights
        # can generate sparse matrix, speed up the convergence
        'reg_alpha': (0, 2),

        # 迭代次数（决策树的数量）
        # Number of iterations, also the number of decision trees
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

def process(x_train, y_train, x_test, y_test, n, file):
    '''
        Unknown function, maybe witten by Longshen or Bozhou
        '''
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
        print("The catelogy number is " + str(n))
        po.apply_async(process, (x_train, y_train, x_test, y_test, n, name,))
    po.close()
    po.join()
