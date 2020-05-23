'''
疑似无用历史代码
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
cnt = 0
lis = []
for i in range(1, 2874):
    if i == 2245:
        continue
    path = "data_init/data/" + str(i) + ".csv"
    dataSet = pd.read_csv(path, encoding='utf8')
    feature_name = []
    columns = []
    for x in dataSet.columns:
        feature_name.append(x)
        columns.append(x)
    print(i)
    feature_name.remove("Label")
    x = dataSet[feature_name]
    row_number = x.shape[0]
    col_number = x.shape[1]
    X = []
    for index in feature_name:
        X.append(list(x[index]))
    x = X
    x = np.array(x).reshape(col_number, row_number).T
    y = np.array(list(dataSet['Label'])).reshape(row_number)
    if row_number >= 1000:
        cnt += 1
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=33)
        fileOne = "TrainData/" + str(i) + ".csv"
        fileTwo = "TestData/" + str(i) + ".csv"
        num = len(y_train)
        y_train = np.array([y_train]).reshape(num, 1)
        num = len(y_val)
        y_val = np.array([y_val]).reshape(num, 1)
        X = np.append(x_train, y_train, axis=1)
        Y = np.append(x_val, y_val, axis=1)
        df = pd.DataFrame(X, columns=columns)
        df.to_csv(fileOne, index=False)
        df = pd.DataFrame(Y, columns=columns)
        df.to_csv(fileTwo, index=False)
    if cnt == 287:
        break

lis = os.listdir("TrainData")
with open("cbz.txt", 'w', encoding="utf8") as file:
    for s in lis:
        file.write(str(s))
        file.write("\n")

