import numpy as np

def parameter_clean(parameter, range, include_min, include_max, type, delta=1e-5, inf=1e10):
    '''
    Clean Hyper-parameters
    :param parameter:
    :param range:
    :param include_min:
    :param include_max:
    :param type:
    :param delta:
    :param inf:
    :return:
    '''
    min = range[0]
    max = range[1]
    if min == 'inf':
        min = -inf
    if max == 'inf':
        max = inf
    p = parameter
    # First adjust range
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
    # Then adjust data type
    if type == 'int':
        p = int(p)
    return p

def analyze_feature(feature):
    '''
    flatten weight and bias which is to be the meta-feature
    :param feature: the output of encoder
    :return:
    '''
    ret = []
    for i in range(0, len(feature)):
        if i == 1 or i == 3 or i == 5:
            continue
        t1 = feature[i][0].flatten()
        for j in t1:
            ret.append(j)
        t2 = feature[i][1].flatten()
        for j in t2:
            ret.append(j)
    ret = np.array(ret)
    return ret