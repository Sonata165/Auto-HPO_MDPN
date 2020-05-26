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
