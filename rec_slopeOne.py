# coding=utf-8

import pandas as pd
import numpy as np
import math
import random

'''
nearly done
'''

users = 943 # 1-943
items = 1682 # 1-1682

def load_data():

    train = {}
    test = {}
    train_file = 'ml-100k/ua.base'
    test_file = 'ml-100k/ua.test'

    for line in open(train_file):
        u, i, r, t = map(int, line.strip().split())
        train.setdefault(u,{})
        train[u][i] = float(r)

    for line in open(test_file):
        u, i, r, t = map(int, line.strip().split())
        test.setdefault(u,{})
        test[u][i] = float(r)

    return train, test

def init_parameters(iLen):
    diff = np.zeros((iLen, iLen), dtype=np.double)
    rcount = np.zeros((iLen, iLen), dtype=np.int8)
    return diff, rcount

def get_user_mean(train, u):
    return np.mean(np.array(train[u].values()))


def prediction(u, i, train, diff, rcount, is_weighted = False):
    N = [j for j in train[u].keys() if rcount[i, j] > 0]
    pred = get_user_mean(train, u)

    if N:
        if is_weighted:
            # need to addition
            pass
        else:
            pred += np.mean([diff[i, j] for j in N])
    return pred

def get_rmse(data, diff, rcount, is_weighted = False):
    error_sum = 0
    count = 0.
    for u in train:
        for i in train[u]:
            r_hat = prediction(u, i, train, diff, rcount, is_weighted)
            error_sum += (data[u][i] - r_hat)**2
            count += 1.
    rmse = math.sqrt(error_sum / count)
    return rmse


def slopone(is_weighted = False):
    train, test = load_data()
    diff, rcount = init_parameters(items)
    for u in train:
        for i1 in train[u]:
            for i2 in train[u]:
                if i1 != i2:
                    diff[i1][i2] += train[u][i1] - train[u][i2]
                    rcount[i1][i2] += 1

    # diff /= rcount 对应位置相除，PS：需要保证除数不为0
    none0 = rcount.nonzero()
    # for i, j in zip(none0):
    #     diff[i, j] /= rcount[i, j]
    diff[none0] /= rcount[none0]

    train_rmse = get_rmse(train, diff, rcount, is_weighted)
    test_rmse = get_rmse(test, diff, rcount, is_weighted)

    print(index, 'train_RMSE =', train_rmse, '><', 'test_RMSE =', test_rmse)


if __name__ == '__main__':
    is_weighted = False
    slopone(is_weighted)