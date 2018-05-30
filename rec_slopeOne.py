# coding=utf-8

import pandas as pd
import numpy as np
import math
import random


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
    if not is_weighted:
        # train_rmse 预测每个训练数据的ratings，与真实ratings计算rmse
        # test_rmse 预测每个测试数据的ratings，与真实ratings计算rmse
    else:
        # train_rmse
        # test_rmse





if __name__ == '__main__':
    is_weighted = False
    slopone(is_weighted)