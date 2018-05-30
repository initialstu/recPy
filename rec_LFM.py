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

def init_parameters(train, cap_K):
    P = {}
    Q = {}
    weight = 1. / np.sqrt(cap_K)
    for u in train:
        for i in train[u]:
            if u not in P:
                P[u] = np.random.rand(1, cap_K) * weight
            if i not in Q:
                Q[i] = np.random.rand(1, cap_K) * weight
    return P, Q

def predict(p, q):
    r_hat = np.dot(p, q.T)
    return r_hat

def rmse(errors):
    errors = np.array(errors)
    return np.sqrt(np.mean(np.power(errors, 2)))

def get_rmse(data, P, Q):
    error_sum = 0
    count = 0.
    for u in data:
        for i in data[u]:
            if u in P and i in Q:
                r_hat = predict(P[u], Q[i])
                error_sum += (data[u][i] - r_hat)**2
                count += 1.
    rmse = math.sqrt(error_sum / count)
    return rmse

def LFM(steps, alpha, lamda, cap_K):
    train, test = load_data()
    P, Q = init_parameters(train, cap_K)
    for index in range(steps):
        errors = []
        for u in train:
            for i in train[u]:
                r_hat = predict(P[u], Q[i])
                error = train[u][i] - r_hat
                P[u] += alpha * (error * Q[i] - lamda * P[u])
                Q[i] += alpha * (error * P[u] - lamda * Q[i])
                errors.append(error)
        train_rmse = rmse(errors)
        test_rmse = get_rmse(test, P, Q)
        print(index, 'train_RMSE =', train_rmse, '><', 'test_RMSE =', test_rmse)



if __name__ == '__main__':
    steps = 100
    alpha = 0.01
    lamda = 0.1
    cap_K = 10
    LFM(steps, alpha, lamda, cap_K)