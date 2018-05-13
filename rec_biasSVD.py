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


def init_parameters(train, cap_K):
    P = {}
    Q = {}
    bu = {}
    bi = {}
    weight = 1. / np.sqrt(cap_K)
    for u in train:
        for i in train[u]:
            if u not in P:
                P[u] = np.random.rand(1, cap_K) * weight
                bu[u] = 0
            if i not in Q:
                Q[i] = np.random.rand(1, cap_K) * weight
                bi[i] = 0
    return bu, bi, P, Q

# def init_b(train):
#     bu = {}
#     bi = {}
#     for u in train:
#         if u not in bu:
#             bu[u] = 0
#         for i in train[u]:
#             if i not in bi:
#                 bi[i] = 0
#     return bu, bi

def get_mean(data):
    r_sum = 0
    count = 0
    for u in data:
        for i in data[u]:
            r_sum += data[u][i]
            count += 1
    mean = float(r_sum) / count

    return mean

def predict(mu, bu, bi, p, q):
    r_hat = mu + bu + bi + np.dot(p, q.T)
    return r_hat

def rmse(errors):
    errors = np.array(errors)
    return np.sqrt(np.mean(np.power(errors, 2)))

def get_rmse(data, mu, bu, bi, P, Q):
    error_sum = 0
    count = 0.
    for u in data:
        for i in data[u]:
            if u in bu and i in bi:
                r_hat = predict(mu, bu[u], bi[i], P[u], Q[i])
                error_sum += (data[u][i] - r_hat)**2
                count += 1.
    rmse = math.sqrt(error_sum / count)
    return rmse

def biasSVD(steps, alpha, lamda, cap_K):
    train, test = load_data()
    mu = get_mean(train)
    # bu, bi = init_b(train)
    bu, bi, P, Q = init_parameters(train, cap_K)
    for index in range(steps):
        errors = []
        for u in train:
            for i in train[u]:
                r_hat = predict(mu, bu[u], bi[i], P[u], Q[i])
                error = train[u][i] - r_hat
                bu[u] += alpha * (error - lamda * bu[u])
                bi[i] += alpha * (error - lamda * bi[i])
                P[u] += alpha * (error * Q[i] - lamda * P[u])
                Q[i] += alpha * (error * P[u] - lamda * Q[i])
                errors.append(error)
        train_rmse = rmse(errors)
        test_rmse = get_rmse(test, mu, bu, bi, P, Q)
        print(index, 'train_RMSE =', train_rmse, '><', 'test_RMSE =', test_rmse)
        # if index % 10 == 9:
        #     test_rmse = get_rmse(test, mu, bu, bi, P, Q)
        #     print('><', index, 'test_RMSE =', test_rmse)


if __name__ == '__main__':
    steps = 100
    # 初始alpha不能过大，否则会出现NaN
    alpha = 0.01
    lamda = 0.1
    cap_K = 10
    biasSVD(steps, alpha, lamda, cap_K)