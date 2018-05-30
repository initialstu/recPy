# coding=utf-8

import pandas as pd
import numpy as np
import math
import random

'''
how to realize?
'''

users = 943 # 1-943
items = 1682 # 1-1682

# def load_data():

#     train = {}
#     test = {}
#     train_file = 'ml-100k/ua.base'
#     test_file = 'ml-100k/ua.test'

#     for line in open(train_file):
#         u, i, r, t = map(int, line.strip().split())
#         train.setdefault(u,{})
#         train[u][i] = float(r)

#     for line in open(test_file):
#         u, i, r, t = map(int, line.strip().split())
#         test.setdefault(u,{})
#         test[u][i] = float(r)

#     return train, test

# def init_parameters(train, cap_K):
#     P = {}
#     Q = {}
#     weight = 1. / np.sqrt(cap_K)
#     for u in train:
#         for i in train[u]:
#             if u not in P:
#                 P[u] = np.random.rand(1, cap_K) * weight
#             if i not in Q:
#                 Q[i] = np.random.rand(1, cap_K) * weight
#     return P, Q

# def predict(p, q):
#     r_hat = np.dot(p, q.T)
#     return r_hat

# def rmse(errors):
#     errors = np.array(errors)
#     return np.sqrt(np.mean(np.power(errors, 2)))

# def get_rmse(data, P, Q):
#     error_sum = 0
#     count = 0.
#     for u in data:
#         for i in data[u]:
#             if u in P and i in Q:
#                 r_hat = predict(P[u], Q[i])
#                 error_sum += (data[u][i] - r_hat)**2
#                 count += 1.
#     rmse = math.sqrt(error_sum / count)
#     return rmse

# def NMF(steps, alpha, cap_K):
#     train, test = load_data()
#     P, Q = init_parameters(train, cap_K)
#     for index in range(steps):
#         errors = []
#         for u in train:
#             for i in train[u]:
#                 r_hat = predict(P[u], Q[i])
#                 error = train[u][i] - r_hat
#                 # 速度好慢好慢= =思考如何优化↓
#                 for k in range(cap_K):
#                     P[u][0][k] = max(0, P[u][0][k] + alpha * error * Q[i][0][k])
#                     Q[i][0][k] = max(0, Q[i][0][k] + alpha * error * P[u][0][k])
#                 # 速度好慢好慢= =思考如何优化↑
#                 errors.append(error)
#         train_rmse = rmse(errors)
#         test_rmse = get_rmse(test, P, Q)
#         print(index, 'train_RMSE =', train_rmse, '><', 'test_RMSE =', test_rmse)


# if __name__ == '__main__':
#     steps = 100
#     alpha = 0.01
#     cap_K = 10
#     NMF(steps, alpha, cap_K)



#################################################################################################################################################
# 没太懂如何实现NMF

def load_data():
    R_train = np.zeros((users, items))
    R_test = np.zeros((users, items))
    train = []
    test = []
    train_file = 'ml-100k/ua.base'
    test_file = 'ml-100k/ua.test'

    for line in open(train_file):
        u, i, r, t = map(int, line.strip().split())
        train.append((u-1, i-1, r))
        R_train[u-1][i-1] = float(r)

    for line in open(test_file):
        u, i, r, t = map(int, line.strip().split())
        test.append((u-1, i-1, r))
        R_test[u-1][i-1] = float(r)

    return R_train, R_test, train, test

def init_parameters(cap_K):
    weight = 1. / np.sqrt(cap_K)
    P = np.random.rand(users, cap_K) * weight
    Q = np.random.rand(items, cap_K) * weight
    return P, Q

def rmse(R, P, Q):
    return np.sqrt(np.mean(np.power(R - np.dot(P, Q.T), 2)))

def NMF(steps, cap_K):
    R_train, R_test, train, test = load_data()
    P, Q = init_parameters(cap_K)
    for index in range(steps):
        # 太慢不适用
        t = 0
        for u, i, r in train:
            t += 1
            P[u] = P[u] * np.dot(R_train, Q)[u] / np.dot(P, np.dot(Q.T, Q))[u]
            Q[i] = Q[i] * (np.dot(P.T, R_train) / np.dot(np.dot(P.T, P), Q.T))[:,i]
            if t % 1000 == 0:
                print(t, '-----------nice------------')
        # 太快无法迭代
        # P = P * np.dot(train, Q) / np.dot(P, np.dot(Q.T, Q))
        # Q = Q * (np.dot(P.T, train) / np.dot(np.dot(P.T, P), Q.T)).T
        train_rmse = rmse(R_train, P, Q)
        print(index, 'train_RMSE =', train_rmse)

if __name__ == '__main__':
    steps = 10
    cap_K = 20
    NMF(steps, cap_K)