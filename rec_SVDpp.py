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
    Y = {}
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
                Y[i] = np.random.rand(1, cap_K) * weight
                bi[i] = 0
    return bu, bi, P, Q, Y

def get_mean(data):
    r_sum = 0
    count = 0
    for u in data:
        for i in data[u]:
            r_sum += data[u][i]
            count += 1
    mean = float(r_sum) / count

    return mean

def get_Nu(train, u):
    return train[u].keys()

def predict(mu, bu, bi, p, q, implicit):
    r_hat = mu + bu + bi + np.dot(q, (p + implicit).T)
    return r_hat

def rmse(errors):
    errors = np.array(errors)
    return np.sqrt(np.mean(np.power(errors, 2)))

def svdpp(steps, alpha, lamda, cap_K):
    train, test = load_data()
    mu = get_mean(train)
    bu, bi, P, Q, Y = init_parameters(train, cap_K)
    for index in range(steps):
        errors = []
        for u in train:
            Nu = get_Nu(train, u)
            sqrt_Nu_len = np.sqrt(len(Nu))
            Yu = np.sum(Y[nu] for nu in Nu)
            implicit = Yu / sqrt_Nu_len
            for i in train[u]:
                r_hat = predict(mu, bu[u], bi[i], P[u], Q[i], implicit)
                error = train[u][i] - r_hat
                bu[u] += alpha * (error - lamda * bu[u])
                bi[i] += alpha * (error - lamda * bi[i])
                P[u] += alpha * (error * Q[i] - lamda * P[u])
                Q[i] += alpha * (error * (P[u] + implicit) - lamda * Q[i])
                Y[i] += alpha * (error * Q[i] / sqrt_Nu_len - lamda * Q[i])
                errors.append(error)
        alpha *= 0.9
        train_rmse = rmse(errors)
        print(index, 'train_RMSE =', train_rmse)


if __name__ == '__main__':
    steps = 100
    # 初始alpha不能过大，否则会出现NaN
    alpha = 0.02
    lamda = 0.1
    cap_K = 10
    svdpp(steps, alpha, lamda, cap_K)