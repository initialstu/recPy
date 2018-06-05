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
        # 以下两行为i->u
        train.setdefault(i,{})
        train[i][u] = float(r)

    for line in open(test_file):
        u, i, r, t = map(int, line.strip().split())
        # 以下两行为i->u
        test.setdefault(i,{})
        test[i][u] = float(r)

    return train, test

def get_rec_result(userid, K, data, weights):
    '''
    得到用户user的推荐rank
    '''
    # 参考推荐系统实践的userCF实现部分
    # TODO: relation rank & get recommendation result
    # 思路：取userid对应的行或列，然后排序取topK即为最相似的K个user，得到user的交互item表，从中剔除与userid有交互的item，计算userid对其他item的感兴趣程度
    pass

def usercf(userid, K):
    train, test = load_data()
    weights = np.zeros((users, users))
    Nu = np.zeros(users)
    # Nu = np.zeros((1, users))
    for i in train:
        for u1 in train[i]:
                Nu[u1-1] += 1
            for u2 in train[i]:
                if u1 != u2:
                    weights[u1-1, u2-1] += 1

    # 余弦相似度
    none0 = zip(weights.nonzero())
    for u in none0:
        (u1, u2) = u
        weights[u1][u2] /= math.sqrt(Nu[u1] * Nu[u2])

    # TODO: calculate train rmse
    # TODO: calculate test tmse

    return get_rec_result(userid, K, train, weights)




if __name__ == '__main__':
    userid = 10
    K = 5
    print usercf(userid, K)