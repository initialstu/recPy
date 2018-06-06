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

# 参考推荐系统实践的userCF实现部分

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

def get_rec_result(userid, K, train, weights):
    '''
    得到用户user的推荐rank
    '''
    # 参考推荐系统实践的itemCF实现部分55-56页
    rank = {}
    Nu = train[userid]
    for j in Nu:
    	for i, wi in sorted(enumerate(weights[j-1]), key = lambda x:x[1], reverse = True)[:K]:
    		if i not in Nu:
    			if i in rank:
    				rank[i] += wi * Nu[j]
    			else:
    				rank[i] = wi * Nu[j]
    return rank


def itemcf(itemid):
    train, test = load_data()
    weights = np.zeros((items, items))
    Ni = np.zeros(items)
    # Ni = np.zeros((1, items))
    for u in train:
        for i1 in train[u]:
            Ni[i1-1] += 1
            for i2 in train[u]:
                if i1 != i2:
                    weights[i1-1, i2-1] += 1

    # 余弦相似度
    none0 = zip(weights.nonzero())
    for i in none0:
        (i1, i2) = i
        weights[i1][i2] /= math.sqrt(Ni[i1] * Ni[i2])

    # TODO: calculate train rmse
    # TODO: calculate test tmse

    return get_rec_result(userid, K, train, weights)


if __name__ == '__main__':
    userid = 10
    K = 5
    print itemcf(userid, K)