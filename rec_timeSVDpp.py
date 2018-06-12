# coding=utf-8

import pandas as pd
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from collections import OrderedDict
from operator import itemgetter
import struct
import gzip
import matplotlib as mp
from random import randint


'''
参考桌面的timeSVDpp实现方式
'''

users = 943 # 1-943
items = 1682 # 1-1682

def load_data():

    train = {}
    test = {}
    train_file = 'ml-100k/ua.base'
    test_file = 'ml-100k/ua.test'
    timestamp_min = float('inf')
    timestamp_max = -1


    for line in open(train_file):
        u, i, r, t = map(int, line.strip().split())
        timestamp_min = min(timestamp_min, t)
        timestamp_max = max(timestamp_max, t)
        train.setdefault(u,{})
        train[u][i] = (float(r), float(t))

    for u in train:
        for i in train[u]:
            train[u][i][1] = int((train[u][i][1] - timestamp_min)/86400)

    duration = int((timestamp_max - timestamp_min)/86400)

    for line in open(test_file):
        u, i, r, t = map(int, line.strip().split())
        test.setdefault(u,{})
        test[u][i] = float(r)

    return train, test, duration, timestamp_min

def init_parameters():
    gamma1 = 0.005
    gamma2 = 0.007
    gamma3 = 0.001
    g_alpha = 0.00001
    tau1 = 0.005
    tau2 = 0.015
    tau3 = 0.015
    l_alpha = 0.0004
    max_days = duration
    min_days = 0
    timestamp_min = timestamp_min
    

def timesvdpp():
    pass


if __name__ == "__main__":
    timesvdpp()