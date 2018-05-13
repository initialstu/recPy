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

