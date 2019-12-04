import numpy as np
import pandas as pd
import random


def k_means(x_train):
    K = 3
    y = np.empty(num)
    mu = np.empty((i_dimen, K))
    distance = np.empty(K)
    mu[:, 0] = x_train[:, 100]
    mu[:, 1] = x_train[:, 1000]
    mu[:, 2] = x_train[:, 3000]

    for max_epoch in range(100):
        mean = np.zeros(len(x_train[:, 0]))
        for i in range(num):
            for p in range(K):
                distance[p] = (np.linalg.norm(x_train[:, i]-mu[:, p]))**2
            y[i] = np.argmin(distance)

        for p in range(K):
            mean = 0
            for i in range(num):
                mean += indicator(y[i], p) * x_train[:, i]
            if indicatorsum(y, p) == 0:
                print(p)
            else:
                mu[:, p] = np.divide(mean, indicatorsum(y, p))

    return mu


def indicatorsum(pre, p):
    s = 0
    for c in range(len(pre)):
        if pre[c] == p:
            s += 1
    return s


def indicator(pre, p):
    if pre == p:
        return 1
    else:
        return 0


data_in = pd.read_csv("grade_students.csv")
x = np.empty((6, data_in.__len__()))
x[0, :] = np.asarray(data_in['g1freelunch'])
x[1, :] = np.asarray(data_in['g1absent'])
x[2, :] = np.asarray(data_in['g1readscore'])
x[3, :] = np.asarray(data_in['g1mathscore'])
x[4, :] = np.asarray(data_in['g1listeningscore'])
x[5, :] = np.asarray(data_in['g1wordscore'])
num = len(x[0])
i_dimen = len(x[:, 0])
mu_ = k_means(x)

print(mu_)


"""분석? how?"""

