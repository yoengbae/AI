import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


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


name_file = 'data_kmeans.txt'
columns = ['x1', 'x2']
data_in = pd.read_csv(name_file, names=columns, sep=' ')
# data_in.plot(kind='scatter', x='x1', y='x2', color='red')
x1 = np.asarray(data_in['x1'])
x2 = np.asarray(data_in['x2'])
num = len(x1) - 30
mu = np.empty((2, 3))
x = np.empty((2, len(x1)))
x[0, :] = x1
x[1, :] = x2
distance = np.empty(3)
y = np.empty(len(x1))

# mu = np.array([[2, 4, 6], [1, 8, 9]])
for i in range(2):
    for q in range(3):
        mu[i][q] = random.randint(0, 9)

for max_epoch in range(1000):
    mean = np.zeros(2)
    for i in range(num):
        for k in range(3):
            distance[k] = (np.linalg.norm(x[:, i]-mu[:, k]))**2
        y[i] = np.argmin(distance)

    for k in range(3):
        mean = 0
        for i in range(num):
            mean += indicator(y[i], k) * x[:, i]
        mu[:, k] = np.divide(mean, indicatorsum(y, k))

for i in range(num):
    if y[i] == 0:
        plt.plot(x1[i], x2[i], 'ro')
    elif y[i] == 1:
        plt.plot(x1[i], x2[i], 'bo')
    else:
        plt.plot(x1[i], x2[i], 'co')

# def test(num, )
for i in range(num, num+30):
    for k in range(3):
        distance[k] = (np.linalg.norm(x[:, i] - mu[:, k])) ** 2
    y[i] = np.argmin(distance)

for i in range(num, num+30):
    if y[i] == 0:
        plt.plot(x1[i], x2[i], 'r+')
    elif y[i] == 1:
        plt.plot(x1[i], x2[i], 'b+')
    else:
        plt.plot(x1[i], x2[i], 'c+')

plt.show()

"""
if indicatorsum = 0
    hold present
else
    normal
"""